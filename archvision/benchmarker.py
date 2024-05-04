import pandas as pd
from benchmarks import NSDBenchmark
from deepjuice.alignment import compute_pearson_rdm
from deepjuice.alignment import get_scoring_method
from deepjuice import *


def load_benchmark(cfg):
    if cfg.benchmark_name.lower() == "nsd":
        benchmark = NSDBenchmark(voxel_set=cfg.region)
        benchmark.build_rdms(compute_pearson_rdm)
        return benchmark
    else:
        raise NotImplementedError(
            f"Benchmarking for {cfg.benchmark_name} is not implemented"
        )


def get_benchmarking_results(
    benchmark,
    feature_extractor,
    layer_index_offset=0,
    metrics=["crsa", "srpr", "ersa"],
    rdm_distance="pearson",
    rsa_distance="pearson",
    score_types=["pearsonr"],
    stack_final_results=True,
    feature_map_stats=None,
    alpha_values=np.logspace(-1, 5, 7).tolist(),
    regression_means=True,
    device="auto",
):

    # use a CUDA-capable device, if available, else: CPU
    if device == "auto":
        device = get_device_id(device)

    # record key information about each method for reference
    method_info = {
        "regression": {"encoding_model": "ridge"},
        "rsa": {"rdm_distance": rdm_distance, "rsa_distance": rsa_distance},
    }

    # initialize an empty list to record scores over layers
    scoresheet_lists = {metric: [] for metric in metrics}

    # if no feature_map_stats provided, make empty dict:
    if feature_map_stats is None:
        feature_map_stats = {}

    # get the voxel (neuroid) indices for each specified roi
    roi_indices = benchmark.get_roi_indices(row_number=True)

    # put all the train-test RDMS on the GPU for fast compute
    target_rdms = apply_tensor_op(benchmark.splithalf_rdms, lambda x: x.to(device))

    # convert the benchmark response_data to a tensor on GPU
    y = convert_to_tensor(benchmark.response_data.to_numpy()).to(
        dtype=torch.float32, device=device
    )

    # split response data into train and test sets
    y = {"train": y[:, ::2].T, "test": y[:, 1::2].T}

    # initialize the regression, in this case ridge regression with LOOCV over alphas
    regression = TorchRidgeGCV(alphas=alpha_values, device=device, scale_X=True)

    # initialize a dictionary of scoring metrics to apply to the predicted outputs
    score_funcs = {
        score_type: get_scoring_method(score_type) for score_type in score_types
    }

    # initialize a single "global SRP matrix" for projection
    srp_kwargs = feature_extractor.get_srp_kwargs(mega_matrix=True)

    # keep one SRP matrix on CPU, + one on GPU:
    srp_matrix = make_srp_matrix(**srp_kwargs)
    srp_matrix_gpu = srp_matrix.clone().to(device)

    layer_index = 0  # keeps track of depth

    # now, we iterate over our extractor
    for feature_maps in tqdm(feature_extractor, desc="Feature Extraction (Batch)"):

        # for each feature_map from the extractor batch, we first need to compute the traintest RDMS
        model_rdms = {
            uid: {
                "train": compute_rdm(
                    feature_map[::2], method=rdm_distance, device=device
                ),
                "test": compute_rdm(
                    feature_map[1::2], method=rdm_distance, device=device
                ),
            }
            for uid, feature_map in tqdm(
                feature_maps.items(), desc="Making RDMs (Layer)"
            )
        }

        # now, we loop over our batch of feature_maps from the extractor...
        # ...starting by defining an iterator that will track our progress
        feature_map_iterator = tqdm(feature_maps.items(), desc="Brain Mapping (Layer)")

        for feature_map_uid, feature_map in feature_map_iterator:
            layer_index += 1  # one layer deeper in feature_maps

            # main data to add to our scoresheet per feature_map
            feature_map_info = {
                "model_layer": feature_map_uid,
                # layer_index_offset is used here in case of subsetting
                "model_layer_index": layer_index + layer_index_offset,
            }

            try:  # attempt to run the sparse random projection on the GPU ...
                srp_kwargs = {"device": device, "srp_matrix": srp_matrix_gpu}
                feature_map = get_feature_map_srps(feature_map, **srp_kwargs)

            except Exception as error:  # run SRP with CPU
                clean_and_sweep()  # clear the CUDA cache
                srp_kwargs = {"device": "cpu", "srp_matrix": srp_matrix}
                feature_map = get_feature_map_srps(feature_map, **srp_kwargs)

            # now, our X Variable: the splithalf feature_map on GPU
            X = feature_map.squeeze().to(torch.float32).to(device)
            X = {"train": X[0::2,], "test": X[1::2, :]}  # splithalf

            # here, we calculate auxiliary stats on our feature_maps
            aux_stats = {
                split: {
                    stat: stat_func(Xi) for stat, stat_func in feature_map_stats.items()
                }
                for split, Xi in X.items()
            }

            regression.fit(
                X["train"], y["train"]
            )  # fit the regression on the train split

            # RidgeGCV gives us both internally generated LOOCV values for the train dataset
            # as well as the ability to predict our test set in the same way as any regressor
            y_pred = {
                "train": regression.cv_y_pred_,
                "test": regression.predict(X["test"]),
            }

            # loop over cRSA, SRPR, eRSA...
            for metric in scoresheet_lists:

                # classical RSA score
                if metric == "crsa":
                    for split in ["train", "test"]:
                        # get the relevant train-test split of the model RDM
                        model_rdm = model_rdms[feature_map_uid][split]
                        for region in benchmark.rdms:
                            for subj_id in benchmark.rdms[region]:
                                # get the relevant train-test split of the brain RDM
                                target_rdm = target_rdms[region][subj_id][split]

                                # compare lower triangles of model + brain RDM
                                # with our specified 2nd-order distance metric
                                score = compare_rdms(
                                    model_rdm, target_rdm, method=rsa_distance
                                )

                                # add the scores to a "scoresheet"
                                scoresheet = {
                                    **feature_map_info,
                                    "region": region,
                                    "subj_id": subj_id,
                                    "cv_split": split,
                                    "score": score,
                                    **aux_stats[split],
                                    **method_info["rsa"],
                                }

                                # append the scoresheet to our running list
                                scoresheet_lists["crsa"].append(scoresheet)

                # encoding model score
                if metric == "srpr":
                    for split in ["train", "test"]:
                        for score_type, score_func in score_funcs.items():
                            # calculate score per neuroid_id with score_type
                            scores = score_func(y[split], y_pred[split])

                            for region in benchmark.rdms:
                                for subj_id in benchmark.rdms[region]:
                                    # get the response_indices for current ROI group
                                    response_indices = roi_indices[region][subj_id]

                                    # average the scores across the response_indices
                                    score = scores[response_indices].mean().item()

                                    # add the scores to a "scoresheet"
                                    scoresheet = {
                                        **feature_map_info,
                                        "region": region,
                                        "subj_id": subj_id,
                                        "cv_split": split,
                                        "score": score,
                                        **aux_stats[split],
                                        **method_info["regression"],
                                    }

                                    # append the scoresheet to our running list
                                    scoresheet_lists["srpr"].append(scoresheet)

                # encoding RSA score
                if metric == "ersa":
                    for split in ["train", "test"]:
                        for region in benchmark.rdms:
                            for subj_id in benchmark.rdms[region]:
                                # get the relevant train-test split of the brain RDM
                                target_rdm = target_rdms[region][subj_id][split]

                                # get the response_indices for current ROI group
                                response_indices = roi_indices[region][subj_id]

                                # get predicted values for each response_index...
                                y_pred_i = y_pred[split][:, response_indices]

                                # ... and use them to calculate the weighted RDM
                                model_rdm = compute_rdm(y_pred_i, rdm_distance)

                                # compare brain-reweighted model RDM to brain RDM
                                # with our specified 2nd-order distance metric...
                                score = compare_rdms(
                                    model_rdm, target_rdm, method=rsa_distance
                                )

                                # add the scores to a "scoresheet"
                                scoresheet = {
                                    **feature_map_info,
                                    "region": region,
                                    "subj_id": subj_id,
                                    "cv_split": split,
                                    "score": score,
                                    **aux_stats[split],
                                    **method_info["rsa"],
                                }

                                # append the scoresheet to our running list
                                scoresheet_lists["ersa"].append(scoresheet)

    # return all the train-test RDMS tn the CPU after use
    apply_tensor_op(benchmark.splithalf_rdms, lambda x: x.to("cpu"))

    # if we don't stack, results are a dictionary of dataframes...
    results = {
        metric: pd.DataFrame(scores) for metric, scores in scoresheet_lists.items()
    }

    if stack_final_results:
        # if we do stack, results are a single concatenated dataframe
        # with only the common_columns of each (excluding method data)
        result_columns = pd.unique(
            [col for results in results.values() for col in results.columns]
        ).tolist()

        common_columns = [
            col
            for col in result_columns
            if all(col in result.columns for result in results.values())
        ]

        common_columns = ["metric"] + common_columns  # indicator

        results_list = []
        for metric, result in results.items():
            result.insert(0, "metric", metric)
            results_list.append(result[common_columns])
            # ...if we do stack, results are a single dataframe

        new_results = pd.concat(results_list)

    return pd.concat(results_list) if stack_final_results else results
