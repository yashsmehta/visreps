"""Pin the layer and cross BN source: is the gap from BN stats or from layer selection?"""
import sys, pandas as pd, torch
from dotenv import load_dotenv
load_dotenv()
from omegaconf import OmegaConf
import visreps.utils as utils, visreps.evals as ev, visreps.models.utils as mu
from visreps.dataloaders.neural import load_all_nsd_data, load_all_nsd_synthetic_data, _make_loader
from visreps.models.batchnorm import prepare_eval_batchnorm, training_image_ids

dev = torch.device("cuda")
REGIONS = ["early visual stream", "ventral visual stream"]
SUBJ = list(range(8))
frames = []
for label, ckpt, cfg_id in [("clip32", "/data/ymehta3/clip_pca", 32),
                            ("default1000", "/data/ymehta3/default", 1000)]:
    for bn in ("checkpoint", "nsd"):
        cfg = utils.validate_config(utils.load_config("configs/eval/base.json", [
            "mode=eval", "neural_dataset=nsd_synthetic", "analysis=rsa", f"checkpoint_dir={ckpt}",
            f"cfg_id={cfg_id}", "seed=1", "eval_checkpoint_at_epoch=20", "log_expdata=false",
            "bootstrap=false", "verbose=false", "batchsize=128", "num_workers=8"]))
        cfg = ev._load_cfg(cfg)
        cfg.return_nodes = list(mu.TORCHVISION_RETURN_NODES[cfg.model_name])
        data = load_all_nsd_synthetic_data(cfg, subjects=SUBJ, regions=REGIONS)
        model = mu.load_model(cfg, dev)
        if bn == "nsd":
            nsd_cfg = OmegaConf.merge(cfg, {"neural_dataset": "nsd"})
            nsd = load_all_nsd_data(nsd_cfg, subjects=SUBJ, regions=REGIONS)
            dl = _make_loader(nsd["stimuli"], ev._get_eval_transform(nsd_cfg), cfg.batchsize, cfg.num_workers)
            prepare_eval_batchnorm(model, nsd_cfg, dl, training_image_ids(nsd["neural"], nsd["stimuli"].keys()), dev)
            del nsd, dl
        model = mu.configure_feature_extractor(cfg, model)
        for layer in ("conv4", "fc1"):
            df = ev._reextract_and_score(model, cfg, dev, data["stimuli"], data["test_ids"],
                                         data["neural"], {r: layer for r in REGIONS}, REGIONS, SUBJ)
            df["model"], df["bn"], df["pinned_layer"] = label, bn, layer
            frames.append(df)
            print(f"DONE {label} bn={bn} layer={layer}", flush=True)
        del model; torch.cuda.empty_cache()
pd.concat(frames, ignore_index=True).drop(columns=["layer_selection_scores","bootstrap_scores"], errors="ignore").to_csv(sys.argv[1], index=False)
