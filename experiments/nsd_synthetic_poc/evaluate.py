"""OOD RSA: regular-NSD-selected layers/BN, reliable synthetic test responses."""
import gc
import argparse
import hashlib
import json
import pickle
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv
from omegaconf import OmegaConf
import pandas as pd
import torch

import visreps.evals as ev
from visreps.dataloaders.neural import load_all_nsd_data, _make_loader
from visreps.models import utils as mu
from visreps.models.batchnorm import prepare_eval_batchnorm, training_image_ids
from experiments.bn_recalibration.compare_dataset_stats import digest_tensors, file_digest, forbid_db
from experiments.nsd_synthetic_poc.preprocessing import SyntheticReferenceTransform

OUT = Path('experiments/nsd_synthetic_poc')
PREV = Path('experiments/bn_recalibration/dataset_stats_comparison')
REGIONS = {'early': 'early visual stream', 'ventral': 'ventral visual stream'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corrected-imagenet', action='store_true',
                        help='Reference synthetic transform and original ImageNet BN; save separately')
    args = parser.parse_args()
    output = OUT / 'corrected_imagenet' if args.corrected_imagenet else OUT
    output.mkdir(exist_ok=True)
    load_dotenv()
    torch.set_num_threads(8)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db_before = file_digest(Path('results.db'))
    synth = pickle.load((OUT / 'reliable_synthetic.pkl').open('rb'))
    ids = synth['shared_stimulus_names']
    assert len(ids) == len(set(ids)) == 220
    stimuli = {sid: str(Path('datasets/neural/nsd_synthetic/stimuli') / f'{sid}.png') for sid in ids}
    assert all(Path(v).is_file() for v in stimuli.values())
    neural = {}
    for short, full in REGIONS.items():
        neural[full] = {}
        for subj in range(8):
            arr = synth['data'][short][subj]
            assert (arr.ncsnr.values > .2).all()
            neural[full][subj] = {sid: arr.sel(stimulus=sid).values for sid in ids}
    frames = []
    for label in ['clip32', 'default1000']:
        condition = 'saved_imagenet' if args.corrected_imagenet else 'dataset_recalibrated'
        prior = json.loads((PREV / f'nsd_{label}_{condition}_metadata.json').read_text())
        cfg = OmegaConf.create(prior['effective_config'])
        assert not cfg.log_expdata and not cfg.bootstrap
        model = mu.load_model(cfg, dev)
        assert digest_tensors(model.named_parameters()) == prior['weights_sha256']
        assert digest_tensors(model.named_buffers()) == prior['original_buffers_sha256']
        calibration_ids = []
        id_hash = None
        if not args.corrected_imagenet:
            data = load_all_nsd_data(cfg, subjects=list(range(8)), regions=list(REGIONS.values()))
            calibration_ids = training_image_ids(data['neural'], data['stimuli'].keys())
            id_hash = hashlib.sha256('\n'.join(sorted(calibration_ids)).encode()).hexdigest()
            assert id_hash == prior['calibration_ids_sha256']
            loader = _make_loader(data['stimuli'], ev._get_eval_transform(cfg), cfg.batchsize, cfg.num_workers)
            prepare_eval_batchnorm(model, cfg, loader, calibration_ids, dev)
            del data, loader
        model.eval()
        assert digest_tensors(model.named_parameters()) == prior['weights_sha256']
        assert digest_tensors(model.named_buffers()) == prior['evaluation_buffers_sha256']
        assert cfg.bn_calibration == prior['effective_config']['bn_calibration']
        previous = pd.read_json(PREV / f'nsd_{label}_{condition}.json')
        layers, selections = {}, {}
        for region in REGIONS.values():
            rs = previous[previous.region == region]
            assert len(rs) == 8 and rs.layer.nunique() == 1
            layers[region] = rs.layer.iloc[0]
            selections[region] = {int(row.subject_idx): row.layer_selection_scores for row in rs.itertuples()}
        gc.collect()
        checked_inputs = [0]
        handle = None
        if args.corrected_imagenet:
            expected = torch.load(OUT / 'preprocessing_audit/expected_model_inputs.pt', weights_only=True)
            assert expected['ids'] == sorted(ids)
            def check_inputs(module, inputs):
                start = checked_inputs[0] % len(ids)
                count = inputs[0].shape[0]
                torch.testing.assert_close(inputs[0].detach().cpu(), expected['tensors'][start:start+count], rtol=0, atol=0)
                checked_inputs[0] += count
            handle = model.register_forward_pre_hook(check_inputs)
        model = mu.configure_feature_extractor(cfg, model)
        cfg.neural_dataset = 'nsd_synthetic'
        cfg.bn_calibration_source = 'original_imagenet_checkpoint' if args.corrected_imagenet else 'regular_nsd_training'
        print(f'\nSTART {label}: {layers}; {len(calibration_ids)} NSD calibration images; 220 OOD test images', flush=True)
        original_transform = ev._get_eval_transform
        def test_transform(runtime_cfg):
            return SyntheticReferenceTransform() if args.corrected_imagenet else original_transform(runtime_cfg)
        with patch.object(ev, 'save_results', forbid_db), patch.object(ev, '_get_eval_transform', test_transform):
            result = ev._reextract_and_score(model, cfg, dev, stimuli, ids, neural,
                                            layers, list(REGIONS.values()), list(range(8)), selections)
        assert len(result) == 16 and result.score.notna().all()
        if handle:
            handle.remove()
            assert checked_inputs[0] == len(ids) * len(set(layers.values()))
            assert digest_tensors(model.model.named_buffers()) == prior['original_buffers_sha256']
        result.to_json(output / f'{label}_rsa.json', orient='records', indent=2, double_precision=15)
        result['model'] = label
        frames.append(result)
        (output / f'{label}_metadata.json').write_text(json.dumps(dict(
            effective_config=OmegaConf.to_container(cfg, resolve=True),
            selected_layers=layers, selection_source=str(PREV / f'nsd_{label}_{condition}.json'),
            checked_model_input_images=checked_inputs[0],
            synthetic_transform=repr(test_transform(cfg)),
            calibration_images=len(calibration_ids), calibration_ids_sha256=id_hash,
            weights_sha256=prior['weights_sha256'], bn_buffers_sha256=prior['evaluation_buffers_sha256'],
            test_ids=ids, ncsnr_threshold=.2, reliability_source=synth['reliability_source']), indent=2))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    results = pd.concat(frames, ignore_index=True)
    results.drop(columns=['layer_selection_scores']).to_csv(output / 'per_subject.csv', index=False)
    summary = results.groupby(['region','model']).agg(mean_score=('score','mean'), sd=('score','std'), layer=('layer','first')).reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    paired = results.pivot(index=['region','subject_idx'], columns='model', values='score')
    paired['clip32_minus_default1000'] = paired.clip32 - paired.default1000
    paired.to_csv(output / 'paired_differences.csv')
    db_after = file_digest(Path('results.db'))
    assert db_before == db_after
    (output / 'database_check.json').write_text(json.dumps(dict(before_sha256=db_before, after_sha256=db_after), indent=2))
    print(summary.to_string(index=False), flush=True)
    print(paired.to_string(), flush=True)
    print('Verified database unchanged', flush=True)


if __name__ == '__main__':
    main()
