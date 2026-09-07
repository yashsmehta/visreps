"""One-seed RSA comparison of saved ImageNet BN versus dataset recalibration.

Run from repo root: PYTHONPATH=. .venv/bin/python -u experiments/bn_recalibration/compare_dataset_stats.py
Writes JSON/CSV locally only; the database writer is disabled for this process.
"""
import gc
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv
from omegaconf import OmegaConf
import pandas as pd
import torch

import visreps.evals as evaluation
from visreps.models.batchnorm import prepare_eval_batchnorm
from visreps.utils import load_config, validate_config

OUT = Path('experiments/bn_recalibration/dataset_stats_comparison')


def digest_tensors(items):
    h = hashlib.sha256()
    for name, tensor in items:
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def file_digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def forbid_db(*args, **kwargs):
    raise RuntimeError('Database writing is disabled in this experiment')


def main():
    load_dotenv()
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    db = Path('results.db')
    db_before = file_digest(db)
    (OUT / 'database_before.sha256').write_text(db_before + '\n')
    for dataset, subjects, regions in [
        ('tvsd', [0, 1], ['V1', 'V4', 'IT']),
        ('nsd', list(range(8)), ['early visual stream', 'ventral visual stream']),
    ]:
        for label, folder, cfg_id in [('clip32', 'clip_pca', 32), ('default1000', 'default', 1000)]:
            for condition in ['saved_imagenet', 'dataset_recalibrated']:
                name = f'{dataset}_{label}_{condition}'
                target = OUT / f'{name}.json'
                if target.exists():
                    continue
                overrides = [f'neural_dataset={dataset}', f'subject_idx={subjects}',
                             f'region={json.dumps(regions)}', 'seed=1',
                             'load_model_from=checkpoint', f'checkpoint_dir=/data/ymehta3/{folder}',
                             f'cfg_id={cfg_id}', 'checkpoint_model=checkpoint_epoch_20.pth',
                             'analysis=rsa', 'compare_method=spearman', 'bootstrap=false',
                             'log_expdata=false', 'reconstruct_from_pcs=false',
                             'num_workers=8', 'batchsize=128']
                cfg = validate_config(load_config('configs/eval/base.json', overrides))
                provenance = {'condition': condition, 'overrides': overrides}

                def prepare(model, runtime_cfg, loader, image_ids, device):
                    model.eval()
                    bn = {n: m for n, m in model.named_modules()
                          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
                    assert bn, 'Expected BatchNorm in these CustomCNN checkpoints'
                    weights_before = digest_tensors(model.named_parameters())
                    buffers_before = digest_tensors(model.named_buffers())
                    provenance.update(bn_modules=list(bn), weights_sha256=weights_before,
                                      original_buffers_sha256=buffers_before,
                                      eligible_calibration_images=len(image_ids),
                                      calibration_ids_sha256=hashlib.sha256('\n'.join(sorted(image_ids)).encode()).hexdigest())
                    if condition == 'dataset_recalibrated':
                        prepare_eval_batchnorm(model, runtime_cfg, loader, image_ids, device)
                    else:
                        runtime_cfg.bn_calibration = 'saved_imagenet'
                        print('  Using original checkpoint BN buffers; no recalibration', flush=True)
                    assert digest_tensors(model.named_parameters()) == weights_before
                    assert all(not m.training for m in model.modules())
                    buffers_after = digest_tensors(model.named_buffers())
                    if condition == 'saved_imagenet':
                        assert buffers_after == buffers_before
                    else:
                        assert buffers_after != buffers_before
                    provenance.update(evaluation_buffers_sha256=buffers_after,
                                      effective_config=OmegaConf.to_container(runtime_cfg, resolve=True))
                    return model

                print(f'\nSTART {name}', flush=True)
                with patch.object(evaluation, 'prepare_eval_batchnorm', prepare), \
                     patch.object(evaluation, 'save_results', forbid_db):
                    result = evaluation.eval(cfg)
                assert len(result) == len(subjects) * len(regions)
                assert result.score.notna().all()
                result.to_json(target, orient='records', indent=2, double_precision=15)
                (OUT / f'{name}_metadata.json').write_text(json.dumps(provenance, indent=2))
                print(f'SAVED {target}', flush=True)
                gc.collect()
                torch.cuda.empty_cache()
    db_after = file_digest(db)
    (OUT / 'database_after.sha256').write_text(db_after + '\n')
    assert db_after == db_before, 'Database changed during experiment'
    rows = []
    for dataset in ['tvsd', 'nsd']:
        for label in ['clip32', 'default1000']:
            for condition in ['saved_imagenet', 'dataset_recalibrated']:
                frame = pd.read_json(OUT / f'{dataset}_{label}_{condition}.json')
                frame['dataset'], frame['model'], frame['bn_condition'] = dataset, label, condition
                rows.append(frame)
    results = pd.concat(rows, ignore_index=True)
    results.drop(columns=['layer_selection_scores']).to_csv(OUT / 'per_subject.csv', index=False)
    summary = results.groupby(['dataset', 'region', 'model', 'bn_condition'], sort=False).agg(
        mean_score=('score', 'mean'), layer=('layer', 'first'), n_subjects=('subject_idx', 'count')).reset_index()
    summary.to_csv(OUT / 'summary.csv', index=False)
    print(summary.to_string(index=False), flush=True)
    print('Confirmed: results.db SHA-256 unchanged', flush=True)


if __name__ == '__main__':
    main()
