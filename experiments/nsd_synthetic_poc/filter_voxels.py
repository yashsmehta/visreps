"""Restore synthetic voxel identities, verify against raw betas, filter NSD NCSNR."""
import gc
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault('BONNER_DATASETS_HOME', '/data/shared/datasets')
import h5py
import numpy as np
from loguru import logger
logger.remove()
from bonner.datasets.allen2021_natural_scenes import load_brain_mask, load_rois, load_ncsnr
from bonner.datasets.gifford2025_nsd_synthetic._data import load_validity, load_presentations

OUT = Path('experiments/nsd_synthetic_poc')
SOURCE = Path('datasets/neural/nsd_synthetic/nsd_synthetic_data.pkl')


def main():
    source = pickle.load(SOURCE.open('rb'))
    filtered = {r: {} for r in ['early', 'ventral']}
    counts = []
    presentations = load_presentations()['stimulus'].to_numpy()
    for subj in range(8):
        print(f'Subject {subj}: restoring coordinates', flush=True)
        brain = load_brain_mask(subject=subj, resolution='1pt8mm')
        valid = load_validity(subject=subj, resolution='1pt8mm')
        assert brain.dims == valid.dims == ('x', 'y', 'z')
        valid_flat = (brain.values.astype(bool) & valid.values.astype(bool)).ravel()
        rois = load_rois(subject=subj, resolution='1pt8mm')
        ncsnr = load_ncsnr(subject=subj, resolution='1pt8mm', preprocessing='fithrf_GLMdenoise_RR')
        # All three arrays must have the beta loader's x-major/z-minor ordering.
        for coord, expected in zip(['x', 'y', 'z'], np.indices(brain.shape).reshape(3, -1)):
            np.testing.assert_array_equal(rois.coords[coord].values, expected)
            np.testing.assert_array_equal(ncsnr.coords[coord].values, expected)
        raw_path = Path('/data/shared/datasets/gifford2025.nsd_synthetic/nsddata_betas/ppdata') / f'subj{subj+1:02}/func1pt8mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR/betas_nsdsynthetic.hdf5'
        with h5py.File(raw_path) as f:
            for region in filtered:
                mask = np.zeros(valid_flat.shape, bool)
                for key in rois.roi.values:
                    if key[0] == 'streams' and key[1] == region:
                        mask |= rois.sel(roi=key).values > 0
                flat_indices = np.flatnonzero(mask & valid_flat)
                arr = source['data'][region][subj]
                assert len(flat_indices) == arr.sizes['neuroid'], (subj, region, len(flat_indices), arr.shape)
                xyz = np.stack(np.unravel_index(flat_indices, brain.shape), axis=1)
                # Verify recovered ordering with actual repeated-response data.
                for idx in sorted({0, len(xyz)//2, len(xyz)-1}):
                    x, y, z = xyz[idx]
                    raw = f['betas'][:, z, y, x].astype(np.float32)
                    raw = (raw - raw.mean()) / raw.std()
                    averaged = np.array([raw[presentations == stim].astype(np.float64).mean()
                                         for stim in arr.stimulus.values], dtype=np.float32)
                    np.testing.assert_allclose(averaged, arr.values[:, idx], rtol=1e-5, atol=1e-6)
                reliability = ncsnr.values[flat_indices]
                keep = np.isfinite(reliability) & (reliability > 0.2)
                assert keep.any() and not keep.all()
                arr = arr.assign_coords({k: ('neuroid', xyz[:, j]) for j,k in enumerate(['x','y','z'])})
                arr = arr.assign_coords(ncsnr=('neuroid', reliability)).isel(neuroid=np.flatnonzero(keep))
                assert np.isfinite(arr.values).all()
                filtered[region][subj] = arr
                counts.append(dict(subject_idx=subj, region=region, retained=int(keep.sum()), total=len(keep)))
                print(f'  {region}: {keep.sum()}/{len(keep)} voxels; raw-beta spot checks passed', flush=True)
        del rois, ncsnr, brain, valid
        gc.collect()
    with (OUT / 'reliable_synthetic.pkl').open('wb') as f:
        pickle.dump(dict(data=filtered, shared_stimulus_names=source['shared_stimulus_names'],
                         ncsnr_threshold=0.2, reliability_source='regular NSD, fithrf_GLMdenoise_RR, 1pt8mm',
                         voxel_counts=counts), f)
    (OUT / 'voxel_counts.json').write_text(json.dumps(counts, indent=2))
    print('Saved reliable synthetic responses separately; original archive unchanged', flush=True)


if __name__ == '__main__':
    main()
