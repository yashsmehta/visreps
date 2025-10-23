"""
Process Cusack 2025 fMRI data and save in NSD-compatible format.
Output: datasets/neural/cusack2025/fmri_responses.pkl
Structure: data[region][age] = {stim_id: response_array}
"""
import pickle, numpy as np, os

def process_cusack_data():
    """Process Cusack fMRI data and save in NSD-compatible format."""
    # Load datasets
    data2 = pickle.load(open('/data/shared/datasets/cusack2025/pictures_roi_betas_bysubj_twomonth_vcovthreshold10_eg_julich_nonicu_newcons.pickle', 'rb'))
    data9 = pickle.load(open('/data/shared/datasets/cusack2025/pictures_roi_betas_bysubj_ninemonth_vcovthreshold10_eg_julich_nonicu_newcons.pickle', 'rb'))

    condnames = data2['condnames']
    fmri_data = {}

    # Process each region
    for region in data2['roivalues']:
        clean_region = region.replace('both_', '')
        subjects_2month = data2['roivalues'][region]
        subjects_9month = data9['roivalues'][region]

        # Get dimensions and print progress
        first_subj_2month = next(iter(subjects_2month))
        first_sess_2month = next(iter(subjects_2month[first_subj_2month]))
        n_voxels_2month, n_trials = subjects_2month[first_subj_2month][first_sess_2month]['betas'].shape

        first_subj_9month = next(iter(subjects_9month))
        first_sess_9month = next(iter(subjects_9month[first_subj_9month]))
        n_voxels_9month, _ = subjects_9month[first_subj_9month][first_sess_9month]['betas'].shape

        print(f"Region {clean_region}: 2month={len(subjects_2month)}s ({n_voxels_2month}v), 9month={len(subjects_9month)}s ({n_voxels_9month}v)")

        # Process 2-month: average sessions per subject, then median across subjects
        responses_2month = [np.mean([subjects_2month[s][sess]['betas'] for sess in subjects_2month[s]], axis=0).T
                          for s in subjects_2month]
        median_2month = np.median(np.stack(responses_2month, axis=1), axis=1)

        # Process 9-month: same pipeline
        responses_9month = [np.mean([subjects_9month[s][sess]['betas'] for sess in subjects_9month[s]], axis=0).T
                           for s in subjects_9month]
        median_9month = np.median(np.stack(responses_9month, axis=1), axis=1)

        # Structure like NSD synthetic: fmri[region][subject] = {stim_id: response_array}
        fmri_data[clean_region] = {
            '2month': {stim_id: median_2month[i] for i, stim_id in enumerate(condnames)},
            '9month': {stim_id: median_9month[i] for i, stim_id in enumerate(condnames)}
        }

    # Save in NSD-compatible format
    os.makedirs('datasets/neural/cusack2025', exist_ok=True)
    pickle.dump(fmri_data, open('datasets/neural/cusack2025/fmri_responses.pkl', 'wb'))

    print(f"\n✅ Complete! {len(fmri_data)} regions saved")
    print(f"Format: data[region][age] = {{stim_id: array}}")
    print(f"Sample: V1/2month/cat1 → {fmri_data['V1']['2month']['cat1'].shape}")

if __name__ == "__main__":
    process_cusack_data()