"""
Process Cusack 2025 fMRI data and save in NSD-compatible format.
Output: datasets/neural/cusack2025/fmri_responses.pkl
Structure: data[region][age] = {stim_id: response_array}
"""
import pickle
import numpy as np
import os

def process_cusack_data():
    """Process Cusack fMRI data and save in NSD-compatible format."""
    
    # Load raw data
    data2 = pickle.load(open('/data/shared/datasets/cusack2025/pictures_roi_betas_bysubj_twomonth_vcovthreshold10_eg_julich_nonicu_newcons.pickle', 'rb'))
    data9 = pickle.load(open('/data/shared/datasets/cusack2025/pictures_roi_betas_bysubj_ninemonth_vcovthreshold10_eg_julich_nonicu_newcons.pickle', 'rb'))
    
    condnames = data2['condnames']
    fmri_data = {}
    
    print(f"Processing {len(data2['roivalues'])} regions...")
    
    # Process each region
    for region in data2['roivalues']:
        clean_region = region.replace('both_', '')
        subjects_2month = data2['roivalues'][region]
        subjects_9month = data9['roivalues'][region]
        
        # Process 2-month data: average sessions within each subject, then median across subjects
        responses_2month = [
            np.mean([subjects_2month[s][sess]['betas'] for sess in subjects_2month[s]], axis=0).T
            for s in subjects_2month
        ]
        median_2month = np.nanmedian(np.stack(responses_2month, axis=1), axis=1)
        
        # Process 9-month data: same pipeline
        responses_9month = [
            np.mean([subjects_9month[s][sess]['betas'] for sess in subjects_9month[s]], axis=0).T
            for s in subjects_9month
        ]
        median_9month = np.nanmedian(np.stack(responses_9month, axis=1), axis=1)
        
        # Store in NSD-compatible format
        fmri_data[clean_region] = {
            '2month': {stim_id: median_2month[i] for i, stim_id in enumerate(condnames)},
            '9month': {stim_id: median_9month[i] for i, stim_id in enumerate(condnames)}
        }
        
        print(f"  ✓ {clean_region}: 2month={len(subjects_2month)} subjects, 9month={len(subjects_9month)} subjects")
    
    # Save output
    os.makedirs('datasets/neural/cusack2025', exist_ok=True)
    output_path = 'datasets/neural/cusack2025/fmri_responses.pkl'
    pickle.dump(fmri_data, open(output_path, 'wb'))
    
    print(f"\n✅ Saved {len(fmri_data)} regions to {output_path}")
    print(f"Structure: data[region][age][stim_id] → array")
    print(f"Example: data['V1']['2month']['cat1'] → shape {fmri_data['V1']['2month']['cat1'].shape}")

if __name__ == "__main__":
    process_cusack_data()
