import pandas as pd
import numpy as np

def analyze_class_size(dataset, n_classes):
    print(f"\n{'='*20} Analyzing {n_classes} classes {'='*20}")
    
    # Load both CSV files
    new_labels = pd.read_csv(f'datasets/obj_cls/{dataset}/pca_labels/n_classes_{n_classes}.csv')
    legacy_labels = pd.read_csv(f'datasets/obj_cls/{dataset}/pca_labels_legacy/n_classes_{n_classes}.csv')

    # Merge on image names to compare labels
    merged = pd.merge(new_labels, legacy_labels, on='image', suffixes=('_new', '_legacy'))
    
    # Calculate agreement percentage
    total_samples = len(merged)
    matching_labels = (merged['pca_label_new'] == merged['pca_label_legacy']).sum()
    agreement_pct = (matching_labels / total_samples) * 100

    # Calculate class distributions
    new_dist = merged['pca_label_new'].value_counts(normalize=True).sort_index()
    legacy_dist = merged['pca_label_legacy'].value_counts(normalize=True).sort_index()

    print(f"\nTotal samples analyzed: {total_samples}")
    print(f"Label agreement: {matching_labels}/{total_samples} ({agreement_pct:.2f}%)")
    
    # Compare distributions and only show differences
    print("\nClass distribution differences (new - legacy):")
    has_differences = False
    for label in sorted(set(new_dist.index) | set(legacy_dist.index)):
        new_pct = new_dist.get(label, 0) * 100
        legacy_pct = legacy_dist.get(label, 0) * 100
        diff = new_pct - legacy_pct
        if abs(diff) > 0.01:  # Account for floating point precision
            has_differences = True
            print(f"Class {label}: {diff:+.2f}%")
    
    if not has_differences:
        print("No significant distribution differences")

    # If agreement is not perfect, analyze mismatches
    if agreement_pct < 100:
        print("\nAnalyzing mismatches:")
        mismatches = merged[merged['pca_label_new'] != merged['pca_label_legacy']]
        print(f"\nSample of mismatched labels (first 5):")
        print(mismatches[['image', 'pca_label_new', 'pca_label_legacy']].head())

def test_pca_label_consistency():
    dataset = 'imagenet'
    for n_classes in [2, 4, 8]:
        analyze_class_size(dataset, n_classes)

if __name__ == '__main__':
    test_pca_label_consistency()
