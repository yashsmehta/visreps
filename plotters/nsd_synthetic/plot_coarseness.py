"""NSD-Synthetic coarseness: fine-grained ROIs (V1, V2, V3, hV4, FFA, PPA)."""
import sys, argparse
sys.path.insert(0, "plotters")
from plot_helpers import plot_coarseness_bars, plot_per_subject, FOLDER_DISPLAY

DCFG = {
    "neural_dataset": "nsd_synthetic",
    "regions": ["V1", "V2", "V3", "hV4", "FFA", "PPA"],
    "region_labels": {
        "V1": "V1", "V2": "V2", "V3": "V3",
        "hV4": "hV4", "FFA": "FFA", "PPA": "PPA",
    },
    "has_subjects": True,
    "layout": (2, 4, [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]),
}
OUTPUT_DIR = "plotters/nsd_synthetic/figures"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True, choices=list(FOLDER_DISPLAY))
    p.add_argument("--analysis", default="rsa", choices=["rsa", "encoding_score"])
    p.add_argument("--compare_method", default="spearman", choices=["spearman", "pearson"])
    args = p.parse_args()

    DCFG["analysis"] = args.analysis
    DCFG["compare_method"] = args.compare_method
    
    if args.analysis == "encoding_score":
        DCFG["output_suffix"] = "_encoding"

    plot_coarseness_bars(DCFG, args.folder, OUTPUT_DIR, dataset_label="NSD-SYNTHETIC")
    plot_per_subject(DCFG, args.folder, OUTPUT_DIR, dataset_label="NSD-SYNTHETIC")
