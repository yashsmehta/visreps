"""TVSD coarseness: V1, V4, IT."""
import sys, argparse
sys.path.insert(0, "plotters")
from plot_helpers import plot_coarseness_bars, plot_per_subject, FOLDER_DISPLAY

DCFG = {
    "neural_dataset": "tvsd",
    "regions": ["V1", "V4", "IT"],
    "region_labels": {"V1": "V1", "V4": "V4", "IT": "IT"},
    "has_subjects": True,
}
OUTPUT_DIR = "plotters/tvsd/figures"

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

    plot_coarseness_bars(DCFG, args.folder, OUTPUT_DIR, dataset_label="TVSD")
    plot_per_subject(DCFG, args.folder, OUTPUT_DIR, dataset_label="TVSD")
