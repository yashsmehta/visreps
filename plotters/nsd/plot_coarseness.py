"""NSD coarseness: early + ventral visual streams."""
import sys, argparse
sys.path.insert(0, "plotters")
from plot_helpers import plot_coarseness_bars, plot_per_subject, FOLDER_DISPLAY

DCFG = {
    "neural_dataset": "nsd",
    "regions": ["early visual stream", "ventral visual stream"],
    "region_labels": {
        "early visual stream": "Early Visual Stream",
        "ventral visual stream": "Ventral Visual Stream",
    },
    "has_subjects": True,
}
OUTPUT_DIR = "plotters/nsd/figures"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True, choices=list(FOLDER_DISPLAY))
    p.add_argument("--analysis", default="rsa", choices=["rsa", "encoding_score"])
    p.add_argument("--compare_method", default="spearman", choices=["spearman", "pearson"])
    args = p.parse_args()
    
    DCFG["analysis"] = args.analysis
    DCFG["compare_method"] = args.compare_method
    
    # Optionally append suffix if not default RSA/Spearman
    if args.analysis == "encoding_score":
        DCFG["output_suffix"] = "_encoding"

    plot_coarseness_bars(DCFG, args.folder, OUTPUT_DIR, dataset_label="NSD")
    plot_per_subject(DCFG, args.folder, OUTPUT_DIR, dataset_label="NSD")
