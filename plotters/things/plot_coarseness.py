"""THINGS-behavior coarseness: single panel, no per-subject."""
import sys, argparse
sys.path.insert(0, "plotters")
from plot_helpers import plot_coarseness_bars, plot_per_subject, PCA_MODELS

DCFG = {
    "neural_dataset": "things-behavior",
    "regions": ["N/A"],
    "region_labels": {"N/A": "THINGS Behavior"},
    "has_subjects": False,
}
OUTPUT_DIR = "plotters/things/figures"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pca_labels", default="alexnet", choices=list(PCA_MODELS))
    p.add_argument("--analysis", default="rsa", choices=["rsa", "encoding_score"])
    p.add_argument("--compare_method", default=None, choices=["spearman", "pearson"],
                   help="Defaults to spearman for RSA, pearson for encoding_score")
    args = p.parse_args()

    DCFG["analysis"] = args.analysis
    DCFG["compare_method"] = args.compare_method or (
        "pearson" if args.analysis == "encoding_score" else "spearman")

    if args.analysis == "encoding_score":
        DCFG["output_suffix"] = "_encoding"

    plot_coarseness_bars(DCFG, args.pca_labels, OUTPUT_DIR, dataset_label="THINGS")
    plot_per_subject(DCFG, args.pca_labels, OUTPUT_DIR, dataset_label="THINGS")
