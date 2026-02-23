"""NSD-Synthetic coarseness plots.

Supports both fine-grained ROIs and visual streams via --regions flag.
"""
import sys, argparse
sys.path.insert(0, "plotters")
from plot_helpers import plot_coarseness_bars, plot_per_subject, PCA_MODELS

REGION_PRESETS = {
    "finegrained": {
        "regions": ["V1", "V2", "V3", "hV4", "FFA", "PPA"],
        "region_labels": {
            "V1": "V1", "V2": "V2", "V3": "V3",
            "hV4": "hV4", "FFA": "FFA", "PPA": "PPA",
        },
        "layout": (2, 4, [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]),
        "output_suffix": "_finegrained",
    },
    "streams": {
        "regions": ["early visual stream", "ventral visual stream"],
        "region_labels": {
            "early visual stream": "Early Visual Stream",
            "ventral visual stream": "Ventral Visual Stream",
        },
        "output_suffix": "",
    },
}
OUTPUT_DIR = "plotters/nsd_synthetic/figures"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pca_labels", default="alexnet", choices=list(PCA_MODELS))
    p.add_argument("--regions", default="finegrained", choices=list(REGION_PRESETS),
                   help="Region set to plot (default: finegrained)")
    p.add_argument("--analysis", default="rsa", choices=["rsa", "encoding_score"])
    p.add_argument("--compare_method", default=None, choices=["spearman", "pearson"],
                   help="Defaults to spearman for RSA, pearson for encoding_score")
    args = p.parse_args()

    preset = REGION_PRESETS[args.regions]
    suffix = preset["output_suffix"]
    if args.analysis == "encoding_score":
        suffix = suffix + "_encoding" if suffix else "_encoding"

    dcfg = {
        "neural_dataset": "nsd_synthetic",
        "has_subjects": True,
        "analysis": args.analysis,
        "compare_method": args.compare_method or (
            "pearson" if args.analysis == "encoding_score" else "spearman"),
        "output_suffix": suffix,
        **{k: v for k, v in preset.items() if k != "output_suffix"},
    }

    plot_coarseness_bars(dcfg, args.pca_labels, OUTPUT_DIR, dataset_label="NSD-SYNTHETIC")
    plot_per_subject(dcfg, args.pca_labels, OUTPUT_DIR, dataset_label="NSD-SYNTHETIC")
