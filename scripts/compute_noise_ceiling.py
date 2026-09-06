"""Compute within-subject RSA noise ceilings for NSD ROIs.

Writes datasets/neural/noise_ceilings.json, which the plotters read to draw the
ceiling line. Nothing here touches results.db.

Usage:
    python scripts/compute_noise_ceiling.py                        # NSD streams
    python scripts/compute_noise_ceiling.py --regions V1 V2 V3 hV4 FFA PPA
    python scripts/compute_noise_ceiling.py --dataset tvsd         # V1, V4, IT
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from visreps.analysis.noise_ceiling import nsd_ceiling, tvsd_ceiling

OUT_PATH = "datasets/neural/noise_ceilings.json"
CEILINGS = {"nsd": nsd_ceiling, "tvsd": tvsd_ceiling}
DEFAULT_REGIONS = {
    "nsd": ["early visual stream", "ventral visual stream"],
    "tvsd": ["V1", "V4", "IT"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nsd", choices=list(CEILINGS))
    p.add_argument("--regions", nargs="+", default=None)
    p.add_argument("--compare_method", default="spearman",
                   choices=["spearman", "kendall", "pearson"])
    p.add_argument("--n_splits", type=int, default=20,
                   help="Random 1-vs-1 split-half repeats per subject")
    p.add_argument("--out", default=OUT_PATH)
    args = p.parse_args()
    regions = args.regions or DEFAULT_REGIONS[args.dataset]

    ceilings = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            ceilings = json.load(f)

    for region in regions:
        c = CEILINGS[args.dataset](
            region, method=args.compare_method, n_splits=args.n_splits)
        ceilings[f"{args.dataset}|{region}|{args.compare_method}"] = c
        print(f"{region:24s} [{args.compare_method}]  ceiling={c['ceiling']:.4f} "
              f"(+-{c['sem']:.4f} SEM, reliability {c['mean_reliability']:.4f}, "
              f"{c['n_stimuli']} stimuli)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(ceilings, f, indent=2, sort_keys=True)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
