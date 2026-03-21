"""NSD: RSA score across training epochs (32-way vs 1000-way)."""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DB_PATH = "results.db"
OUTPUT_DIR = "plotters/nsd/figures"

sns.set_theme(style="ticks", context="paper", font_scale=1.1)

COLOR_32 = sns.color_palette("Blues", 7)[5]
COLOR_1000 = "#FFA500"

REGIONS = [
    {"region": "early visual stream", "title": "Early Visual Stream"},
    {"region": "ventral visual stream", "title": "Ventral Visual Stream"},
]

MODELS = [
    {
        "model_name": "ViTBase",
        "checkpoint_filter": "checkpoint_dir LIKE '%vitbase%'",
        "output_suffix": "vitbase",
    },
    {
        "model_name": "ResNet50",
        "checkpoint_filter": "checkpoint_dir LIKE '%resnet50%'",
        "output_suffix": "resnet50",
    },
]

conn = sqlite3.connect(DB_PATH)

for model in MODELS:
    df = pd.read_sql(f"""
        SELECT cfg_id, seed, epoch, region, subject_idx, score
        FROM results
        WHERE neural_dataset = 'nsd'
          AND analysis = 'rsa'
          AND compare_method = 'spearman'
          AND {model["checkpoint_filter"]}
          AND model_name = '{model["model_name"]}'
          AND region IN ('early visual stream', 'ventral visual stream')
          AND reconstruct_from_pcs = 0
        ORDER BY cfg_id, epoch, region
    """, conn)

    if df.empty:
        print(f"No data found for {model['model_name']}, skipping")
        continue

    # Average across subjects per (cfg_id, epoch, region)
    agg = (
        df.groupby(["cfg_id", "epoch", "region"])["score"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, reg in zip(axes, REGIONS):
        for cfg_id, color, label in [(32, COLOR_32, "32-way"), (1000, COLOR_1000, "1000-way")]:
            sub = agg[(agg["cfg_id"] == cfg_id) & (agg["region"] == reg["region"])].sort_values("epoch")
            ax.plot(sub["epoch"], sub["score"], "o-", color=color, label=label,
                    markersize=5, linewidth=1.8, zorder=3)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("RSA (Spearman $\\rho$)")
        ax.set_title(reg["title"])
        ax.set_xticks(sorted(df["epoch"].unique()))
        ax.legend(frameon=False)
        sns.despine(ax=ax)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/nsd_epoch_trajectory_{model['output_suffix']}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")

conn.close()
