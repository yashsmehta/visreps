"""THINGS-behavior: RSA score across training epochs (32-way vs 1000-way)."""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DB_PATH = "results.db"
OUTPUT_DIR = "plotters/things/figures"

sns.set_theme(style="ticks", context="paper", font_scale=1.1)

COLOR_32 = sns.color_palette("Blues", 7)[5]
COLOR_1000 = "#FFA500"

MODELS = [
    {"model_name": "ResNet50", "checkpoint_pattern": "%resnet50%", "ylim": (0.3, 0.6)},
    {"model_name": "ViTBase", "checkpoint_pattern": "%vitbase%", "ylim": (0.25, 0.5)},
]

conn = sqlite3.connect(DB_PATH)

for model in MODELS:
    df = pd.read_sql(f"""
        SELECT cfg_id, seed, epoch, score
        FROM results
        WHERE neural_dataset = 'things-behavior'
          AND analysis = 'rsa'
          AND compare_method = 'spearman'
          AND checkpoint_dir LIKE '{model["checkpoint_pattern"]}'
          AND model_name = '{model["model_name"]}'
        ORDER BY cfg_id, epoch
    """, conn)

    if df.empty:
        print(f"No data for {model['model_name']}, skipping")
        continue

    fig, ax = plt.subplots(figsize=(5, 4))

    for cfg_id, color, label in [(32, COLOR_32, "32-way"), (1000, COLOR_1000, "1000-way")]:
        sub = df[df["cfg_id"] == cfg_id].sort_values("epoch")
        ax.plot(sub["epoch"], sub["score"], "o-", color=color, label=label,
                markersize=5, linewidth=1.8, zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RSA (Spearman $\\rho$)")
    ax.set_title(f"THINGS Behavioral Alignment — {model['model_name']}")
    ax.set_xticks(sorted(df["epoch"].unique()))
    ax.set_ylim(model["ylim"])
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    tag = model["model_name"].lower()
    out = f"{OUTPUT_DIR}/things_epoch_trajectory_{tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")

conn.close()
