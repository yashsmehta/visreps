import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Import patches
# from matplotlib.ticker import MaxNLocator # No longer needed
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter # Import new tickers
import pandas as pd
import numpy as np
import os # Add os import


def plot_brain_score_barplot(scores_by_cond: dict[str, list[float]],
                             out_png: str) -> None:
    """
    Bar-plot brain similarity (mean ± s.d.) for each training condition.

    Parameters
    ----------
    scores_by_cond : dict[str, list[float]]
        Keys = category labels (display order preserved).
        Values = list of per-seed scores for that category.
    out_png : str
        Destination PNG filename.
    """
    # ── summary stats ────────────────────────────────────────────────────────
    cats  = list(scores_by_cond.keys())
    means = [np.mean(v) for v in scores_by_cond.values()]
    errs  = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in scores_by_cond.values()]

    # ── palette & hatches (same style) ───────────────────────────────────────
    untrained_c, thousand_c = '#AAAAAA', '#FFA500'
    pca_cats = [c for c in cats if c not in ('Untrained', '1000 Classes')]
    blues    = sns.color_palette('Blues', n_colors=max(len(pca_cats), 1) + 1)[1:]

    palette = {c: (untrained_c if c == 'Untrained'
                   else thousand_c if c == '1000 Classes'
                   else blues[pca_cats.index(c)])
               for c in cats}
    hatches = {c: ('' if c in ('Untrained', '1000 Classes') else '/')
               for c in cats}

    # ── plotting ─────────────────────────────────────────────────────────────
    sns.set_theme(style='ticks', context='paper', font_scale=1.1) # Changed style to 'ticks'
    fig, ax = plt.subplots(figsize=(8, 5))

    # Store original hatch color and set to desired gray
    original_hatch_color = plt.rcParams.get('hatch.color') # Use .get for safety
    plt.rcParams['hatch.color'] = 'grey'  # Softer gray for hatches

    bar_w     = .7
    positions = np.arange(len(cats))

    for i, cat in enumerate(cats):
        x0  = positions[i] - bar_w / 2
        rect = mpatches.FancyBboxPatch(
            (x0, 0), bar_w, means[i],
            boxstyle=mpatches.BoxStyle('Round', pad=.02, rounding_size=.1),
            facecolor=palette[cat], edgecolor='black',
            linewidth=.8, hatch=hatches[cat], mutation_aspect=.05
        )
        ax.add_patch(rect)

        if errs[i] > 0:
            ax.errorbar(positions[i], means[i], yerr=errs[i],
                        fmt='none', ecolor='black',
                        elinewidth=1., capsize=4, capthick=1.)

    # ── axis formatting ──────────────────────────────────────────────────────
    ax.set_xticks(positions)
    ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=10) # Updated fontsize
    ax.tick_params(axis='x', direction='out', bottom=False, top=False, length=4, color='black', width=1.5) # Updated x-tick params

    # Enhanced y-axis formatting (similar to bar_plot_things.py)
    ax.tick_params(axis='y', which='major', direction='out', left=True, right=False, labelsize=18, length=5, color='black', width=1.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Custom formatter to hide 0.0 and format others to one decimal place
    def hide_zero_formatter(x, pos):
        return '' if np.isclose(x, 0) else f'{x:.1f}'
    ax.yaxis.set_major_formatter(FuncFormatter(hide_zero_formatter))

    ax.tick_params(axis='y', which='minor', direction='out', left=True, right=False, length=4, color='black', width=1.0)

    current_max_y = max(means) if means else 0
    ax.set_ylim(0, current_max_y + 0.02 if current_max_y > 0 else 0.1)
    ax.set_xlim(-.5, len(cats) - .5)
    ax.set_ylabel('Brain Similarity (RSA)', fontsize=14, labelpad=10) # Updated y-label params

    sns.despine(right=True, top=True, offset=5) # Added offset
    # Make remaining spines thicker
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    plt.tight_layout(pad=1.0) # Adjusted padding
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved → {out_png}")

    # Restore original hatch color
    if original_hatch_color is not None: # Restore only if it was found
        plt.rcParams['hatch.color'] = original_hatch_color

# --- Example Usage ---
if __name__ == "__main__":
    # ---------------- config ----------------
    base_log_path       = 'logs/'
    data_filename       = 'full-vs-pcs_nsd.csv'          # keep as-is
    pc_layer_to_plot       = 'conv4'                        # lowercase
    k1k_layer_to_plot       = 'conv3'                        # lowercase
    region_to_plot      = 'early visual stream'
    pca_classes_to_plot = [2, 4, 8, 16, 32, 64]          # used elsewhere
    out_png             = f"plotters/fig2/barplt_{pc_layer_to_plot}_region_{region_to_plot.lower().replace(' ','_')}.png"

    # ---------------- load & sanitise ----------------
    df = pd.read_csv(os.path.join(base_log_path, data_filename))
    df['layer'] = df['layer'].str.lower()
    df = df[df['region'].str.lower() == region_to_plot.lower()]

    # ---------------- helper ----------------
    def mean_per_seed(sub: pd.DataFrame) -> pd.DataFrame:
        """Collapse over subject_idx → one row per seed with mean score."""
        return (sub.groupby('seed', as_index=False)
                .agg(score=('score', 'mean'))
                .assign(epoch=sub['epoch'].iloc[0]))  # keep epoch for clarity

    # ---------------- selections ----------------
    pc_layer_mask = df['layer'] == pc_layer_to_plot
    k1k_layer_mask = df['layer'] == k1k_layer_to_plot

    # untrained (epoch 0) → three rows (one per seed)
    untrained = mean_per_seed(df[k1k_layer_mask & (df['epoch'] == 0)])

    # --- PCA-trained (epoch 20, pca_labels=True, specific class counts) ---
    pca_mask = (pc_layer_mask
                & (df['epoch'] == 20)
                & df['pca_labels']
                & df['pca_n_classes'].isin(pca_classes_to_plot))

    pca = (df[pca_mask]
        .groupby(['pca_n_classes', 'seed'], as_index=False)  # keep class + seed
        .agg(score=('score', 'mean')))                       # mean across subjects

    # ImageNet-1K-trained (epoch 20, pca_labels=False) → three rows
    trained_1k = mean_per_seed(df[k1k_layer_mask
                                & (df['epoch'] == 20)
                                & (~df['pca_labels'])])

    # ---------------- debug prints ----------------
    print("\n=== Untrained (epoch 0) ===")
    print(untrained)

    print("\n=== PCA (epoch 20) ===")
    print(pca[['seed', 'pca_n_classes', 'score']])

    print("\n=== ImageNet-1K (epoch 20) ===")
    print(trained_1k)

    # ---------------- assemble dict -> lists ----------------
    # Untrained
    scores_by_cond = {'Untrained': untrained['score'].tolist()}

    # PCA classes (will follow pca_classes_to_plot order)
    for n in pca_classes_to_plot:
        label = f"{n} Classes"
        vals  = pca.loc[pca['pca_n_classes'] == n, 'score'].tolist()
        if not vals:
            print(f"Warning: missing PCA scores for {n}-class model.")
            continue
        scores_by_cond[label] = vals

    # ImageNet-1K
    scores_by_cond['1000 Classes'] = trained_1k['score'].tolist()
    print(scores_by_cond)

    # ---------------- plot ----------------
    plot_brain_score_barplot(scores_by_cond, out_png)