"""
Plot PC1-PC2 comparison between pretrained and coarse-trained AlexNet.

Loads pre-computed analysis data (from run_analysis.py) and creates a
publication-quality side-by-side scatter plot for a selected layer.

Usage (from project root):
    python experiments/representation_analysis/2pcs_compare/plot.py --n_classes 4
    python experiments/representation_analysis/2pcs_compare/plot.py --n_classes 4 --layer conv4
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LAYER_LABELS = {
    'conv4': 'Conv4',
    'fc1': 'FC1',
    'fc2': 'FC2',
}


def main():
    parser = argparse.ArgumentParser(description="Plot PC quadrant comparison")
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--layer', type=str, default='fc2',
                        choices=['conv4', 'fc1', 'fc2'])
    args = parser.parse_args()

    # Load analysis data
    data_path = os.path.join(SCRIPT_DIR, f'data_{args.n_classes}way.npz')
    data = np.load(data_path)

    layer = args.layer
    pretrained_pcs = data[f'{layer}_pretrained_pcs']
    trained_pcs = data[f'{layer}_trained_pcs']
    quadrants = data[f'{layer}_quadrants']
    pretrained_var = data[f'{layer}_pretrained_var']
    trained_var = data[f'{layer}_trained_var']
    n_classes = int(data['n_classes'])

    layer_label = LAYER_LABELS.get(layer, layer)

    # --- Style ---
    sns.set_theme(style='ticks', context='paper', font_scale=1.2)

    # Colorblind-friendly (ColorBrewer Dark2)
    colors = ['#1b9e77', '#7570b3', '#e6ab02', '#d95f02']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    panels = [
        (ax1, pretrained_pcs, pretrained_var,
         f'Pretrained AlexNet (1000-way)', 'a'),
        (ax2, trained_pcs, trained_var,
         f'Trained AlexNet ({n_classes}-way)', 'b'),
    ]

    for ax, pcs, var, title, panel_label in panels:
        for q in range(4):
            mask = quadrants == q
            ax.scatter(pcs[mask, 0], pcs[mask, 1], c=colors[q],
                       alpha=0.30, s=2, edgecolors='none', rasterized=True)

        ax.set_xlabel(f'PC 1 ({var[0]:.1f}% var.)', fontsize=15)
        ax.set_ylabel(f'PC 2 ({var[1]:.1f}% var.)', fontsize=15)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax.tick_params(labelsize=13, width=1.8)

        # Panel label
        ax.text(-0.12, 1.08, panel_label, transform=ax.transAxes,
                fontsize=22, fontweight='bold', va='top')

    fig.suptitle(layer_label, fontsize=20, fontweight='bold', y=1.04)

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
    sns.despine(right=True, top=True, offset=8)

    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR,
                               f'pc_quadrant_pretrained_vs_{n_classes}way_{layer}.png')
    plt.savefig(output_path, dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved figure to {output_path}")


if __name__ == '__main__':
    main()
