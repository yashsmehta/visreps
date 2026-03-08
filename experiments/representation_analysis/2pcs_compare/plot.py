"""
Plot PC1 histogram for pretrained vs coarse-trained AlexNet.

Images are sorted by PC1 value and displayed in an ordered filmstrip above
the histogram, making the semantic gradient (e.g., animate → inanimate)
immediately visible.

Usage (from project root):
    python experiments/representation_analysis/2pcs_compare/plot.py --n_classes 2 --tag clip
    python experiments/representation_analysis/2pcs_compare/plot.py --n_classes 2 --tag clip --no_images
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import gaussian_kde
import seaborn as sns
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LAYER_LABELS = {
    'conv4': 'Conv 4',
    'fc1': 'FC 1',
    'fc2': 'FC 2',
}

PALETTES = {
    2: ['#3274a1', '#e1812c'],
    4: ['#2d8e6f', '#6a5acd', '#d4960a', '#d6503f'],
}

TAG_LABELS = {
    'clip': 'CLIP',
    'alexnet': 'AlexNet',
    'dino': 'DINO',
}


def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def make_bordered_thumbnail(img_path, thumb_size=44, border_width=3, border_color='#000'):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
    outer = 2
    rgb = hex_to_rgb(border_color)
    total = thumb_size + 2 * (border_width + outer)
    canvas = Image.new('RGB', (total, total), (255, 255, 255))
    inner = Image.new('RGB',
                      (thumb_size + 2 * border_width, thumb_size + 2 * border_width), rgb)
    canvas.paste(inner, (outer, outer))
    canvas.paste(img, (border_width + outer, border_width + outer))
    return np.array(canvas)


def sample_along_pc1(pc1, labels, n_per_class=10, lo_pct=5, hi_pct=95, seed=42):
    """Sample images evenly along PC1 within each class."""
    rng = np.random.default_rng(seed)
    selected = []
    for c in np.unique(labels):
        class_idx = np.where(labels == c)[0]
        pc1_vals = pc1[class_idx]
        edges = np.linspace(np.percentile(pc1_vals, lo_pct),
                            np.percentile(pc1_vals, hi_pct), n_per_class + 1)
        for i in range(n_per_class):
            bin_mask = (pc1_vals >= edges[i]) & (pc1_vals < edges[i + 1])
            candidates = class_idx[bin_mask]
            if len(candidates) > 0:
                selected.append(rng.choice(candidates))
    return np.array(selected)


def draw_filmstrip(ax, pc1_values, labels, img_paths, colors,
                   thumb_size=44, border_width=3, n_cols=10):
    """Draw a sorted filmstrip of images — left to right by PC1 value.

    Images are evenly spaced in a grid, sorted by PC1.
    """
    n_images = len(pc1_values)
    n_rows = int(np.ceil(n_images / n_cols))

    # Sort by PC1
    order = np.argsort(pc1_values)

    # Layout grid positions (normalized 0-1 in both axes)
    x_margin = 0.04
    y_margin = 0.08
    x_positions = np.linspace(x_margin, 1 - x_margin, n_cols)
    y_positions = np.linspace(1 - y_margin, y_margin, n_rows)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    zoom = 0.50 if n_cols <= 10 else 0.42

    for rank, idx in enumerate(order):
        row = rank // n_cols
        col = rank % n_cols
        if row >= n_rows:
            break

        label = labels[idx]
        thumb = make_bordered_thumbnail(
            str(img_paths[idx]),
            thumb_size=thumb_size,
            border_width=border_width,
            border_color=colors[label],
        )
        im = OffsetImage(thumb, zoom=zoom)
        ab = AnnotationBbox(im, (x_positions[col], y_positions[row]),
                            frameon=False, pad=0)
        ax.add_artist(ab)


def main():
    parser = argparse.ArgumentParser(description="Plot PC1 histogram with sorted filmstrip")
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--tag', type=str, default='clip')
    parser.add_argument('--layer', type=str, default='fc2',
                        choices=['conv4', 'fc1', 'fc2'])
    parser.add_argument('--no_images', action='store_true')
    parser.add_argument('--n_per_class', type=int, default=10)
    parser.add_argument('--thumb_size', type=int, default=48)
    args = parser.parse_args()

    show_images = not args.no_images

    # Load data
    data_path = os.path.join(SCRIPT_DIR, f'data_{args.n_classes}way_{args.tag}.npz')
    data = np.load(data_path, allow_pickle=True)

    layer = args.layer
    pretrained_pcs = data[f'{layer}_pretrained_pcs']
    pretrained_var = data[f'{layer}_pretrained_var']
    trained_pcs = data[f'{layer}_trained_pcs']
    trained_var = data[f'{layer}_trained_var']
    pca_labels = data['pca_labels']
    n_classes = int(data['n_classes'])
    img_paths_arr = data['img_paths'] if show_images else None

    colors = PALETTES.get(n_classes, PALETTES[4][:n_classes])
    tag_label = TAG_LABELS.get(args.tag, args.tag.upper())
    layer_label = LAYER_LABELS.get(layer, layer)

    # Sample images — use SAME images for both panels so you can compare ordering
    if show_images:
        sample_idx = sample_along_pc1(
            trained_pcs[:, 0], pca_labels, n_per_class=args.n_per_class, seed=42)
        print(f"Sampled {len(sample_idx)} images (same set for both panels)")

    # --- Figure setup ---
    sns.set_theme(style='ticks', context='paper')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
    })

    if show_images:
        fig = plt.figure(figsize=(12, 6.5))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.65],
                               hspace=0.35, wspace=0.30)
    else:
        fig, axes_hist = plt.subplots(1, 2, figsize=(12, 3.5))

    panels = [
        (0, pretrained_pcs[:, 0], pretrained_var,
         f'Pretrained (1000-way) \u2014 {layer_label}', 'a'),
        (1, trained_pcs[:, 0], trained_var,
         f'{tag_label}-coarsened ({n_classes}-way) \u2014 {layer_label}', 'b'),
    ]

    for col, pc1, var, title, panel_label in panels:
        if show_images:
            ax_img = fig.add_subplot(gs[0, col])
            ax_hist = fig.add_subplot(gs[1, col])
        else:
            ax_hist = axes_hist[col]
            ax_img = None

        # --- Histogram + KDE ---
        n_bins = 60
        for c in range(n_classes):
            mask = pca_labels == c
            ax_hist.hist(pc1[mask], bins=n_bins, alpha=0.35, color=colors[c],
                         edgecolor='none', density=True)
            kde = gaussian_kde(pc1[mask], bw_method=0.15)
            x_grid = np.linspace(pc1.min(), pc1.max(), 300)
            ax_hist.plot(x_grid, kde(x_grid), color=colors[c],
                         linewidth=2.0, alpha=0.9)

        ax_hist.set_xlabel(f'PC 1  ({var[0]:.1f}% var.)', fontsize=11.5, labelpad=6)
        ax_hist.set_yticks([])
        ax_hist.tick_params(labelsize=9.5)
        for spine in ['top', 'right', 'left']:
            ax_hist.spines[spine].set_visible(False)

        # --- Sorted filmstrip ---
        if show_images and ax_img is not None:
            # Sort the same images by THIS model's PC1
            pc1_sample = pc1[sample_idx]
            draw_filmstrip(ax_img, pc1_sample, pca_labels[sample_idx],
                           img_paths_arr[sample_idx], colors,
                           thumb_size=args.thumb_size, n_cols=10)

            # Subtle arrow: low PC1 → high PC1
            ax_img.annotate('', xy=(0.96, -0.04), xytext=(0.04, -0.04),
                            xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='#aaaaaa',
                                            lw=1.0))
            ax_img.text(0.5, -0.12, 'low PC 1 \u2192 high PC 1',
                        transform=ax_img.transAxes,
                        ha='center', fontsize=8.5, color='#999999')

        # Title
        title_ax = ax_img if (show_images and ax_img is not None) else ax_hist
        title_ax.set_title(title, fontsize=13, fontweight='bold', pad=8)

        # Panel label
        title_ax.text(-0.06, 1.08, panel_label, transform=title_ax.transAxes,
                      fontsize=18, fontweight='bold', va='top',
                      path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    # Legend
    class_names = [f'Class {c}' for c in range(n_classes)]
    handles = [mpatches.Patch(facecolor=colors[c], edgecolor='#555555',
                              linewidth=0.6, label=class_names[c])
               for c in range(n_classes)]
    fig.legend(handles=handles, loc='lower center', ncol=n_classes,
               fontsize=9.5, frameon=False, handlelength=1.4, handletextpad=0.5,
               columnspacing=1.5, bbox_to_anchor=(0.5, -0.03))

    if not show_images:
        sns.despine(offset=6)
        plt.tight_layout()

    suffix = '' if show_images else '_noimages'
    output_path = os.path.join(
        SCRIPT_DIR,
        f'pc1_hist_pretrained_vs_{n_classes}way_{args.tag}_{layer}{suffix}.png')
    plt.savefig(output_path, dpi=600,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
