# Supplementary Figure Style Guide

Reference for maintaining visual consistency with the main manuscript figures (Fig 1–6). All supplementary figures should follow these conventions unless there is a specific reason to deviate.

---

## Global Theme

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="ticks", context="paper", font_scale=1.05)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10.5,
    "axes.titlesize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
```

All style definitions come from `manuscript/figures/fig_utils.py` and `manuscript/figures/things_utils.py`. Import from these rather than redefining constants.

---

## Color Palettes

### Granularity (coarseness) — blue gradient + orange-red for 1000-way

| Classes | Hex | Role |
|---------|-----|------|
| 2 | `#c6dbef` | Lightest blue |
| 4 | `#9ecae1` | |
| 8 | `#6baed6` | |
| 16 | `#4292c6` | |
| 32 | `#2171b5` | |
| 64 | `#084594` | Darkest blue |
| 1000 | `#e6550d` | Orange-red (baseline) |

Defined in `fig_utils.GRAN_COLORS`. Use consistently for any coarseness comparison.

### Granularity markers

| Classes | Marker |
|---------|--------|
| 2 | `o` (circle) |
| 4 | `s` (square) |
| 8 | `^` (up triangle) |
| 16 | `D` (diamond) |
| 32 | `v` (down triangle) |
| 64 | `p` (pentagon) |
| 1000 | `X` (filled X) |

Defined in `fig_utils.GRAN_MARKERS`.

### Architecture colors

| Architecture | Color | Marker |
|---|---|---|
| AlexNet | `#1a9e76` (teal) | `o` |
| CLIP | `#7b3294` (purple) | `s` |
| ViT | `#d62728` (red) | `^` |
| Pixels | `#8c564b` (brown) | `v` |

Defined in `fig_utils.ARCH_STYLE`. 1000-way baseline across architectures uses `#e6550d` with diamond marker.

### Untrained baseline

- Color: `#AAAAAA` (gray)
- Linestyle: dashed `--`
- Linewidth: 1.1
- Alpha: 0.7

### THINGS super-category palette (28 categories)

Defined in `things_utils.PALETTE_28`. Use for any per-category coloring of THINGS concepts.

### Advantage / difference coloring (per-concept analyses)

| Condition | Color |
|---|---|
| Strong coarse advantage (> 0.3) | `#2e7d32` (dark green) |
| Mild coarse advantage (0 to 0.3) | `#a5d6a7` (light green) |
| No difference (0) | `#c8c8c8` (grey) |
| Mild 1K advantage (-0.3 to 0) | `#f4a261` (light orange) |
| Strong 1K advantage (< -0.3) | `#e65100` (dark orange) |

---

## Axis & Spine Conventions

### Despining
- Always remove right and top spines: `sns.despine(ax=ax, offset=3)` (offset 3–5)
- Left and bottom spines kept, linewidth 0.8

### Tick formatting
- Major tick width: 0.6–0.8, length: 3–4
- Minor tick width: 0.4–0.6, length: 2–2.5
- Minor y-axis locator: `AutoMinorLocator(2)`

### Grid
- Major y-axis grid only: color `#EBEBEB`, linewidth 0.3–0.4, zorder 0
- No x-axis grid

### Coarseness x-axis (log scale with broken axis)
- Log base-2 scale for granularity values (2, 4, 8, 16, 32, 64)
- Visual break (`//` marks) before 1000 using `fig_utils.draw_xaxis_break()`
- 1000 placed at a synthetic log-position (BREAK_1K_POS)
- Use `fig_utils.format_coarseness_axes()` for this formatting

### Normalized coarseness x-axis (% of 1000-way)
- Use `fig_utils.format_normalized_coarseness_axes()` when showing relative performance

---

## Markers & Points

- **Marker size**: 7 (`fig_utils.MARKER_SIZE`)
- **Edge color**: white
- **Edge width**: 0.6
- Scatter alpha: 0.62–0.75
- Rasterize scatter points in dense plots (`rasterized=True`)

---

## Error Bars & Confidence Intervals

- Bootstrap 95% CIs (element-wise averaged across seeds, then 2.5th/97.5th percentiles)
- Error bar styling: capsize 1.5, capthick 0.5, linewidth 0.7–1.0
- Fill-between CI bands: alpha 0.15
- Fallback (no bootstrap): +/- 1.96 x SEM across seed means

---

## Line Styles

| Element | Style | Linewidth | Alpha |
|---|---|---|---|
| Data curves | solid `-` | 1.5 | 1.0 |
| 1000-way reference | dashed `--` | 0.6–1.0 | 0.25–0.6 |
| Untrained baseline | dashed `--` | 0.9–1.1 | 0.7 |

---

## Panel Labels (a, b, c, ...)

- **Lowercase bold letters** (not uppercase)
- Fontsize: 13–14 (use 13 for multi-panel, up to 20 for large panels like Fig 5)
- Font: sans-serif, bold
- Position: slightly outside the axes upper-left corner, using `ax.transAxes` with negative x offset (~-0.06 to -0.18) and y slightly above 1.0 (~1.05–1.10)

---

## Titles & Labels

### Panel titles
- Fontsize: 12–13, bold, pad 8–12
- Font: sans-serif

### Subtitles (below or above titles)
- Fontsize: 8.5–9.5
- Color: `#444444` to `#777777`
- Style: italic

### Axis labels
- Fontsize: 10–11
- Labelpad: 1–4

---

## Legends

- Fontsize: 7–8.5
- Frame: on, fancybox False
- Framealpha: 0.90–0.95
- Edgecolor: `#bbbbbb` to `#dddddd`
- Frame linewidth: 0.3
- borderpad: 0.3–0.5
- handletextpad: 0.3–0.4
- labelspacing: 0.2–0.3
- Position: context-dependent, typically upper-left or right side

---

## RDM (Representational Dissimilarity Matrix) Panels

- Colormap: `magma` (data RDMs), `RdBu_r` (difference RDMs)
- Value range: vmin=0, vmax=1 (magma)
- Interpolation: nearest
- Aspect: equal
- Category boundary lines: white, lw 0.45, alpha 0.80
- Category sidebar: width_frac=0.032–0.045, gap_frac=0.005, colored by super-category
- Category labels: fontsize 7, color `#333333`, right-aligned

---

## Export Conventions

- **DPI**: 300 (standard), 200 acceptable for very large figures
- **Format**: PNG primary, PDF/SVG for vector when needed
- **bbox_inches**: `"tight"`
- **facecolor**: `"white"`
- **edgecolor**: `"none"`
- **Output directory**: each figure's own `figures/` subfolder

---

## Figure Sizing

Main figures use cm-scale widths matching Nature's column widths, but specified in inches via matplotlib:

| Type | Approximate Width |
|---|---|
| Single panel | 4–5 inches |
| Two-panel row | 10–14 inches |
| Full-width multi-panel | 13–17 inches |

Supplementary figures should use similar dimensions for consistency.

---

## Data Presentation Patterns

### Coarseness curves (Figs 3, 4, 6)
- X-axis: number of training classes (log scale, broken axis to 1000)
- Y-axis: alignment score (RSA/encoding/behavioral)
- One curve per architecture or condition
- Connected markers with CI error bars or fill-between bands
- Untrained baseline as horizontal dashed gray line

### Per-layer profiles
- 14 positions (pre/post ReLU for 7 layers: conv1–conv5, fc1–fc2)
- Labels shown only at post-ReLU positions
- Layer names: `["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]`

### Scatter plots (concept-level)
- Aspect ratio: equal
- Diagonal reference line (y=x): gray, dashed
- Points colored by super-category or advantage magnitude
- Edge: white, width 0.3

### Histograms / KDE (difference distributions)
- Zero reference line: color `#555555`, lw 0.7, solid
- Overall KDE: color `#444444`, lw 2.0
- Fill: green (positive/coarse advantage), orange (negative/1K advantage), alpha 0.18

---

## Checklist for New Supplementary Figures

1. Import style from `fig_utils.py` / `things_utils.py` — do not redefine colors or markers
2. Apply `sns.set_theme(style="ticks", context="paper", font_scale=1.05)` and rcParams
3. Use `GRAN_COLORS` / `GRAN_MARKERS` / `ARCH_STYLE` for consistent color/marker encoding
4. Despine with offset 3–5, add y-axis grid (`#EBEBEB`, lw 0.3)
5. Use broken x-axis via `format_coarseness_axes()` for coarseness plots
6. Panel labels: lowercase bold, fontsize 13, positioned outside axes
7. Export at 300 DPI, tight bbox, white background
8. Error bars from bootstrap CIs via `plotter_utils.get_condition_summary()`
