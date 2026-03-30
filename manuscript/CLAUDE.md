# Manuscript

This folder contains private manuscript and discussion materials. It is excluded from Git (via `.git/info/exclude`) and should never be committed or pushed.

## Structure

```
manuscript/
├── figures/
│   ├── paper.md               # Master figure plan — layouts, observed results, design notes
│   ├── fig_utils.py           # Shared constants, style, helpers
│   ├── things_utils.py        # Shared THINGS plotting utilities
│   ├── fig1/                  # Figure 1: Method overview + representation analysis
│   ├── fig2/                  # Figure 2: Neural alignment (TVSD + NSD)
│   ├── fig3/                  # Figure 3: THINGS behavioral alignment
│   ├── fig4/                  # Figure 4: Per-concept alignment (RDMs + scatter + histogram)
│   ├── fig5/                  # Figure 5: Data efficiency
│   └── supplementary/         # Supplementary figures (S1–S18), see README.md inside
├── talk/
│   ├── talk_plan.md           # Slide-by-slide plan (content, visuals, sources)
│   ├── shared.py              # Talk-specific style (larger fonts) + shared helpers
│   ├── fig_*.py               # Standalone figure generators (each run independently)
│   └── figs/                  # Generated PNGs (300 DPI)
├── discussion/
│   └── {date}.md              # Supervisor discussion transcripts, named by date (e.g., 17feb2026.md)
├── methods.md                 # Methods section draft
├── NeurIPS2025_submission.md  # NeurIPS 2025 submission notes
└── claude.md                  # This file — context for Claude Code
```

## Guidelines

- **All figures must be saved at 300 DPI** (`dpi=300` in `savefig`). This applies to both paper figures and talk figures.
- **discussion/**: Each file is a transcript of a meeting with the supervisor, named `{DD}{mon}{YYYY}.md` (e.g., `20feb2026.md`). Use these for context on project direction and feedback.
- **figures/paper.md**: The single source of truth for figure layouts, panel descriptions, observed results, and design notes. Always read this before modifying any figure script.
- **methods.md**: Working draft of the methods section.

## Figure Status

All 5 main figures (Fig. 1–5) are up to date. See `figures/paper.md` for detailed layouts and observed results.

| Figure | Script | Description |
|--------|--------|-------------|
| Fig 1 | `fig1/figure1.py` | Method overview (PCA scatter) + learned representation scatter |
| Fig 2 | `fig2/figure2.py` | TVSD + NSD neural alignment (2×3 with schematics) |
| Fig 3 | `fig3/figure3.py` | THINGS: schematic + coarseness + model comparison + PC scatter |
| Fig 4 | `fig4/figure4.py` | RDMs (behavioral vs coarse vs 1K) + per-concept scatter + histogram |
| Fig 5 | `fig5/figure5.py` | Architecture generalization: THINGS coarseness for ResNet-50, ConvNeXt, ViT-B/16 |

## Talk Figures

`talk/` contains standalone figure generators for talk images — individual PNG panels (300 DPI) to be placed into presentation software. Each `fig_*.py` script runs independently from the project root (e.g., `python manuscript/talk/fig_things.py`).

**Key principle: reuse manuscript figure code directly.** Import plotting functions, constants, and data-loading code from `manuscript/figures/` — never reimplement. The only differences from paper figures are larger fonts (`setup_talk_style()`), individual panels, progressive reveal variants, and CLIP-only labels.

| Script | Output |
|--------|--------|
| `fig_label_space.py` | 1K colored, 1K gray, PCA method (from Fig 1A) |
| `fig_representations.py` | 1000-way vs 4-way learned representations (from Fig 1B) |
| `fig_things.py` | THINGS baseline + full progressive reveal (from Fig 3B) |
| `fig_pc_scatter.py` | Model comparison + 3-panel PC scatter (from Fig 3C–D) |
| `fig_data_efficiency.py` | Data efficiency line plot (from Fig 5) |

See `talk/talk_plan.md` for the full slide-by-slide plan.
