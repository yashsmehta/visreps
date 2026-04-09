# Manuscript

This folder contains manuscript and discussion materials for the paper.

## Structure

```
manuscript/
├── figures/
│   ├── paper.md               # Master figure plan — layouts, observed results, design notes
│   ├── fig_utils.py           # Shared constants, style, helpers
│   ├── things_utils.py        # Shared THINGS plotting utilities
│   ├── fig1/                  # Figure 1: Schematic
│   ├── fig2/                  # Figure 2: Categorical nature of representations
│   ├── fig3/                  # Figure 3: Neural alignment (TVSD + NSD)
│   ├── fig4/                  # Figure 4: THINGS behavioral alignment
│   ├── fig5/                  # Figure 5: Per-concept alignment (RDMs + scatter + histogram)
│   ├── fig6/                  # Figure 6: Architecture generalization
│   └── extended_data/         # Extended Data Figs. 1–7 (figures + captions only; index: extended_data.md)
├── supplementary_information.md  # Supplementary Notes (narrative text, no figures) — references Extended Data by number
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

All 6 main figures (Fig. 1–6) are up to date. See `figures/paper.md` for detailed layouts and observed results.

| Figure | Script | Description |
|--------|--------|-------------|
| Fig 1 | `fig1/plot_label_space.py` | Schematic (method and experimental pipeline overview) |
| Fig 2 | `fig2/figure2.py` | Categorical nature of representations (PCA scatter + learned representation scatter) |
| Fig 3 | `fig3/figure3.py` | TVSD + NSD neural alignment (2×3 with schematics) |
| Fig 4 | `fig4/figure4.py` | THINGS: schematic + coarseness + model comparison + PC scatter |
| Fig 5 | `fig5/figure5.py` | RDMs (behavioral vs coarse vs 1K) + per-concept scatter + histogram |
| Fig 6 | `fig6/figure6.py` | Architecture generalization: THINGS coarseness for ResNet-50, ConvNeXt, ViT-B/16 |

## Nature (main journal) submission rules

Target is **Nature** — the flagship journal, *not* Nature Communications or a Nature subjournal (different limits). The rules that actually shape the content we're writing now:

- **Article:** ≤ 3,500 words main text; ~150-word unreferenced abstract; **≤ 6 main display items** (figures + tables) — we are at 6, so no adding a Fig. 7; ~50 references as a guideline. Don't use "Introduction" as a heading.
- **Methods:** ≤ ~3,000 words; must include a **Data Availability** statement and a **Code Availability** statement; stats subsection must report exact *n*, exact *P*, *F*/*t* with df. Methods-only references don't count against the main reference limit.
- **Figure legends (main + Extended Data): < 250 words each.** Start with a title sentence, describe what is *depicted* (not results or methods), must be understandable in isolation.
- **Extended Data:** peer-reviewed, **online-only** (not in print). Up to **10** items (figures + tables combined); we have 7. Must be cited by name in the main text. Final formatting (file naming, vector vs JPEG/TIFF, sizing) is a submission-time task — we don't need to worry about it while drafting.
- **Supplementary Information:** peer-reviewed, online-only, narrative only. **Critical Nature-specific rule:** *"Any figures or small tables should ideally be supplied as Extended Data, not Supplementary Information."* This is why our SI is text-only and all 7 figures live in Extended Data. Must be referenced at least once in the main text.

Where things live in this repo:

| Section | Our file | Contains |
|---|---|---|
| Main text | `paper.md` | ≤ 3,500 words, 6 main figures |
| Methods | `methods.md` | ≤ ~3,000 words + Data/Code Availability |
| Extended Data | `figures/extended_data/` (index: `extended_data.md`) | 7 ED figures + captions, no narrative |
| Supplementary Information | `supplementary_information.md` | 7 Supplementary Notes, text only, no figures |

Source: [Brief guide for submission to Nature (PDF)](https://www.nature.com/documents/nature_3a_initial_revised_submissions.pdf) and [Formatting guide | Nature](https://www.nature.com/nature/for-authors/formatting-guide). Detailed final-submission formatting (300 dpi, Arial 5–7 pt, ≤ 10 MB ED files, `Surname_EDfig1.jpg` naming, SIGuide.doc, etc.) lives in Nature's guides — fetch them again at submission time rather than mirroring them here.

## Talk Figures

`talk/` contains standalone figure generators for talk images — individual PNG panels (300 DPI) to be placed into presentation software. Each `fig_*.py` script runs independently from the project root (e.g., `python manuscript/talk/fig_things.py`).

**Key principle: reuse manuscript figure code directly.** Import plotting functions, constants, and data-loading code from `manuscript/figures/` — never reimplement. The only differences from paper figures are larger fonts (`setup_talk_style()`), individual panels, progressive reveal variants, and CLIP-only labels.

| Script | Output |
|--------|--------|
| `fig_label_space.py` | 1K colored, 1K gray, PCA method (from Fig 2A) |
| `fig_representations.py` | 1000-way vs 4-way learned representations (from Fig 2B) |
| `fig_things.py` | THINGS baseline + full progressive reveal (from Fig 4B) |
| `fig_pc_scatter.py` | Model comparison + 3-panel PC scatter (from Fig 4C–D) |
| `fig_data_efficiency.py` | Data efficiency line plot (from Fig 6) |

See `talk/talk_plan.md` for the full slide-by-slide plan.
