# Manuscript

This folder contains private manuscript and discussion materials. It is excluded from Git (via `.git/info/exclude`) and should never be committed or pushed.

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

## Nature (main journal) submission requirements

This project targets **Nature** (the flagship journal, *not* a Nature-family subjournal such as Nature Neuroscience or Nature Communications — those have different rules and different word limits). The rules below are the ones we must design against; violating them means last-minute restructuring after acceptance. All limits pulled from the official Nature guides (see Sources at the end of this section).

### Article format (what our paper is)

- **Abstract** ~150 words, unreferenced. Non-technical introduction to topic + brief summary of main results and their implication.
- **Main text** ≤ 3,500 words.
- **Display items in main text:** up to **6** (figures + tables combined). We have 6 main figures, so we are at the cap — no adding Fig. 7 without dropping one.
- **References** in main text: up to ~50 (guideline, not hard cap). Methods-only references do *not* count against this limit.
- **Section headings** are used; **subheadings** may appear in Results. Do **not** use "Introduction" as a heading — the opening text is unheaded.
- **Title** must fit on two printed lines: ≤ 90 characters (Articles used to be 75, now aligned with Letters — confirm at acceptance). Avoid technical terms, abbreviations, active verbs.

### Methods

- Separate Methods section, appears in online version.
- Target ≤ 3,000 words; may be longer if genuinely necessary. Written as concisely as possible.
- **Data Availability** statement is *mandatory* — include under a "Data availability" subheading inside Methods.
- **Code Availability** statement is *mandatory* when custom code is central to conclusions — include under "Code availability" inside Methods.
- Methods-only references are numbered continuously with main references but don't count against the main reference limit.
- Must include a **Statistics** subsection: tests used, one- or two-tailed, exact *n*, exact *P* values for both significant and non-significant tests, *F* / *t* values with degrees of freedom for ANOVA / t-tests, definitions of error bars and replicates.
- `manuscript/methods.md` is our working draft.

### Figure legends (main + Extended Data)

- **< 250 words each.** This is a hard style rule — longer legends get cut at copy-edit.
- Legend structure: (i) brief title sentence for the whole figure, (ii) short statement of what is *depicted*, not the results or the methods used, (iii) per-panel descriptions.
- Each figure and caption must be understandable **in isolation** from the main text.
- Per-panel descriptions inside each `figN/figureN_description.md` and `extended_data/SN_*/SN*_description.md` file should respect the < 250-word cap. Audit them before final submission.

### Figure preparation (applies to both main figures and Extended Data)

- **Resolution:** 300 dpi for rasterized content. Exceeding 300 dpi bloats files without improving appearance (online viewers time out).
- **Colour mode:** RGB. Avoid red/green contrasts and the rainbow colour map. Prefer accessible palettes (green/magenta etc.).
- **Typography:** sans-serif, **Arial or Helvetica** only, single typeface throughout. Symbol font for Greek letters. Maximum body text 7 pt, minimum 5 pt. Figure panels labelled with **8 pt bold lowercase** (`a`, `b`, `c`, …) — upright, not italic.
- **Line weights:** 0.25 – 1 pt.
- **Accepted formats (final submission):**
  - Main figures: vector preferred (`.ai`, `.eps`, `.pdf`, `.ps`, `.svg`) with editable layers. Layered `.psd` / `.tif` for editable layered art. Bitmap: `.psd`, `.tif`, `.png`, `.jpg`.
  - **Extended Data:** *only* `JPEG`, `TIFF`, or `EPS`. Rasterized or flattened. **Each file ≤ 10 MB.**
- **Figure size:** prepare at the size they will appear in print. At print size, 7 pt text is optimum. Our `fig_utils.py` `FigureLayout.scaled()` assumes print-size layout already.
- Our current pipeline saves PNG at 300 dpi from matplotlib; for final submission we will need to re-export main figures as vector PDFs and Extended Data as flat JPEG/TIFF ≤ 10 MB.

### Extended Data (our `manuscript/figures/extended_data/`)

- **Purpose:** peer-reviewed display items that "will not appear in print but are included in the online versions" (HTML + end of the online PDF). These *are* reviewed, so their content carries the same rigour as main figures.
- **Number limit:** up to **10 multi-panel display items** (figures + tables combined). We currently have 7 Extended Data figures — room for 3 more if needed, but we are not adding any.
- **Sizing:** max page dimension **183 mm × 247 mm** (per the Extended Data formatting PDF); each figure occupies its **own page**, centred, with its legend set below on the same page. One figure per page — never two.
- **Format:** JPEG, TIFF, or EPS only; rasterized/flattened; ≤ 10 MB per file.
- **File naming convention:** `CorrespondingAuthorSurname_EDfig1.jpg`, `CorrespondingAuthorSurname_EDfig2.jpg`, … (e.g. `Mehta_EDfig1.jpg`). At final submission we must rename the outputs from their internal `S1_*/S1a_neural.png` layout to this pattern. Tables use `_EDtable1.*`.
- **Must be cited as discrete items** in the main text (e.g. "(Extended Data Fig. 2)"). Every Extended Data figure has to be referenced at least once in the main text body or Methods.
- **Not subedited.** Nature's art department does not restyle Extended Data, so authors must follow the formatting guide themselves — our `extended_data/figure_style.md` needs to match Nature's rules before submission.
- **Legend location:** per-panel description files (`SN*_description.md`) are the source of truth; they will be assembled into the final caption under each ED figure. Keep each legend < 250 words.

### Supplementary Information (`manuscript/supplementary_information.md`)

- **Purpose:** "material that is essential background (e.g. large data sets and calculations) but too large, impractical or specialized to justify inclusion in the printed version." Think derivations, long tables, extended methodological discussion, code listings, large data summaries.
- **CRITICAL rule specific to Nature (main journal):** *"Any figures or small tables should ideally be supplied as Extended Data, not Supplementary Information."* This is why our 7 figures moved to Extended Data and SI now contains only narrative notes.
- **Peer-reviewed**, but **not subedited** by Nature's staff — authors are responsible for final presentation quality. Must be clearly and succinctly written; terminology should match the main paper.
- **Must be referenced** in the main text at least once (e.g. "(see Supplementary Information)").
- **File size:** single PDFs / sound / video files ≤ 30 MB per file; **cumulative total ≤ 150 MB** across all SI files.
- **SI Guide file required at final submission:** a separate `SIGuide.doc` (or text file) listing each SI file with a title and a ≤ 50-word summary of its contents. We'll need to generate this at submission time.
- **Numbering:** Supplementary Tables, Supplementary Figures, Supplementary Videos each have their own independent numbering, separate from main figures/tables *and* separate from Extended Data. Since we have *no* supplementary figures or tables (they are all Extended Data), this rule only matters if we later add a table.
- **Our current SI is text-only:** 7 Supplementary Notes referencing Extended Data Figs. 1–7. No images, no figure blocks. This matches Nature's preference.

### Distinction we must keep consistent

| Section | Contains | Peer reviewed? | In print? | Subedited? | Our file |
|---|---|---|---|---|---|
| Main text | ≤ 3,500 words, ≤ 6 display items | Yes | Yes | Yes | `paper.md` |
| Methods | ≤ ~3,000 words + mandatory Data/Code Availability | Yes | Online + print in supplementary sections of print PDF | Yes | `methods.md` |
| Extended Data | Up to 10 figures/tables, peer-reviewed online-only display items | Yes | Online only (HTML + online PDF) | **No** | `figures/extended_data/*` |
| Supplementary Information | Narrative notes, large datasets, derivations — *ideally no figures/tables* | Yes | Online only | **No** | `supplementary_information.md` |

### Things we still need to do before submission

1. Verify every main figure and Extended Data figure is cited by name in the main text of `paper.md`.
2. Verify `supplementary_information.md` is cited at least once in the main text.
3. Re-export main figures to vector (`.pdf`/`.eps`) and Extended Data to JPEG/TIFF ≤ 10 MB.
4. Rename Extended Data output files to `Mehta_EDfig{1..7}.jpg` / `.tif` / `.eps`.
5. Audit every figure caption for ≤ 250 words (main and Extended Data).
6. Add Data Availability + Code Availability statements to `methods.md`.
7. Generate `SIGuide.doc` with ≤ 50-word summary of the SI contents.
8. Confirm title ≤ 90 characters and fits two printed lines.
9. Confirm main text ≤ 3,500 words and Methods ≤ ~3,000 words.

### Sources (Nature main journal authoritative guides — last fetched 2026-04-09)

- [Formatting guide | Nature](https://www.nature.com/nature/for-authors/formatting-guide)
- [Supplementary information | Nature](https://www.nature.com/nature/for-authors/supp-info)
- [Brief guide for initial submissions to Nature (PDF)](https://www.nature.com/documents/nature_3a_initial_revised_submissions.pdf)
- [Extended Data formatting guide (PDF)](https://www.nature.com/documents/nature-extended-data.pdf)
- [Extended Data formatting guidelines | Nature research figure guide](https://research-figure-guide.nature.com/figures/extended-data-formatting-guidelines/)
- [Final submission | Nature](https://www.nature.com/nature/for-authors/final-submission)

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
