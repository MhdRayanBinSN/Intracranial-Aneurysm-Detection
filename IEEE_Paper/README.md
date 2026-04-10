# IEEE Journal Paper — Intracranial Aneurysm Detection

## Folder Structure
```
IEEE_Paper/
├── main.tex          ← Main paper (IEEE two-column format)
├── figures/          ← Put your figures here (.png / .pdf)
│   ├── architecture.png
│   ├── heatmap_example.png
│   └── results_plot.png
└── README.md
```

## How to Compile

### Option 1 — VS Code (Recommended)
1. Install **LaTeX Workshop** extension
2. Open `main.tex`
3. Press `Ctrl+Alt+B` to compile
4. Compile **4 times** on first run (for bibliography):
   `pdflatex → bibtex → pdflatex → pdflatex`

### Option 2 — Command Line
```bash
cd "C:\Users\Rayan\Desktop\Main Project\IEEE_Paper"
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Paper Structure

| Section | Content |
|---------|---------|
| Abstract | 250-word summary of the work |
| I. Introduction | Motivation, clinical burden, contributions |
| II. Medical Background | IA pathophysiology, anatomy, imaging modalities |
| III. Related Work | Prior CAD systems, U-Net variants, VLMs |
| IV. Dataset | RSNA 2025 — 4,348 series, statistics, HU values |
| V. Methodology | Preprocessing, nnU-Net arch, blob regression, MedGemma |
| VI. Evaluation Pipeline | ZIP-based streaming evaluation, metrics |
| VII. Results & Discussion | Quantitative, per-location, failure modes |
| VIII. Conclusion | Summary and future work |

## What to Do Next

1. **Add figures** — Place your architecture diagram, heatmap examples,
   and results plots in the `figures/` folder and uncomment the
   `\includegraphics` lines in `main.tex`

2. **Fill in results table** — Replace `--` placeholders in Table II
   with your actual metric values from `evaluation_results.json`

3. **Target Journal** — This template is formatted for:
   - *IEEE Journal of Biomedical and Health Informatics (JBHI)*
   - *IEEE Transactions on Medical Imaging (TMI)*
   - *Computers in Biology and Medicine* (Elsevier — change template)

## IEEE Submission Checklist
- [ ] All figures at 300 DPI minimum
- [ ] No colour-coded content that loses meaning in greyscale
- [ ] Author bios and photos (some IEEE journals require these)
- [ ] Ethics statement if using patient data
- [ ] Conflict of interest statement
- [ ] Figures folder zipped separately for submission
