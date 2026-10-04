# ML Capstone Phase 3 — Smart Imaging Lab

Reproduction and extension of **Tungal et al., “Artificial Intelligence Integrated Smart Medical Imaging Lab Framework for Enhanced Diagnosis and Treatment of Pandemic-Prone Diseases”**, *Health Science Reports* 2026;9:e71972 — with an animated showcase website.

* **Imaging:** the paper's 74,018-parameter 16-layer CNN (X-ray + CT) rebuilt exactly, plus **CovidNet-Plus** (residual + squeeze-excitation, augmentation, label smoothing, one-cycle, TTA, calibrated ensemble), Grad-CAM / Grad-CAM++.
* **Blood severity:** the paper's seven models, re-evaluated with **patient-grouped** validation (the paper's row-wise CV after SMOTE-ENN leaks), plus engineered ratios, age/sex, nested Optuna, OOF stacking, SHAP.
* **Website:** `site/` — open `site/index.html` (or `python -m http.server --directory site`).

Status, results and next steps: see [HANDOFF.md](HANDOFF.md).

## Run
```bash
pip install -r requirements.txt            # install the torch build matching your machine
python src/run_all.py --profile quick      # CPU laptop
python src/run_all.py --profile full       # GPU: paper's settings (100 epochs, 10-fold, lr 1e-4, batch 16)
```
Device is auto-detected (CUDA → Apple MPS → CPU). Data goes to `./data` or `$CAPSTONE_DATA` (git-ignored).

## Data (public)
* Extensive COVID-19 X-ray and CT Chest Images — Menoufia University, [Mendeley Data](https://data.mendeley.com/datasets/8h65ywd2jr/3)
* BIGDATA-COVID19 blood markers — San Raffaele Hospital, [Zenodo 4686707](https://zenodo.org/record/4686707)

Research demonstration only — not a medical device.
