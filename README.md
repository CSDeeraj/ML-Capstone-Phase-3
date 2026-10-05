# ML Capstone Phase 3 — Smart Imaging Lab

Reproduction and extension of **Tungal et al., “Artificial Intelligence Integrated Smart Medical Imaging Lab Framework for Enhanced Diagnosis and Treatment of Pandemic-Prone Diseases”**, *Health Science Reports* 2026;9:e71972 — with an animated showcase website.

* **Imaging:** the paper's 74,018-parameter 16-layer CNN (X-ray + CT) rebuilt exactly, plus **CovidNet-Plus** (residual + squeeze-excitation, augmentation, label smoothing, one-cycle, TTA, calibrated ensemble), Grad-CAM / Grad-CAM++.
* **Blood severity:** the paper's seven models, re-evaluated with **patient-grouped** validation (the paper's row-wise CV after SMOTE-ENN leaks), plus engineered ratios, age/sex, nested Optuna, OOF stacking, SHAP.
* **Website:** `site/` — open `site/index.html` (or `python -m http.server --directory site`).

Status, results and next steps: see [HANDOFF.md](HANDOFF.md).

## Evaluation documents (Phase 3 rubric)
* [docs/LITERATURE.md](docs/LITERATURE.md) — problem, literature survey, nine research gaps, future directions
* [docs/DATA.md](docs/DATA.md) — data sources, every inconsistency found and how it is handled, feature justification
* [docs/PRESENTATION.md](docs/PRESENTATION.md) — timed talk script, demo plan, viva question bank
* [docs/RUBRIC.md](docs/RUBRIC.md) — where the evidence for each rubric row lives
* Test cases: `src/test_cases.py` (blood, T1–T9), `src/imaging_tests.py` (corruptions + shortcut tests), `python -m pytest -q` (unit checks)

## Run on a new PC (Windows / Linux / macOS)
```bash
git clone https://github.com/CSDeeraj/ML-Capstone-Phase-3.git   # keep the path short on Windows
cd ML-Capstone-Phase-3
python setup_env.py                     # creates .venv; installs CUDA or CPU PyTorch automatically
# activate:  .venv\Scripts\activate   (Windows)   or   source .venv/bin/activate
python src/run_all.py --profile full    # GPU: paper's settings (100 epochs, 10-fold CV, lr 1e-4, batch 16)
python src/run_all.py --profile quick   # CPU: scaled-down budget
```
`run_all.py` downloads the data (resumable, ~4 GB), caches images, trains X-ray + CT + blood models and refreshes the website data.
Individual steps: `src/download_data.py`, `src/prepare_images.py`, `src/data_audit.py`, `src/imaging.py --modality xray|ct`, `src/imaging_tests.py --modality xray|ct`, `src/blood.py`, `src/test_cases.py`, `src/export_site.py` (`--skip-tests` on `run_all.py` skips the test suites).
A GPU run can be checked first with `python setup_env.py --check`.
Device is auto-detected (CUDA → Apple MPS → CPU). Data goes to `./data` or `$CAPSTONE_DATA` (git-ignored).

## Data (public)
* Extensive COVID-19 X-ray and CT Chest Images — Menoufia University, [Mendeley Data](https://data.mendeley.com/datasets/8h65ywd2jr/3)
* BIGDATA-COVID19 blood markers — San Raffaele Hospital, [Zenodo 4686707](https://zenodo.org/record/4686707)

Research demonstration only — not a medical device.
