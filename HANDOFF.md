# Handoff — state as of 2026-10-04 (read this first)

## What this project is
Reproduction + extension of Tungal et al., *Smart Imaging Lab Framework…*, Health Science Reports 2026;9:e71972
(COVID-19 detection from X-ray/CT with a 16-layer CNN; severity from blood tests), plus a showcase website in `site/`.

## Done and verified
- Datasets downloaded and verified (Mendeley zip SHA-256 matches; counts match paper: X-ray 4044/5500, CT 5427/2628; blood 4430 rows after dropna = 3342/1088).
- `src/` pipeline: `download_data.py`, `prepare_images.py`, `imaging.py`, `blood.py`, `export_site.py`, `run_all.py`, `config.py`.
- **Blood (final, quick profile, patient-grouped 10-fold):** `results/blood.json`.
  - Paper protocol reproduced: ERT 97.58% acc (paper: 98.00%).
  - Honest protocol (patients kept apart, SMOTE-ENN inside folds): ERT 81.09% acc, AUC 0.870.
  - Ours (ratios + age/sex, nested Optuna, OOF stacking): AUC 0.904, balanced acc 81.9%, sensitivity 81.3% (paper recipe: 72.9%).
  - Key finding for the site: the paper's number is inflated by row-wise CV after SMOTE-ENN on the whole set (1136 patients, 4430 rows).
- **X-ray (final, quick profile, 1 held-out 10% fold):** `results/imaging_xray.json`.
  - Paper CNN reproduced: 87.85% (paper: 99.02% — NOT reproduced under CPU budget: 50 epochs max vs 100, lr 1e-3 vs 1e-4).
  - CovidNet-Plus ensemble (calibrated): 91.41%, AUC 0.976 (paper CNN AUC 0.952).
- Website `site/` complete and browser-tested (hero lung point cloud, journey, architecture, charts, Grad-CAM slider, live in-browser forest predictor, triage simulator, upgrades list, limits). Open `site/index.html` or `python -m http.server --directory site`.

## NOT done — do these next
1. **CT run was interrupted** (PC shut down). `results/imaging_ct.json` etc. were deliberately deleted because they were stale 1-epoch smoke outputs. Re-run:
   `python src/imaging.py --modality ct --profile quick` (~1.5 h on a 16-thread CPU; much faster on GPU).
2. Optionally re-run **both imaging modalities with `--profile full`** on a GPU (paper protocol: 100 epochs, 10-fold, lr 1e-4, batch 16). Expect X-ray/CT numbers to move toward the paper's; the site reads whatever is in `results/`.
3. After any rerun: `python src/export_site.py`, then check the site (`#imaging` section; the note above the KPIs auto-reports profile/epochs/folds).
4. Possible improvements if time: the CNNs under-fit at ~92% train acc in the quick profile — try more epochs / weaker augmentation for CovidNet-Plus; consider 128px input.
5. User asked about a **cloud run**: a cloud session likely has no GPU; only worth it if GPU is available. Check `nvidia-smi` first; otherwise run on the user's GPU PC.

## Rubric work (branch claude/rubric-excellent-54zut3, PR #1)
- Docs: `docs/LITERATURE.md`, `docs/DATA.md`, `docs/PRESENTATION.md`, `docs/RUBRIC.md`.
- To run on the PC with the images, then `python src/export_site.py`:
  `python src/data_audit.py` (image audit), `python src/imaging_tests.py --modality xray` and `--modality ct` (after imaging.py has saved models).
- Optional paper-identical image balancing: `python src/imaging.py --modality xray --balance smoteenn-all` (SMOTE-ENN on all images before CV, as published;
  needs several GB of RAM) and `--balance smoteenn` (training splits only). Results go to `results/imaging_<mod>_<balance>.json`; the default run is untouched.

## Setup on a new machine (the user will continue on a different PC, possibly with a GPU)
```
git clone https://github.com/CSDeeraj/ML-Capstone-Phase-3.git && cd ML-Capstone-Phase-3
python setup_env.py            # venv + CUDA/CPU PyTorch auto-picked;  `--check` reports GPU status
# activate .venv, then:
python src/run_all.py --profile full      # if GPU (paper settings);  --profile quick on CPU
```
Data (4 GB) is not in git: `src/download_data.py` fetches it, resumable + checksum-verified. The blood CSV is committed in `data/`.
Imaging results currently in `results/` came from a CPU (quick profile); a GPU `full` run will overwrite them — keep the old JSON if a comparison is wanted.
Note for Windows: keep the repo/data paths short (git failed once on a very long scratch path).

## Honesty rules for the site
Never present quick-profile numbers as the paper's; always show profile/epochs/folds. Keep the leakage finding and the failure-case in the Grad-CAM gallery.
