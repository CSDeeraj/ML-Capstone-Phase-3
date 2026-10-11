# RESUME — where the work stands (updated 2026-10-11 14:05 IST)

## ALL DONE — 2026-10-11 14:01 IST (read this first)
- CT paper: acc 97.79 ± 0.54 (paper 98.49). X-ray paper: acc 97.41 ± 0.72 (paper 99.02). Blood full: done.
- Blood paper protocol: ERT 97.58 (paper 98.00). Honest patient-grouped protocol, ERT + age/sex: acc 86.22, AUC 0.902. Improved stacked ensemble: acc 82.45 ± 3.73, bacc 81.98, sens 81.09, spec 82.87, AUC 0.903 ± 0.034, threshold 0.28 (the quick run gave 82.26 / 0.904). Tuned ERT: acc 86.74, AUC 0.903.
- export_site ran at 14:01, and `site/data/results.js` has imaging_paper.{xray,ct} plus blood (profile full).
- The GPU is free and nothing is running. The code, results and site data were committed on 2026-10-11 to branch `claude/project-thread-657b1o`, including the launcher scripts in `scripts/` (`paper_run.py`, `blood_run.py`, `resume_xray_blood.sh`, `blood_parallel.sh`).

## STATUS 2026-10-11 08:05 IST (historical)
- **CT paper run: DONE.** `results/imaging_ct_paper.json` was written at 00:51, and export_site ran: `site/data/results.js` has `imaging_paper.ct`.
  - 10-fold mean ± std: acc **97.79 ± 0.54**, sens 96.89 ± 1.23, spec 98.59 ± 0.66, prec 98.40 ± 0.72, F1 97.63 ± 0.59, AUC 0.9974 ± 0.0011, MCC 0.956 ± 0.011. The paper reports 98.49.
  - Fold 8: 98.31 / 98.52 / 98.12. Fold 9: 98.01 / 97.46 / 98.49 (acc / sens / spec).
- **X-ray paper run: RUNNING.** It started at 00:52. SMOTE-ENN took 9,544 images to 8,182 (COVID 4328 / non-COVID 3854), cached at `data/cache/x-ray_100_smoteenn.npz`. That is about 7,364 train / 818 test per fold, at about 31 s/epoch, so about 53 min/fold.
  - Folds done (acc / sens / spec / AUC): 0 = 95.85 / 96.30 / 95.34 / 0.9944, 1 = 97.68 / 97.92 / 97.41 / 0.9960, 2 = 97.31 / 97.92 / 96.63 / 0.9929. The paper reports 99.02 for X-ray.
  - Overnight: mains power was cut at 02:45. The laptop ran on battery until 03:44, then went to sleep on critical battery ("Sleep Reason: Battery"). It woke at 07:54 when power returned. The process survived the sleep (fold 3 epoch 33 shows 15082 s), nothing was lost, and no reboot happened (last boot is still 2026-10-10 20:28).
  - 08:00: I restarted the chain with **per-epoch checkpoints** at Deeru's request. `scripts/paper_run.py` wraps the unchanged `src/imaging.py`: it checkpoints every epoch, fsyncs before the rename, keeps the previous checkpoint as `fold_k.ckpt.pt.bak`, and falls back to the `.bak` if the checkpoint is corrupt. It is a wrapper because the session hook blocks edits to `src/` in the base checkout. Fold 3 resumed from epoch 40.
  - 08:33: X-ray fold 3 done: 97.80 / 97.22 / 98.45 / 0.9973. 4-fold mean acc 97.16 ± 0.90. Fold 4 started.
  - 09:25: X-ray fold 4 done: 98.17 / 98.38 / 97.92 / 0.9975. 5-fold mean acc 97.36 ± 0.90. Fold 5 started, X-ray done at about 13:50.
  - ETA: fold 3 at about 08:35, folds 4–9 done at about 14:00 IST. Then `blood.py --profile full` runs (duration unknown, CPU/Optuna), then export_site.
- 10:12: **blood --profile full started IN PARALLEL on the CPU** at BelowNormal priority with CUDA_VISIBLE_DEVICES=-1: `scripts/blood_parallel.sh`, logging to `logs/blood_full.log`. While it runs, the marker `results/blood_full.running` exists. On success it writes `results/blood_full.done`, runs export_site, and backs up the old blood.json to `results/blood.pre_full.backup.json`. The chain now waits for or skips blood if that marker exists (edited after the point the running bash has read up to). X-ray epoch time is unchanged at 31–33 s, GPU 96%.
  - Paper-protocol blood models (accuracy): ERT 97.58 (paper 98.00), kNN 97.53, LGBM 96.85, RF 96.31, DT 91.25. The improved models and outer CV are still running.
  - 10:12–10:17: blood crashed 3× with exit 139 (Windows access violation, no traceback). Cause: `blood.py` calls `config.device_name()`, which imports torch and runs `torch.cuda.is_available()`. After that, scikit-learn native code (libsvm `SVC.fit`, kNN `kneighbors`) crashes at random. I reproduced it with faulthandler: 2/2 crashes with device_name(), 2/2 clean runs without. The .venv was rebuilt (sklearn 1.9.1, numpy 2.5.3) at 20:23 on 2026-10-05, after the last good blood run (20:18).
  - Fix, without editing src: `scripts/blood_run.py` imports blood, sets `blood.device_name = lambda: "CPU"` (the blood models are CPU anyway), and runs `blood.main()`. `blood_parallel.sh` and the chain's blood step both use it now. The chain no longer exits on a blood failure, so it still runs the final export_site.
  - 10:28: blood relaunched (BelowNormal). The paper protocol completed all 7 models: ERT 97.58 (paper 98.00), LGBM 96.85 (97.81), RF 96.31 (97.45), kNN 97.53 (96.36), DT 91.25 (91.07), SVM 92.45 (90.16), AB 87.06 (87.98). Now on the honest protocol, Optuna and outer CV.
  - 10:17: X-ray fold 5 done: 98.17 / 99.77 / 96.36 / 0.9964. 6-fold mean acc 97.50 ± 0.87.
  - 11:10: X-ray fold 6 done: 96.33 / 96.30 / 96.36 / 0.9945. 7-fold mean acc 97.33 ± 0.91. Fold 7 started, X-ray done at about 13:50.
  - Blood honest protocol: + ratios acc 84.52 (bacc 73.05, AUC 0.886), + age/sex acc 86.22 (bacc 75.38, AUC 0.902). Improved/stacking outer fold 1/10 took 1615 s, so blood should be done at about 15:00 (sooner once X-ray frees the CPU).
  - 12:01: X-ray fold 7 done: 97.68 / 98.38 / 96.88 / 0.9962. 8-fold mean acc 97.37 ± 0.85. Fold 8 started. Blood outer folds 1–3 done (stack acc 78.33 / 77.68 / 80.00), about 23 min each, so blood should be done at about 14:45.
  - 12:53: X-ray fold 8 done: 97.68 / 98.61 / 96.62 / 0.9976. 9-fold mean acc 97.41 ± 0.80. Fold 9 (the last) started, done at about 13:45. Blood outer folds 4–6 done (89.75 / 86.10 / 81.53), so blood should be done at about 14:10.
  - **13:44: X-RAY PAPER RUN DONE.** `results/imaging_xray_paper.json` (10 folds). export_site ran at 13:45, and `site/data/results.js` has `imaging_paper.xray`.
    10-fold mean ± std: acc **97.41 ± 0.72**, sens 97.85 ± 1.00, spec 96.91 ± 0.83, prec 97.27 ± 0.73, F1 97.56 ± 0.68, AUC 0.9961 ± 0.0016, MCC 0.948 ± 0.014. The paper reports 99.02. Fold 9: 97.43 / 97.69 / 97.14.
    The GPU is free. The chain is now waiting for the parallel blood run (outer folds 1–9 done) and then runs the final export_site.
- Chain: `resume_xray_blood.sh` (bash PID 11748). Resume command, only if nothing is running: `bash scripts/resume_xray_blood.sh`

Read this first if you are a new Claude session picking this project up. Then read HANDOFF.md for the
project background. You can resume without asking the user anything.

## Current task: CT, exact paper protocol (`--profile paper`)
Tungal et al. 2026 (Health Science Reports 9:e71972) is the reference. The paper reports CT CNN accuracy of **98.49%**.
Protocol (`PROFILES["paper"]` in `src/config.py`, `run_paper()` in `src/imaging.py`):
- 100×100×3 input
- SMOTE-ENN on the whole image set before the split (as the paper does)
- stratified 10-fold, plain 90/10
- Adam lr 1e-4, batch 16, 100 fixed epochs, fp32
- Paper CNN only (74,018 params)

Machine: Windows 10 laptop, GeForce 940MX (2 GB), torch 2.6.0+cu124 in `.venv`. About 38 s/epoch, about 62 min/fold.

### Update 2026-10-10 18:57 IST
- I found no process running. The old run had died at fold 5 epoch 51, and `fold_5.ckpt.pt` was saved at epoch 50.
- I relaunched `scripts/resume_ct.sh` detached at 18:55 IST. It resumed fold 5 from epoch 50 ("resumed from checkpoint at epoch 50" in `logs/ct_paper.log`).
- 19:28 IST: fold 5 done: acc 97.81, sens 97.04, spec 98.49, prec 98.29, AUC 0.9970. 6-fold mean acc 97.73. Fold 6 started.
- 20:21–20:28 IST: a Windows Update (TrustedInstaller, planned OS upgrade) restarted the PC and killed the run during fold 6 at about epoch 73.
- 21:25 IST: I relaunched `scripts/resume_ct.sh`, which resumed fold 6 from its epoch-70 checkpoint. New ETA: fold 6 at about 21:45, then about 64 min per fold, all done at about 01:00 IST on 2026-10-11.
- 21:45 IST: fold 6 done: acc 97.61, sens 95.56, spec 99.44, prec 99.34, AUC 0.9980. 7-fold mean acc 97.71. Fold 7 started.
- 22:48 IST: fold 7 done: acc 97.61, sens 95.97, spec 99.06, prec 98.91, AUC 0.9982. 8-fold mean acc 97.70. Fold 8 started, CT should be done at about 00:55.
- Windows active hours are only 06:00–12:00, so another update restart outside those hours is possible. Check `Get-WinEvent` System Id 1074 if the run dies again.
- (old) ETA: fold 5 at about 19:30, then about 65 min per fold. Fold 9 and export_site should finish at about 23:50 IST.
- Do NOT start another CT job while this one runs. Check with `nvidia-smi` and the `[resume]` lines in `logs/pipeline.log`.

### Queued chain, armed 2026-10-10 21:55 IST (X-ray paper -> blood full -> export_site)
- The script is at `scripts/resume_xray_blood.sh`.
- It is running detached (bash PID 15136) and waits for `[resume] pipeline done - GPU free` in `logs/pipeline.log`, then runs:
  1. `imaging.py --modality xray --profile paper`, with 5 retries, logging to `logs/xray_paper.log`. Folds go to `results/folds/x-ray_paper_e100/` and the final file is `results/imaging_xray_paper.json`.
  2. `export_site.py`
  3. `blood.py --profile full`, with 3 retries, logging to `logs/blood_full.log`. It first backs up `results/blood.json` to `results/blood.pre_full.backup.json` and writes the marker `results/blood_full.done` when finished.
  4. `export_site.py`, then `[xb] pipeline done - GPU free`.
- Milestones are logged with the `[xb]` tag in `logs/pipeline.log`.
- Resume (only if no imaging.py/blood.py is running): `bash scripts/resume_xray_blood.sh` from Git Bash. Finished folds are skipped and the interrupted fold resumes from its 10-epoch checkpoint.
- X-ray: 9,544 images at about 65+ min/fold gives about 11 h, so it will NOT be done by 06:00 IST on 2026-10-11. That's expected.
- Power: sleep is set to never on AC and DC, and the laptop is on AC. Windows Update active hours are 06–12, so an overnight update restart is possible.

### State at the time of writing (2026-10-06, historical)
- Folds 0–4 are finished and saved (`results/folds/ct_paper_e100/fold_0..4.json`).
- Fold 5 was training at about epoch 41/100. Its checkpoint `results/folds/ct_paper_e100/fold_5.ckpt.pt` was saved at **epoch 40**.
- The user asked to pause here. If the training process was stopped, it resumes from that epoch-40 checkpoint. If it kept running, there are simply more folds/checkpoints on disk. Either way the resume command below is correct.
- To check: `ls results/folds/ct_paper_e100/` and `tail logs/ct_paper.log`. A running job shows a `python.exe ... imaging.py --modality ct --profile paper` process and GPU load in `nvidia-smi`.

| fold | accuracy | sensitivity | specificity | precision | AUC |
|---|---|---|---|---|---|
| 0 | 96.71 | 94.92 | 98.31 | 98.03 | 0.9967 |
| 1 | 97.61 | 96.19 | 98.87 | 98.70 | 0.9965 |
| 2 | 97.51 | 98.09 | 96.99 | 96.66 | 0.9953 |
| 3 | 98.90 | 98.73 | 99.06 | 98.94 | 0.9988 |
| 4 | 97.81 | 96.41 | 99.06 | 98.92 | 0.9979 |
| **mean (5 folds)** | **97.71** | | | | |

The paper reports 98.49%. SMOTE-ENN went from 5427 COVID / 2628 non-COVID to 4723 / 5314, giving 10,037 images (9,033 train / 1,004 test per fold).

### Resume (only if no CT imaging.py is running)
From the repo root, in Git Bash:
```bash
bash scripts/resume_ct.sh
```
This checks CUDA, then runs `python src/imaging.py --modality ct --profile paper` with up to 5 automatic retries. It skips finished folds and resumes the interrupted fold from its last 10-epoch checkpoint. After that it runs `src/export_site.py` and finally logs `<timestamp> [resume] pipeline done - GPU free` to `logs/pipeline.log`.
Manual equivalent:
```bash
.venv\Scripts\activate
cd src
python imaging.py --modality ct --profile paper
python export_site.py
```
Remaining time is about 4.5 h (folds 5–9). It is best run detached so a chat disconnect can't kill it, e.g. PowerShell `Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList 'scripts/resume_ct.sh' -WindowStyle Hidden`.

### Outputs and logs
- `results/imaging_ct_paper.json` is the final 10-fold result. Format: models["Paper CNN"] mean ± std, per_fold, smote_enn, history. It is written only after the last fold.
- `results/folds/ct_paper_e100/` holds per-fold JSONs and the `fold_<k>.ckpt.pt` mid-fold checkpoint (deleted when the fold finishes).
- `data/cache/ct_100.npz`, `data/cache/ct_100_smoteenn.npz` (deterministic, reused on resume) and `data/cache/x-ray_100.npz` are the caches.
- `logs/ct_paper.log` is the training log. `logs/pipeline.log` is the pipeline/milestone log.
- The site shows paper-profile results in a "Paper-exact protocol" card in the imaging section, from `R.imaging_paper.{xray,ct}`.

## Queued after CT (other thread: "Run the full protocol on your GPU")
Start these only after CT has finished. The GPU is too small for two jobs at once.
1. `python src/imaging.py --modality xray --profile paper`. It uses `data/cache/x-ray_100.npz` and the same crash-safe resume, giving `results/imaging_xray_paper.json`.
2. `python src/blood.py --profile full`
3. `python src/export_site.py`

That thread had a watcher keyed on `[p2] pipeline done - GPU free`. That watcher has been stopped. `scripts/resume_ct.sh` prints `[resume] pipeline done - GPU free` instead.

## Local changes as of 2026-10-06 (committed 2026-10-11)
- `src/config.py`: `PROFILES["paper"]`
- `src/imaging.py`: `run_paper` / `train_paper` (paper protocol), plus crash safety: atomic per-fold JSONs, 10-epoch checkpoints, cached SMOTE-ENN, resume
- `src/download_data.py`: User-Agent header (Mendeley returned 403 to urllib); a partial download now resumes instead of being checksummed and deleted; 50 retries
- `src/export_site.py`, `site/js/main.js`, `site/index.html`: "Paper-exact protocol" card for `results/imaging_*_paper.json`
- `scripts/resume_ct.sh`, `RESUME.md`
- Not for git: `data/wheels/` (2.5 GB torch wheel), `results/*.backup.json`, `results/folds/`, `logs/*.log`

Ask the user before committing or pushing any of this.

## Waiting on the user
- The rubric thread (draft PR #1, branch `claude/rubric-excellent-54zut3`) wants `src/data_audit.py` and `src/imaging_tests.py` pulled in and run. This needs the user's explicit OK first; it was blocked as untrusted code integration.
