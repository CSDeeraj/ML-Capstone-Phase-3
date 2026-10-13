# Test cases: results and what they mean

Model: the proposed **stacked ensemble** (Optuna‑tuned ERT + LightGBM + XGBoost, logistic meta‑learner) on the paper's 15 markers + NLR, PLR, log SII, Mentzer + age, sex. Every number is on patients the model never saw: patient‑grouped 10‑fold CV, 20 Optuna trials per model inside each training fold. Source: `results/test_cases.json`, produced by `python src/test_cases.py`. 95 % CIs come from 300 patient‑level bootstrap resamples.

## Blood severity (T1–T9)

| # | Test case | Result | What it shows |
|---|---|---|---|
| T1 | Unseen patients (4430 rows, 1136 patients) | **AUC 0.901** (95 % CI 0.878–0.921); balanced acc. 81.7 %; sensitivity 81.8 %, specificity 81.7 %; NPV 93.2 % | Honest headline. Compare with the paper's 98 % from a leaky split. |
| T2 | Later patients: trained on the 795 patients first seen up to 6 Apr 2020, tested on 341 patients first seen after | **AUC 0.855**; balanced acc. 78.8 %; sensitivity 73.2 %; NPV 94.7 % | About 0.05 AUC is lost on a later wave. That loss is what a hospital should expect, and the paper never measured it. |
| T3 | Subgroups | Sex code 1: 0.898 · sex code 0: 0.890 · age < 50: 0.879 · 50–64: 0.913 · 65–79: 0.892 · 80+: 0.846 (CI 0.58–0.92) | Consistent across sex and most ages; the over‑80 group is small and uncertain. |
| T3 | **First visit (admission) only** | **AUC 0.727** (CI 0.675–0.779); sensitivity 36.5 % at the default threshold | **Most important limitation.** At admission, the moment the smart lab is meant for, the blood picture is much less separable. Most of the overall performance comes from follow‑up visits (AUC 0.909), when severity has already shown in the blood. |
| T3 | Patients whose label changes | AUC 0.784 | Transitions are harder, as expected. |
| T4 | Operating points (thresholds set on training folds only) | Default (t ≈ 0.27): sens 81.8 / spec 81.7 · **Screening** (t ≈ 0.13): sens **91.2** / spec 67.0 / NPV 95.9 · **Confirmatory** (t ≈ 0.39): sens 72.7 / spec **89.3** / PPV 69.0 | The same model can serve as a rule‑out screen or as an ICU confirmation step. The paper reports only one threshold. |
| T5 | Lab measurement noise on every value | 0 %: 0.901 · 5 %: 0.880 · 10 %: 0.857 | Degrades gradually, without a cliff. |
| T6 | Missing‑aware model on all 4995 rows | Complete rows AUC 0.891 · **the 565 rows the paper drops: AUC 0.828** | Rows with missing differentials can still be scored usefully instead of being discarded. |
| T7 | Mortality (never used in training) | 84 % of visits from the 115 patients who died were flagged severe; AUC 0.841 for death across all rows, 0.569 within severe rows | The score tracks a hard outcome. It does not rank risk of death among patients already severe. |
| T8 | Negative control: labels shuffled | AUC 0.488 | The pipeline itself does not leak. |
| T9 | Leakage mechanism | Under the paper's row‑wise CV, a test row's nearest training row is **the same patient 38.3 %** of the time (1‑NN accuracy 86.3 %). Under patient‑grouped CV: 0 % (77.6 %) | This explains most of the gap between 98 % and 81 %. |

## Imaging (run on the GPU PC)
`python src/imaging_tests.py --modality xray|ct` evaluates the saved fold‑0 models (Paper CNN and CovidNet‑Plus) on the same held‑out fold‑0 test images, changed in controlled ways. Results appear on the website after `python src/export_site.py`.

| # | Test | What it asks | Why it matters |
|---|---|---|---|
| I‑T1 | 9 corruptions: noise (2 levels), blur, brightness ±25 %, contrast −40 %, rotation 10°, JPEG q20, low resolution, horizontal flip, random 30 px occlusion | Does accuracy hold when the image comes from a different machine, is compressed, or has an object in the way? | Real hospitals differ in scanners, exposure and storage (Hendrycks & Dietterich 2019) |
| I‑T2 | Robustness sweeps: noise σ 0→0.2, blur 0→3 px, rotation 0→30°, occlusion 0→50 px | Where does each model break? | A curve is stronger evidence than one corrupted point; it shows the safe operating range |
| I‑T3 | Lungs blacked out (centre 60 %) | Can the model still "diagnose" without the lungs? | A high score here means shortcut learning (text markers, borders, source hospital), as Maguolo & Nanni and DeGrave et al. found |
| I‑T4 | Border removed (centre only) | Does the model work from the lung field alone? | The complement of I‑T3 |
| I‑T5 | Never‑seen split | Accuracy on test images with an identical copy (after resizing) in training versus those with none | The data audit found hundreds of duplicate images; the never‑seen score is the honest one |
| I‑T6 | Confidence | Calibration (ECE) and errors made with >90 % confidence | Confident mistakes are the dangerous ones in screening |

### Results (fold‑0 test set: X‑ray 955 images, CT 806; run 2026‑10‑13 on the GeForce 940MX)

Accuracy %, Paper CNN / CovidNet‑Plus. "Majority" is the score of always guessing the larger class (X‑ray 57.6 %, CT 67.4 %).

| Case | X‑ray | CT |
|---|---|---|
| Clean | 87.9 / 91.2 | 90.6 / 95.9 |
| Gaussian noise σ = 0.03 | 87.6 / 86.9 | 90.8 / 89.0 |
| Gaussian noise σ = 0.08 | 86.9 / **46.2** | 90.8 / **48.4** |
| Blur 1.5 px | 82.6 / 75.6 | 86.0 / 87.7 |
| Brightness −25 % | 77.1 / 87.4 | **69.9** / 94.3 |
| Contrast −40 % | 77.3 / 87.9 | 78.5 / 77.9 |
| Rotation 10° | 85.0 / 85.3 | 88.0 / 86.1 |
| JPEG quality 20 | 87.1 / 90.5 | 90.6 / 88.7 |
| Low resolution | 86.1 / 89.3 | 87.8 / 93.3 |
| Horizontal flip | 82.6 / 91.2 | 83.5 / 95.9 |
| Random 30 px occlusion | **65.4** / 84.6 | 82.6 / 94.3 |
| Lungs blacked out | 57.6 / 58.2 (AUC 0.67 / 0.76) | 67.4 / **80.5** (AUC 0.43 / 0.89) |
| Border removed | 46.2 / 47.2 | 85.6 / 73.9 |
| Never‑seen images only | 88.2 / 91.4 (865 images) | 91.1 / 96.0 (799 images) |
| Confident errors (> 90 % sure) | 13 of 116 / 10 of 84 | 10 of 76 / 6 of 33 |
| Calibration error (ECE) | 0.035 / 0.045 | 0.038 / 0.060 |

Sweeps (accuracy at the strongest level): noise σ 0.2 → X‑ray 77.3 / 42.9, CT 78.4 / 32.6; blur 3 px → 68.2 / 46.7, 81.4 / 76.4; rotation 30° → 58.3 / 77.8, 83.3 / 78.8; occlusion 50 px → 56.9 / 73.6, 70.4 / 86.1.

What this shows:
* **No shortcut learning on X‑ray.** With the lungs hidden, both models fall to the majority‑class rate (57.6 %), and with only the lungs left they still need the full image. AUC stays a little above 0.5 (0.67–0.76), so a weak signal sits outside the central box.
* **CT: the Paper CNN reads the lungs; CovidNet‑Plus partly does not.** The Paper CNN falls to the majority rate without the centre (67.4 %, AUC 0.43) and keeps 85.6 % with the centre only. CovidNet‑Plus keeps 80.5 % (AUC 0.89) with the centre hidden. CT lung fields extend beyond the central 60 % box, so this is partly real lung tissue, but it is a shortcut risk to state openly.
* **Duplicates do not inflate the score.** Test images with an identical copy in training (90 X‑ray, 7 CT) actually score *lower* (X‑ray 84.4 / 88.9), because many duplicates carry both labels; the never‑seen score equals or beats the overall score.
* **The two models fail differently.** CovidNet‑Plus is more accurate and much more robust to occlusion, flips and rotation, but collapses under sensor noise (σ ≥ 0.08), probably because its training augmentation never added noise. The Paper CNN barely notices noise but is fragile to occlusion, brightness and rotation. This is a concrete argument for an ensemble and for noise augmentation as the next step.
* **Errors are mostly uncertain ones.** About 1 in 10 errors is made with > 90 % confidence, and wrong answers average 72–75 % confidence versus 92–95 % for correct ones, so a confidence threshold could route doubtful scans to a radiologist.

The clean accuracies here are for fold‑0 models from the quick profile (50 epochs). The paper‑protocol 10‑fold results (X‑ray 97.41 %, CT 97.79 %) use SMOTE‑ENN before splitting and are reported separately.

The tested models are the fold‑0 models saved by the default (quick/full) profile, not the paper‑protocol (SMOTE‑ENN) runs, whose scores are reported separately.

## How to say it in the viva
"On new patients our model reaches AUC 0.90, and 0.86 on a later group of patients. It stays stable across sex and age, degrades gradually with lab noise, and gives 0.5 on shuffled labels, so the pipeline does not leak. Its weak point is the first visit, where AUC falls to 0.73. That tells us severity from a routine blood count is easier to confirm than to predict at admission, and it is the clearest direction for future work."
