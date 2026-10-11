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
`python src/imaging_tests.py --modality xray|ct` evaluates the saved fold‑0 models on 7 corruptions (noise, blur, brightness, contrast, rotation, JPEG, low resolution) and 2 shortcut tests (lungs blacked out; border removed). Results appear on the website's Test cases section after `python src/export_site.py`.

## How to say it in the viva
"On new patients our model reaches AUC 0.90, and 0.86 on a later group of patients. It stays stable across sex and age, degrades gradually with lab noise, and gives 0.5 on shuffled labels, so the pipeline does not leak. Its weak point is the first visit, where AUC falls to 0.73. That tells us severity from a routine blood count is easier to confirm than to predict at admission, and it is the clearest direction for future work."
