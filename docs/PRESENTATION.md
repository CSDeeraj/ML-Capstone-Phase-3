# Presentation and viva guide

A 12‑minute talk that walks the website top to bottom, followed by a live demo and a question bank. Open the site with `python -m http.server --directory site` and go full‑screen (F11). Each part names the rubric row it earns marks for.

## Before the talk (checklist)
- [ ] `python src/export_site.py` run after the latest results; the imaging note above the KPIs says which profile produced the numbers.
- [ ] Site opened once with internet (Chart.js and fonts load from a CDN), then left open.
- [ ] `docs/` open in a second tab for detail questions; base paper PDF open in a third.
- [ ] One sentence ready for each research gap G1–G9 (below).

## Talk script (≈12 min)

| Time | Site section | Say this (in your own words) | Rubric row |
|---|---|---|---|
| 0:00 | Hero | "Our base paper is Tungal et al. 2026 in Health Science Reports. It proposes a Smart Imaging Lab: one small CNN reads chest X‑rays and CT scans for COVID‑19, and a blood test decides who goes to ICU. It reports 99.02, 98.49 and 98 percent accuracy. We rebuilt it exactly, then asked whether those numbers hold for patients the model has never seen." | Problem (40) |
| 0:45 | 01 The paper | Walk the six cards: unified lab, one CNN for both modalities (74,018 parameters, about 1,800× smaller than VGG‑16), RFE‑selected blood markers, Grad‑CAM, speed. Badges show what we reproduced and what we re‑examined. | Problem |
| 1:45 | 02 Literature & gaps | "On the same image dataset, earlier work used InceptionV3, VGG, EfficientNet. On the same blood dataset, earlier studies got AUC 0.88 and 89.5 percent. The base paper jumps to 98. Field‑wide reviews (Roberts 2021, Wynants 2020) say most COVID models fail because of leakage and missing external validation." Then name 3–4 of the nine gaps and point at the answer line on each card. End on future directions. | **Problem (40)** |
| 3:30 | 03 Framework | Let the patient journey animate once: emergency → RT‑PCR → imaging → blood → ICU or ward. | Problem |
| 4:00 | 04 Data | Sources, exact counts matching the paper, SHA‑256 checked. Then the audit table: "1,218 patients but 4,995 rows, so patients repeat. 565 rows are missing values, and missingness is 2.5× higher in severe rows, so dropping them removes the sickest visits. 68 patients change severity between visits." Then the two charts: missingness by class, and the feature ablation (ratios and age/sex raise AUC from 0.882 to 0.902). "Our RFE picks 10 of the paper's 15 markers; SHAP ranks NLR and SII, two engineered ratios, in the top three." | **Data (40)** |
| 6:00 | 05 Imaging | Architecture diagram (paper CNN vs CovidNet‑Plus), tabs X‑ray / CT, KPIs, ROC, confusion matrix, calibration. Say which profile produced the numbers. In the metrics table: "the paper's sensitivity equals its specificity in every row, so it is a macro average; we line it up against our balanced accuracy and also show true per‑class sensitivity." | Algorithm (40) |
| 7:00 | 06 Explainability | Drag the slider on one success, then click the failure case. "We show a mistake on purpose." | Algorithm |
| 7:30 | 07 Blood | The key finding: "Same models, same data. Paper's protocol: 97.6 percent, we reproduce it. Keep each patient on one side of the split: 81 percent." Switch to AUC. Then the upgrade ladder up to the stacked ensemble. | **Algorithm (40)** |
| 8:45 | 08 Test cases | "We then tested the final model nine ways." Read the four KPIs: unseen patients, later patients, 10 % lab noise, shuffled labels ≈ 0.5. Be upfront about the weak spot: "at the first visit AUC drops to 0.73, so this works better to confirm severity than to predict it at arrival." Point at T9 ("38 percent of test rows have the same patient as nearest neighbour under the paper's split; zero under ours"), the subgroup chart, mortality, and the operating‑point table (screening vs confirmatory). Imaging tab: corruption and lung‑masking shortcut test. | **Algorithm (40)** |
| 10:30 | 09–10 Live demo + simulator | See the demo plan below. | Presentation (30) |
| 11:30 | 11–12 Upgrades, limits | "Every upgrade has a practical, technical and theoretical reason. And here is what this does not show: public retrospective data, no patient IDs for images, not a medical device." | Presentation |
| 12:00 | Close | "We reproduced the paper, found where its evaluation is optimistic, fixed that, improved the models, and tested them case by case." | — |

## Live demo plan (≈1 min, rehearse twice)
1. In **09 Live demo**, click the *Severe* preset: gauge goes high, the "why" list shows NLR and age.
2. Drag **Lymphocytes %** up and **Neutrophils %** down: gauge falls. "That is the NLR effect clinicians already use."
3. In **10 Simulator**, click one COVID case: scan → CNN → blood forest → ICU or ward routing.
If the projector has no internet, the charts need Chart.js from the CDN; keep a screen recording as backup.

## One line per gap (memorise)
G1 leaky split → patient‑grouped CV. G2 no later patients → temporal hold‑out. G3 missing not at random → missing‑aware model. G4 severity changes → per‑visit labels, patient splits. G5 accuracy only → AUC, MCC, calibration, operating points. G6 Grad‑CAM on successes → Grad‑CAM++, failure case, shortcut test. G7 no subgroups → age/sex CIs. G8 no improvement path → CovidNet‑Plus. G9 lab only drawn → working simulator.

## Viva question bank

**Problem and literature**
1. *Why this paper?* It is recent (2026), uses public data, and claims a unified imaging + blood pipeline, so every claim can be checked.
2. *What exactly is new in your work over the paper?* Leakage‑free evaluation, a data audit, engineered clinical features, tuned stacking, calibration, nine test cases, CovidNet‑Plus, and a working demo.
3. *Is the paper wrong?* Its numbers are reproducible under its own protocol. The protocol lets the same patient appear in train and test, so they describe performance on known patients, not new ones.
4. *Which papers support the leakage argument?* Saeb et al. 2017 (record‑wise vs subject‑wise CV), Kaufman et al. 2012 (leakage), Roberts et al. 2021 (COVID imaging pitfalls).

**Data**
5. *Why drop rows with missing values?* To match the paper. We show the cost: missingness is label‑dependent, and a missing‑aware model can score those rows (T6).
6. *Why not SMOTE‑ENN on images like the paper?* Interpolating pixels of two X‑rays is not a valid X‑ray, and doing it before the split leaks. Class‑weighted loss gives the same balancing effect without synthetic data.
7. *Why 100×100?* The paper's setting; it keeps the CNN at 74k parameters and ~2–5 ms per image. A 128 px run is listed as an improvement.
8. *Why NLR and SII?* Both are established clinical markers of COVID severity and mortality (Liu 2020, Fois 2020); the ablation and SHAP confirm they help.
9. *Why does age push towards non‑severe in SHAP?* Inferred dataset effect: severity here tracks an ICU‑type decision in spring‑2020 Milan; it is not a claim that age protects.
10. *Can the images leak too?* The image set has no patient IDs, so patient splits are impossible; the audit measures near‑duplicates across the split and the limits section says image accuracy is an upper bound.

**Algorithm**
11. *Explain the paper CNN.* 8 Conv2D(32, 3×3) in pairs, 4 max‑pool, dropout, flatten (128), dense 64, output 2; ReLU, softmax, Adam lr 1e‑4, batch 16, 100 epochs: 74,018 parameters.
12. *What does squeeze‑excitation do?* Global‑average‑pools each channel, passes it through a tiny bottleneck MLP and rescales channels, so informative channels are amplified.
13. *Why stacking?* Base learners make partly different errors; a logistic meta‑learner trained on out‑of‑fold predictions combines them without in‑sample optimism.
14. *Why is tuning nested?* Optuna runs inside each outer training fold, so no test patient influences hyper‑parameters.
15. *What is temperature scaling?* One scalar divides the logits, fitted on validation data to minimise log‑loss; accuracy is unchanged, probabilities become trustworthy.
16. *How do you pick the threshold?* On training folds: default maximises balanced accuracy; screening keeps sensitivity ≥ 90 %; confirmatory keeps specificity ≥ 90 %.
17. *How do you know your pipeline doesn't leak too?* Shuffled‑label negative control gives AUC ≈ 0.5 (T8), and a unit test asserts no patient appears in two folds.
18. *Why does the paper's sensitivity equal its specificity in every row?* In a two‑class problem that only happens if both columns are the macro average of the two class recalls, which is balanced accuracy. So we compare the paper's "sensitivity" with our balanced accuracy, and we also report true per‑class sensitivity (COVID or severe cases caught) and specificity, which the paper does not.
19. *Where does your model fail?* At admission (first visit) AUC is 0.73 and default sensitivity only 37 %; it is much better on follow‑up visits (0.91). Severity from a blood count is easier to confirm than to predict at arrival (see `docs/TEST_CASES.md`).
20. *How would it do on a new wave?* Trained on patients up to 6 April 2020 and tested on later ones: AUC 0.855 (T2).
21. *Why are the imaging numbers lower than the paper's?* Say which profile (quick CPU vs full GPU). The full profile uses the paper's 100 epochs, 10 folds, lr 1e‑4, batch 16.

**Presentation habits**
- Lead with the number, then the reason. Always say "on unseen patients" when quoting honest results.
- If you do not know: "We did not test that; the closest evidence we have is …".
