# Problem identification and literature survey

**Base paper (the absolute reference for this project):** A. Tungal, P. Singh, K. Singh, P. D. Kaur, S. Bharany, R. Pant, A. Kumar, A. U. Rehman, S. Hussen, *"Artificial Intelligence Integrated Smart Medical Imaging Lab Framework for Enhanced Diagnosis and Treatment of Pandemic‑Prone Diseases"*, Health Science Reports 2026;9:e71972, doi:10.1002/hsr2.71972.
Reference numbers written as **[P‑n]** are the numbered references of that paper; **[Mn]** are the methodological references listed at the end of this file.

---

## 1. The problem

During a pandemic surge, an emergency department has to answer two questions for every suspected patient, quickly and with little specialist time:

1. **Is it COVID‑19?** RT‑PCR is the gold standard but is slow, has false negatives and ran short of kits [P‑4, P‑5, P‑18, P‑19]. Chest X‑ray and CT are already acquired for symptomatic patients and can confirm lung involvement within minutes [P‑6].
2. **How severe is it?** Routine complete blood count (CBC) markers shift with severity, notably lymphopenia and neutrophilia [P‑7, P‑8, M18, M19], so a severity score from blood can route a patient to ICU or the ward.

The base paper's answer is a **Smart Imaging Lab**: RT‑PCR → X‑ray/CT classified by one compact 16‑layer CNN → severity from 15 RFE‑selected CBC markers with Extra Trees, with Grad‑CAM for transparency. It reports **99.02 %** (X‑ray), **98.49 %** (CT) and **98.00 %** (blood severity) accuracy.

**Our problem statement.** Reproduce that framework exactly, then answer the question a hospital would ask before adopting it: *do those numbers hold for patients the model has never seen, and can the system be made more reliable where they do not?*

## 2. What the base paper does (summary of its method)

| Stage | Base paper (Sections 2.1–2.5, Tables 1–2) | Where in this repo |
|---|---|---|
| Imaging data | Menoufia "Extensive COVID‑19 X‑ray and CT Chest Images" [P‑21]: X‑ray 4044 COVID / 5500 non‑COVID, CT 5427 / 2628 | `src/download_data.py`, `src/prepare_images.py` |
| Image preprocessing | Resize 100×100, RGB, divide by 255 | `prepare_images.py`, `imaging.prep` |
| Image balancing | SMOTE‑ENN → X‑ray 4960/4186, CT 5300/4659 | default: class‑weighted loss; paper's method available with `--balance smoteenn-all` (see §5) |
| Image model | 16 layers: 8 Conv2D(32, 3×3) + 4 MaxPool + Dropout + Flatten + Dense + output; ~74k params; Adam, lr 1e‑4, batch 16, 100 epochs | `imaging.PaperCNN` (74,018 params, checked by `tests/`) |
| Hybrids | CNN features → HGB, ERT, GB, DT | `imaging.run` |
| Blood data | BIGDATA‑COVID19, San Raffaele, 4995 records / 1218 patients; 3342 non‑severe, 1088 severe after removing nulls | `data/prognostic_data.csv`, `blood.load` |
| Blood preprocessing | Drop nulls, standardise, RFE → 15 markers (Table 1), SMOTE‑ENN → 2378 / 3107 | `blood.paper_protocol` (gives 2393 / 3103, within 0.6 % of the paper) |
| Blood models | ERT, LGBM, RF, kNN, DT, SVM, AB; stratified 10‑fold CV | `blood.paper_models` |
| XAI | Grad‑CAM | `imaging.grad_cam` |
| Metrics | Accuracy, sensitivity, specificity, precision, FDR, ms/sample, ROC/AUC. Sensitivity = specificity in every row, so these are two‑class macro averages (= balanced accuracy; precision and FDR likewise macro) | all, plus MCC, balanced accuracy, Brier, ECE. The paper's sensitivity/specificity are compared with our **balanced accuracy**, and its precision with our macro precision |

## 3. Literature survey

### 3.1 COVID‑19 detection from chest images (the base paper's Table 7 plus context)

| Study | Data | Method | Reported | Limitation relevant to us |
|---|---|---|---|---|
| Ismail et al. 2021 [P‑30] | 4000 X‑rays (Menoufia) | InceptionV3 transfer learning | X‑ray 96.00 % | X‑ray only, single split |
| El‑Shafai et al. 2022 [P‑31] | 1000 X‑ray + 1000 CT | CNN (SGDM) | X‑ray 91.67 %, CT 100 % | very small subset |
| Mohbey et al. 2022 [P‑32] | 5000 CT | VGG, 224×224 | CT 95.00 % | CT only, heavy model |
| Ravi et al. 2022 [P‑22] | 9544 X‑ray, 8055 CT | EfficientNet features + PCA + stacking | 99 % / 99 % | heavy, multi‑stage |
| Hayat et al. 2023 [P‑33] | 9544 X‑ray, 8055 CT | SCovNet | 97.62 % | — |
| Constantinou et al. 2023 [P‑15] | chest X‑ray | deep CNNs | — | imaging only |
| Lanjewar et al. 2023, 2024 [P‑5, P‑6] | X‑ray / CT | small CNNs on cloud / smartphone | — | supports compact deployable CNNs |
| Wang et al. 2020, COVID‑Net [M7] | COVIDx | tailored CNN | — | early open benchmark |
| **Tungal et al. 2026 (base)** | 9544 X‑ray, 8055 CT | 16‑layer CNN, 74k params | 99.02 % / 98.49 % | see gaps below |

Critical appraisals of this whole body of work are important for us:
* **Roberts et al. 2021** [M3] reviewed 62 imaging COVID models and found none fit for clinical use, mainly because of duplicated or mixed‑source data, missing external validation and leakage between train and test.
* **Wynants et al. 2020** [M4] rated almost all COVID prediction models at high risk of bias.
* **DeGrave et al. 2021** [M5] and **Maguolo & Nanni 2021** [M6] showed X‑ray classifiers can score well while looking at shortcuts (text markers, borders, source hospital) instead of lung tissue; Maguolo & Nanni got high accuracy with the lungs blacked out. **Zech et al. 2018** [M8] showed the same for pneumonia models across hospitals.

### 3.2 COVID‑19 severity from blood (the base paper's Table 8 plus context)

| Study | Data | Method | Reported | Limitation |
|---|---|---|---|---|
| Famiglini et al. 2021 [P‑34] | BIGDATA‑COVID19 (same) | feature selection, SMOTE, SMBO‑tuned ensemble | AUC 0.88 (ICU admission) | — |
| Şiddeti et al. 2023 [P‑35] | BIGDATA‑COVID19 (same) | kNN imputation, min‑max, SMOTE, AdaBoost | Acc 89.54 % | SMOTE placement unclear |
| Zhang et al. 2022 [P‑17] | multivariate blood tests | several ML models | mild vs severe | single centre |
| Xiaoyan et al. 2025 [P‑16] | blood biomarkers | predictive model | severity | retrospective |
| Brinati et al. 2020 [M9]; Cabitza et al. 2021 [M10] | San Raffaele routine bloods | RF and others; external validation | COVID detection | show how much performance drops on external sites |
| **Tungal et al. 2026 (base)** | BIGDATA‑COVID19 | RFE‑15 + standardise + SMOTE‑ENN + ERT, stratified 10‑fold | **Acc 98.00 %** | see gaps below |

Note the jump: two earlier studies on the *same* dataset reached AUC 0.88 and 89.5 % accuracy, while the base paper reports 98 %. Explaining that jump is the core of our analysis.

Clinical evidence behind the features: lymphopenia and a high **neutrophil‑to‑lymphocyte ratio (NLR)** predict severity and death [M18, M19]; the **systemic immune‑inflammation index (SII)** predicts in‑hospital mortality [M20]; MCV is linked to COVID mortality [P‑8]; routine lab shifts in COVID are reviewed in [M22].

### 3.3 Methodology literature that frames our evaluation

* **Record‑wise vs subject‑wise CV.** When a dataset has several records per subject, record‑wise CV puts the same subject in train and test and overstates accuracy, sometimes massively (Saeb et al. 2017 [M1]). Kaufman et al. [M2] formalise this as leakage.
* **Resampling must be inside the training fold.** SMOTE [M13] and SMOTE‑ENN [M14] create synthetic points by interpolating neighbours. Applied to the full dataset before CV, synthetic copies of test patients end up in training.
* **Calibration** (Guo et al. 2017 [M12]) matters for triage, where the probability itself drives an ICU decision. The base paper reports no calibration.
* **Robustness to corruptions** (Hendrycks & Dietterich 2019 [M23]) is a standard way to test image models beyond a clean test set.
* **Reporting standards:** TRIPOD [M21] asks for internal *and* external or temporal validation and for handling of missing data to be described.

## 4. Research gaps (what is missing, in the literature and in the base paper)

| # | Gap | Evidence | How this project addresses it |
|---|---|---|---|
| **G1** | **Leaky evaluation of the blood model.** The base paper applies SMOTE‑ENN to all 4430 rows and then runs *row‑wise* stratified 10‑fold CV. The data has 1136 patients with up to 39 visits each, so the same patient (and synthetic neighbours of them) sits in both train and test. The same SMOTE‑ENN‑before‑CV order is described for the images. | Base paper §2.4, Table 8; [M1, M2]. Our test T9: under row‑wise CV the nearest training neighbour of a test row is **the same patient 38 % of the time**; under patient‑grouped CV, 0 %. | Paper protocol reproduced (ERT 97.6 % vs paper 98.0 %), then re‑run with patient‑grouped CV and resampling inside folds: **ERT 81.1 %, AUC 0.870**. This is the honest baseline. |
| **G2** | **No validation on later or unseen patients.** Only internal CV is reported; the paper itself lists real hospital validation as future work. | Base paper §4.3; [M3, M4, M10, M21] | Patient‑grouped CV (T1) and a **temporal hold‑out** trained on the earliest 70 % of patients and tested on later admissions (T2). |
| **G3** | **Missing data is discarded, and not at random.** 565 rows (11 %) are dropped. Missing differential counts are 2.5× more common in severe rows (19.9 % vs 8.1 %), so dropping them removes the sickest visits. | `results/data_audit.json` | Documented in `docs/DATA.md`; a missing‑aware model scores those rows instead of dropping them (T6). |
| **G4** | **Severity treated as a static patient label.** 68 patients change severity between visits; the paper never says whether a row or a patient is the unit. | Data audit | Treat severity as per‑visit, but always split by patient; report admission‑visit performance separately (T3). |
| **G5** | **Accuracy only, at one threshold, uncalibrated, and macro‑averaged class metrics.** On an imbalanced task (25 % severe) accuracy hides missed severe cases. The paper's sensitivity and specificity are identical in every row of Tables 3–5, which in a two‑class problem means both columns report the macro average of the two recalls, i.e. balanced accuracy; the per‑class sensitivity for COVID or severe cases is never shown. | Base paper Tables 3–5 | Add AUC, balanced accuracy, MCC, Brier, ECE; temperature scaling and calibration plots; screening vs confirmatory operating points (T4). |
| **G6** | **Grad‑CAM shown only on chosen successes.** Shortcut learning is a known failure in COVID imaging. | [M5, M6, M8] | Grad‑CAM++ beside Grad‑CAM, a deliberate failure case in the gallery, and a **lung‑masking shortcut test** plus corruption tests (`src/imaging_tests.py`). |
| **G7** | **No subgroup analysis.** Performance by age or sex is not reported. | Base paper | Subgroup table with patient‑bootstrap 95 % CIs (T3). |
| **G8** | **Compact model, but no improvement path.** The 74k CNN is efficient, yet modern low‑cost ingredients (residual connections, attention, augmentation, calibration) are not explored. | [M15, M16] | **CovidNet‑Plus** (≈275k params, still ~15× smaller than MobileNet): residual + squeeze‑excitation, augmentation, label smoothing, one‑cycle, TTA, calibrated ensemble. |
| **G9** | **Unified framework is described, not demonstrated end to end.** | Base paper Fig. 1 | Website triage simulator runs imaging + blood models on real held‑out cases in the browser. |

## 5. Deviations from the base paper, and why

We keep the paper's data, preprocessing, CNN and the seven blood models exactly. Two deliberate differences, both reported side by side with the paper protocol:

1. **Splitting by patient** for the blood data (G1). The paper protocol is still run and shown, so the 98 % figure is reproduced, not discarded.
2. **No SMOTE‑ENN on images.** Interpolating raw pixels of two X‑rays does not produce a valid X‑ray, and applying it before the split leaks test images into training. Class imbalance (≈1.4:1) is handled with class‑weighted loss inside training only. For a strictly paper‑identical run, `python src/imaging.py --modality xray --balance smoteenn-all` applies SMOTE‑ENN to the whole image set before CV exactly as published, and `--balance smoteenn` applies it to training splits only; both save to separate result files so all three can be compared.

## 6. Future research directions

1. **External validation** on a second hospital's CBC data and on a different X‑ray source (e.g. COVIDx [M7]), to measure the drop Cabitza et al. [M10] and Roberts et al. [M3] warn about.
2. **True multimodal fusion**: the paper links imaging and blood sequentially; a joint model needs paired image + blood data from the same patients, which no public dataset here provides.
3. **Longitudinal models** that use a patient's visit sequence (68 patients change severity) to predict deterioration before it happens.
4. **Principled missing‑data handling** (multiple imputation or missing‑aware models) as standard, given the label‑dependent missingness found here.
5. **Lung segmentation before classification** to remove shortcut cues, and quantitative XAI (pointing‑game or deletion metrics) instead of picked examples.
6. **Prospective, workflow‑level evaluation** of the smart lab: time to triage, ICU allocation accuracy and privacy/security when deployed with IoT/cloud, as the base paper itself proposes (§4.3).
7. **Other pandemic‑prone diseases**, reusing the same compact pipeline, as the base paper suggests.

---

## Methodological references

* [M1] S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, K. P. Kording. The need to approximate the use‑case in clinical machine learning. *GigaScience* 6(5), 2017.
* [M2] S. Kaufman, S. Rosset, C. Perlich, O. Stitelman. Leakage in data mining: formulation, detection, and avoidance. *ACM TKDD* 6(4), 2012.
* [M3] M. Roberts et al. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID‑19 using chest radiographs and CT scans. *Nature Machine Intelligence* 3, 199–217, 2021.
* [M4] L. Wynants et al. Prediction models for diagnosis and prognosis of covid‑19: systematic review and critical appraisal. *BMJ* 369:m1328, 2020.
* [M5] A. J. DeGrave, J. D. Janizek, S.‑I. Lee. AI for radiographic COVID‑19 detection selects shortcuts over signal. *Nature Machine Intelligence* 3, 610–619, 2021.
* [M6] G. Maguolo, L. Nanni. A critic evaluation of methods for COVID‑19 automatic detection from X‑ray images. *Information Fusion* 76, 1–7, 2021.
* [M7] L. Wang, Z. Q. Lin, A. Wong. COVID‑Net: a tailored deep convolutional neural network design for detection of COVID‑19 cases from chest X‑ray images. *Scientific Reports* 10, 19549, 2020.
* [M8] J. R. Zech et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross‑sectional study. *PLoS Medicine* 15(11), e1002683, 2018.
* [M9] D. Brinati, A. Campagner, D. Ferrari, M. Locatelli, G. Banfi, F. Cabitza. Detection of COVID‑19 infection from routine blood exams with machine learning: a feasibility study. *Journal of Medical Systems* 44, 135, 2020.
* [M10] F. Cabitza, A. Campagner et al. The importance of being external: methodological insights for the external validation of machine learning models in medicine. *Computer Methods and Programs in Biomedicine* 208, 106288, 2021.
* [M11] D. H. Wolpert. Stacked generalization. *Neural Networks* 5, 241–259, 1992.
* [M12] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On calibration of modern neural networks. *ICML* 2017.
* [M13] N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer. SMOTE: synthetic minority over‑sampling technique. *JAIR* 16, 321–357, 2002.
* [M14] G. E. A. P. A. Batista, R. C. Prati, M. C. Monard. A study of the behavior of several methods for balancing machine learning training data. *SIGKDD Explorations* 6(1), 20–29, 2004.
* [M15] K. He, X. Zhang, S. Ren, J. Sun. Deep residual learning for image recognition. *CVPR* 2016.
* [M16] J. Hu, L. Shen, G. Sun. Squeeze‑and‑excitation networks. *CVPR* 2018.
* [M17] R. R. Selvaraju et al. Grad‑CAM: visual explanations from deep networks via gradient‑based localization. *ICCV* 2017; A. Chattopadhay et al. Grad‑CAM++. *WACV* 2018.
* [M18] Y. Liu et al. Neutrophil‑to‑lymphocyte ratio as an independent risk factor for mortality in hospitalized patients with COVID‑19. *Journal of Infection* 81(1), e6–e12, 2020.
* [M19] C. Qin et al. Dysregulation of immune response in patients with COVID‑19 in Wuhan, China. *Clinical Infectious Diseases* 71(15), 762–768, 2020.
* [M20] A. G. Fois et al. The systemic inflammation index on admission predicts in‑hospital mortality in COVID‑19 patients. *Molecules* 25(23), 5725, 2020.
* [M21] G. S. Collins, J. B. Reitsma, D. G. Altman, K. G. M. Moons. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD). *BMJ* 350, g7594, 2015.
* [M22] G. Lippi, M. Plebani. Laboratory abnormalities in patients with COVID‑2019 infection. *Clinical Chemistry and Laboratory Medicine* 58(7), 1131–1134, 2020.
* [M23] D. Hendrycks, T. Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. *ICLR* 2019.
* [M24] W. C. Mentzer. Differentiation of iron deficiency from thalassaemia trait. *Lancet* 1(7808), 882, 1973.
* [M25] T. Akiba et al. Optuna: a next‑generation hyperparameter optimization framework. *KDD* 2019; S. M. Lundberg, S.‑I. Lee. A unified approach to interpreting model predictions. *NeurIPS* 2017.

Bibliographic details of [M‑] references were written from memory; check volume and page numbers against the publisher before final submission.
