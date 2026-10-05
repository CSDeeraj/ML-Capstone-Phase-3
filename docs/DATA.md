# Datasets and preprocessing

Every number on this page is produced by code in `src/` and stored in `results/` (`data_audit.json`, `blood.json`, `imaging_*.json`). Re‑run `python src/data_audit.py` to regenerate the audit.

## 1. Data sources (identical to the base paper)

| | Imaging | Blood |
|---|---|---|
| Name | Extensive COVID‑19 X‑ray and CT Chest Images Dataset, v3 | BIGDATA‑COVID19 prognostic dataset |
| Owner | Menoufia University (El‑Shafai & Abd El‑Samie), base paper ref [21] | IRCCS San Raffaele Hospital (OSR), Milan |
| Where | Mendeley Data, `8h65ywd2jr/3` | Zenodo record 4686707 (committed here as `data/prognostic_data.csv`, 0.7 MB) |
| Size | X‑ray 9544 (4044 COVID / 5500 non‑COVID); CT 8055 (5427 / 2628) | 4995 rows, 27 columns, 1218 patients, visits 2020‑02‑16 to 2020‑05‑13 |
| Integrity | zip SHA‑256 checked by `download_data.py`; class counts match the paper exactly | row and patient counts match the paper exactly (4995 / 1218) |
| Task | COVID vs non‑COVID | Severe vs non‑severe (per visit) |

Small differences from the paper's text: the paper gives the collection window as 19 Feb – 31 May 2020; the file's visit dates run 16 Feb – 13 May 2020. The paper says "22 features"; the file has 20 CBC markers + age + sex = 22 predictors, plus ID, date, severity and death columns.

## 2. Inconsistencies found and how each is handled

### Blood (from `results/data_audit.json`)

| # | Inconsistency | Measured | Handling | Why |
|---|---|---|---|---|
| B1 | **Repeated measures**: several rows per patient | 1218 patients, mean 4.1 rows, max 39; 501 patients have one row | All honest evaluation splits **by patient** (`StratifiedGroupKFold`) | Row‑wise splits put the same person in train and test; in our test T9 the nearest training row is the same patient 38 % of the time |
| B2 | **Missing values, not at random** | 565 rows (11.3 %) miss values; 419 miss the whole differential count (NE/LY/MO/EO/BA in % and absolute together). Missing in **19.9 % of severe vs 8.1 % of non‑severe rows** | Paper protocol: drop them (as published, leaves 4430 rows / 1136 patients / 3342 vs 1088). Extension: a missing‑aware gradient‑boosting model scores them instead (test T6) | Dropping keeps the paper comparable; but because missingness depends on the label, dropping removes the sickest visits, so we also show what happens if they are kept |
| B3 | **Label changes over time** | 68 patients have both severe and non‑severe visits (27 only worsen, 41 improve or fluctuate) | Severity treated as a per‑visit label; splits still by patient; admission‑visit results reported separately (T3) | A patient's later visit must not teach the model about their earlier one |
| B4 | **Class imbalance** | 3342 : 1088 (3.1 : 1) after cleaning | Paper: SMOTE‑ENN on all rows (reproduced: 2393 / 3103 vs paper 2378 / 3107). Ours: SMOTE‑ENN **inside training folds only**, or class weights | Resampling before splitting leaks synthetic copies of test patients into training |
| B5 | **Physiologically implausible values** | MCHC outside 28–38 g/dL: 8 rows (max 58.7); MCH > 40 pg: 3; MCV outside 55–130 fL: 2 | Kept, flagged | Too few to change results; tree models are insensitive to single extreme values; removing them would be an unreported filter |
| B6 | **Internal consistency of the differential** | % differential does not sum to 100 in 14 rows (min 85 %); absolute differential differs from WBC by > 0.5 ×10⁹/L in 63 rows | Kept, flagged | Rounding and analyser flags; small |
| B7 | **Extreme but real values** | WBC > 50: 6 rows; absolute lymphocytes > 20: 4 rows (likely haematological malignancy) | Kept; ratios use `+0.1` in denominators and log for SII | Real patients; robust features instead of deletion |
| B8 | **Scale differences** across markers (PLT in hundreds, BA near 0) | — | Standardisation fitted on training folds only | Needed by kNN/SVM; tree models are scale‑free |
| B9 | **Out‑of‑scope ages** | 9 rows under 18 (min age 1) | Kept, flagged | Paediatric reference ranges differ; too few to model separately |
| B10 | **Undocumented sex coding** | code 1: 3538 rows, code 0: 1457 | Used as given; reported as "sex code 1/0" | The source does not document which code is male |
| — | Exact duplicate rows | 0 | — | — |
| — | Death vs severity | all 155 death rows are labelled severe | Used as an outside check (T7) | Consistent labels |

### Imaging (run `python src/data_audit.py` on the PC with the images)

The audit checks formats, colour modes, resolution spread, exact duplicates (MD5, including the same file in both classes) and near duplicates (perceptual hash on the 100×100 cache) that cross the train/test split used by `imaging.py`. Results land in `results/data_audit.json → imaging` and on the website automatically. Known issues these checks target:

| # | Issue | Handling |
|---|---|---|
| I1 | Mixed resolutions and aspect ratios | Resize to 100×100 (paper); the audit reports how many are non‑square |
| I2 | Mixed grayscale and RGB files | Convert all to 3‑channel RGB (paper input shape 100×100×3) |
| I3 | Duplicate or near‑duplicate images (common in merged COVID collections, Roberts et al. 2021) | Measured; any near‑duplicates across the split are reported with the results |
| I4 | Class imbalance (X‑ray 1.36 : 1, CT 2.07 : 1) | Paper: SMOTE‑ENN on images (`imaging.py --balance paper` reproduces it; `--balance smoteenn` does it inside training only). Default: class‑weighted loss, no synthetic images (interpolated pixels are not valid scans and SMOTE before splitting leaks) |
| I5 | Shortcut cues (text, borders, source differences) | Lung‑masking and corruption tests in `src/imaging_tests.py` |

## 3. Preprocessing pipeline and justification

### Imaging
| Step | Setting | Justification |
|---|---|---|
| Resize | 100×100, bilinear (`prepare_images.py`) | Paper setting; keeps the 74k‑parameter CNN fast (~2–5 ms/image) |
| Colour | RGB | Paper's input shape (100, 100, 3); also lets grayscale and colour files share one model |
| Normalise | ÷255 at load time | Paper setting; the cache stays uint8 (~290 MB for X‑ray) |
| Split | Stratified 10‑fold; inside each fold a 10 % validation split for early stopping and calibration | Paper's 10‑fold; validation must not be the test fold |
| Augmentation (CovidNet‑Plus only) | flip, ±10° rotation, ±10 % scale, ±8 % shift, brightness/contrast jitter | Mimics positioning and exposure variation between radiographs; never applied to test images |

### Blood
| Step | Setting | Justification |
|---|---|---|
| Column mapping | `XXX0` → admission value; `PLT10` → PLT | dataset naming |
| Missing values | drop (paper) / keep with a missing‑aware model (extension) | see B2 |
| Scaling | `StandardScaler`, fitted on training folds only | paper uses standardisation; fitting inside folds avoids leakage |
| Balancing | SMOTE‑ENN inside training folds, or class weights | see B4 |

## 4. Feature extraction and selection, justified

### Images: learned features
The paper uses the CNN both as classifier and as **feature extractor** (its Figure 5): the 128‑d flattened output feeds HGB/ERT/GB/DT. We reproduce that exactly (`PaperCNN.embed`), and add CovidNet‑Plus's 96‑d global‑average‑pooled embedding for the HGB member of the calibrated ensemble. Learned features are justified because hand‑crafted texture features cannot match CNNs on this dataset (base paper Table 7), and Grad‑CAM can trace each prediction back to image regions.

### Blood: RFE, then clinically motivated additions
**Paper's 15 RFE markers** (Table 1): MCV, NE, RBC, MPV, MCH, MOT, BAT, RDW, HGB, EOT, WBC, BA, MCHC, HCT, LYT. Our own RFE run (random forest, 15 of 20) picks **10 of the same 15**; it swaps the absolute counts MOT, BAT, EOT, LYT and HGB for PLT, NET, EO, LY, MO. This shows the selection is stable in kind (red‑cell indices + white‑cell differential) but not in exact choice, because absolute and % counts carry nearly the same information.

**Engineered features** (`blood.features`), each with a clinical reason:

| Feature | Formula | Why |
|---|---|---|
| NLR | NE% / LY% | High NLR is an independent predictor of COVID severity and death (Liu et al. 2020; Qin et al. 2020) |
| PLR | PLT / LYT | Platelet‑lymphocyte ratio, an inflammation marker |
| log SII | log(1 + PLT × NET / LYT) | Systemic immune‑inflammation index predicts in‑hospital mortality (Fois et al. 2020); log tames its skew |
| Mentzer | MCV / RBC | Separates microcytosis types; tests whether red‑cell indices add beyond MCV |
| Age, Sex | as given | Strongest known clinical risk factors; available at admission at no cost |

**Did they help?** Ablation with the same ERT under honest patient‑grouped CV (`results/blood.json → ablation`):

| Feature set | Accuracy | Balanced acc. | AUC |
|---|---|---|---|
| Paper's 15 markers | 84.3 % | 72.1 % | 0.882 |
| + ratios | 84.5 % | 73.1 % | 0.886 |
| + age, sex | **86.2 %** | **75.4 %** | **0.902** |

**What the model actually uses** (mean |SHAP| on the LightGBM member): Age, **NLR**, **log SII**, MCV, MCHC, MCH, WBC, NE, EOT, RDW. Two of the top three are our engineered ratios, and NLR and SII push towards *severe* as the clinical literature predicts. Age pushes the other way in this cohort, a dataset effect worth stating in the viva: we infer it reflects how "severe" was assigned in spring‑2020 Milan (other work on this dataset predicts ICU admission), not that age is protective.
