"""COVID-19 severity from routine blood tests (BIGDATA-COVID19, San Raffaele Hospital).

Three layers of evidence:
  1. paper protocol   : standardise -> SMOTE-ENN on the *whole* dataset -> stratified 10-fold CV (as published)
  2. honest protocol  : patient-grouped 10-fold CV, scaling + SMOTE-ENN fitted on the training folds only.
                        The dataset has several rows per patient, so row-wise CV after oversampling leaks.
  3. improvements     : engineered haematology ratios, clinical context (age, sex), Optuna-tuned ERT/LightGBM/XGBoost,
                        out-of-fold stacking with a logistic meta-learner, threshold selection, SHAP.

Usage:  python blood.py --profile quick
"""
import argparse
import json
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, confusion_matrix, matthews_corrcoef,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import BLOOD_CSV, PROFILES, RESULTS, SEED, SITE, device_name, ensure_dirs

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# the 15 parameters selected by RFE in the paper (Table 1); dataset columns carry a "0" (admission) suffix
PAPER15 = ["MCV", "NE", "RBC", "MPV", "MCH", "MOT", "BAT", "RDW", "HGB", "EOT", "WBC", "BA", "MCHC", "HCT", "LYT"]
CBC20 = ["MCV", "NE", "PLT", "RBC", "MPV", "MCH", "MOT", "BAT", "RDW", "NET", "EO", "HGB", "LY", "EOT", "WBC", "BA", "MCHC", "HCT", "LYT", "MO"]
ENG = ["NLR", "PLR", "logSII", "Mentzer"]
CLIN = ["Age", "Sex"]
NJ = 6  # keep some cores free for image training running alongside


# ----------------------------------------------------------------------------- data
def load():
    d = pd.read_csv(BLOOD_CSV)
    d = d.rename(columns={**{c + "0": c for c in CBC20 if c != "PLT"}, "PLT10": "PLT"})
    return d.dropna(subset=CBC20).reset_index(drop=True)


def features(d, kind):
    """kind: 'paper15' | 'eng' (15 + ratios) | 'clin' (15 + ratios + age/sex)"""
    f = d[PAPER15].copy()
    if kind in ("eng", "clin"):
        f["NLR"] = d["NE"] / (d["LY"] + 0.1)
        f["PLR"] = d["PLT"] / (d["LYT"] + 0.1)
        f["logSII"] = np.log1p(d["PLT"] * d["NET"] / (d["LYT"] + 0.1))
        f["Mentzer"] = d["MCV"] / d["RBC"]
    if kind == "clin":
        f["Age"], f["Sex"] = d["Age"], d["Sex"]
    return f


# ----------------------------------------------------------------------------- metrics
def ece(y, p, bins=10):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p > a) & (p <= b)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, spec, prec = tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1)
    return dict(accuracy=100 * (tp + tn) / len(y), balanced_accuracy=100 * (sens + spec) / 2, sensitivity=100 * sens,
                specificity=100 * spec, precision=100 * prec, fdr=100 * (1 - prec),
                f1=100 * 2 * prec * sens / max(prec + sens, 1e-9), mcc=float(matthews_corrcoef(y, pred)),
                auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)), ece=ece(y, p))


def summarise(rows):
    out = {}
    for k in rows[0]:
        v = [r[k] for r in rows]
        out[k], out[k + "_std"] = float(np.mean(v)), float(np.std(v))
    return out


def roc_pts(y, p, k=80):
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, k).astype(int))
    return dict(fpr=np.round(fpr[idx], 4).tolist(), tpr=np.round(tpr[idx], 4).tolist())


# ----------------------------------------------------------------------------- model wrappers
class Wrapped:
    """scaler + optional SMOTE-ENN fitted on training data only."""

    def __init__(self, model, scale=True, resample=None):
        self.model, self.scale, self.resample = model, scale, resample

    def fit(self, X, y, groups=None):
        X = np.asarray(X)
        self.sc = StandardScaler().fit(X) if self.scale else None
        Xs = self.sc.transform(X) if self.sc else X
        if self.resample == "smoteenn":
            Xs, y = SMOTEENN(random_state=SEED).fit_resample(Xs, y)
        self.m = clone(self.model).fit(Xs, y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        return self.m.predict_proba(self.sc.transform(X) if self.sc else X)[:, 1]


def paper_models():
    return {"ERT": ExtraTreesClassifier(random_state=SEED, n_jobs=NJ), "LGBM": LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=NJ),
            "RF": RandomForestClassifier(random_state=SEED, n_jobs=NJ), "kNN": KNeighborsClassifier(), "DT": DecisionTreeClassifier(random_state=SEED),
            "SVM": SVC(probability=True, random_state=SEED), "AB": AdaBoostClassifier(random_state=SEED)}


# ----------------------------------------------------------------------------- protocols
def paper_protocol(X, y, log):
    """Faithful to the paper: SMOTE-ENN before CV, row-wise stratified 10-fold."""
    Xs = StandardScaler().fit_transform(X)
    Xr, yr = SMOTEENN(random_state=SEED).fit_resample(Xs, y)
    counts = dict(before=[int((y == 0).sum()), int((y == 1).sum())], after=[int((yr == 0).sum()), int((yr == 1).sum())])
    log(f"paper protocol: SMOTE-ENN {counts['before']} -> {counts['after']}")
    res = {}
    for name, m in paper_models().items():
        rows, ms = [], []
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=SEED).split(Xr, yr):
            mm = clone(m).fit(Xr[tr], yr[tr])
            t = time.time()
            p = mm.predict_proba(Xr[te])[:, 1]
            ms.append((time.time() - t) * 1000 / len(te))
            rows.append(metrics(yr[te], p))
        res[name] = summarise(rows) | dict(ms=float(np.mean(ms)))
        log(f"  {name:5s} acc {res[name]['accuracy']:.2f}  auc {res[name]['auc']:.4f}")
    return res, counts


def grouped_cv(factory, X, y, groups, folds=10, thr=0.5):
    """Patient-grouped CV. Returns per-fold summary, pooled OOF probabilities, ms/sample."""
    oof, rows, ms = np.zeros(len(y)), [], []
    for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=SEED).split(X, y, groups):
        m = factory().fit(X.iloc[tr] if hasattr(X, "iloc") else X[tr], y[tr], groups[tr])
        t = time.time()
        p = m.predict_proba(X.iloc[te] if hasattr(X, "iloc") else X[te])
        ms.append((time.time() - t) * 1000 / len(te))
        thr_use = getattr(m, "thr", thr)
        oof[te] = p
        rows.append(metrics(y[te], p, thr_use))
    return summarise(rows) | dict(ms=float(np.mean(ms))), oof


# ----------------------------------------------------------------------------- improved models
def sw(y):
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def build(kind, params, y):
    if kind == "ERT":
        return ExtraTreesClassifier(**params, class_weight="balanced_subsample", random_state=SEED, n_jobs=NJ)
    if kind == "LGBM":
        return LGBMClassifier(**params, class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=NJ)
    return XGBClassifier(**params, scale_pos_weight=sw(y), tree_method="hist", random_state=SEED, n_jobs=NJ, eval_metric="logloss")


def suggest(kind, t):
    if kind == "ERT":
        return dict(n_estimators=t.suggest_int("n_estimators", 200, 600, step=100), max_features=t.suggest_float("max_features", 0.3, 1.0),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 6), criterion=t.suggest_categorical("criterion", ["gini", "entropy"]))
    if kind == "LGBM":
        return dict(n_estimators=t.suggest_int("n_estimators", 150, 600, step=50), learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    num_leaves=t.suggest_int("num_leaves", 8, 64), min_child_samples=t.suggest_int("min_child_samples", 5, 40),
                    subsample=t.suggest_float("subsample", 0.6, 1.0), subsample_freq=1, colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True))
    return dict(n_estimators=t.suggest_int("n_estimators", 150, 600, step=50), max_depth=t.suggest_int("max_depth", 3, 8),
                learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True), subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0), min_child_weight=t.suggest_int("min_child_weight", 1, 10),
                reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True))


def tune(kind, X, y, groups, trials):
    """Optuna TPE, objective = patient-grouped 3-fold ROC-AUC on the training portion only (nested)."""
    inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(X, y, groups))

    def obj(t):
        prm = suggest(kind, t)
        s = []
        for a, b in inner:
            m = build(kind, prm, y[a]).fit(X[a], y[a])
            s.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
        return float(np.mean(s))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=trials)
    return st.best_params


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Stack:
    """Out-of-fold stacking of tuned ERT / LightGBM / XGBoost with a logistic meta-learner and a tuned threshold."""

    def __init__(self, trials):
        self.trials = trials

    def fit(self, X, y, groups):
        X = np.asarray(X)
        self.prm = {k: tune(k, X, y, groups, self.trials) for k in ("ERT", "LGBM", "XGB")}
        oof = np.zeros((len(y), 3))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups):
            for j, k in enumerate(self.prm):
                oof[b, j] = build(k, self.prm[k], y[a]).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        self.meta = LogisticRegression(C=1.0, max_iter=1000).fit(logit(oof), y)
        po = self.meta.predict_proba(logit(oof))[:, 1]
        self.train_oof = po  # kept so callers can pick other operating points from training data only
        grid = np.linspace(0.2, 0.8, 61)
        self.thr = float(grid[np.argmax([balanced_accuracy_score(y, po >= g) for g in grid])])
        self.base = {k: build(k, self.prm[k], y).fit(X, y) for k in self.prm}
        return self

    def base_proba(self, X):
        return np.column_stack([m.predict_proba(np.asarray(X))[:, 1] for m in self.base.values()])

    def predict_proba(self, X):
        return self.meta.predict_proba(logit(self.base_proba(X)))[:, 1]


def improved(X, y, groups, trials, folds, log):
    """One pass over the outer folds yields tuned ERT/LGBM/XGB and the stack, all on identical splits."""
    names = ["Tuned ERT", "Tuned LightGBM", "Tuned XGBoost", "Stacked ensemble"]
    oof = {n: np.zeros(len(y)) for n in names}
    rows = {n: [] for n in names}
    ms, thrs, params = [], [], []
    X = np.asarray(X)
    for i, (tr, te) in enumerate(StratifiedGroupKFold(folds, shuffle=True, random_state=SEED).split(X, y, groups)):
        t0 = time.time()
        st = Stack(trials).fit(X[tr], y[tr], groups[tr])
        t1 = time.time()
        pb = st.base_proba(X[te])
        ps = st.meta.predict_proba(logit(pb))[:, 1]
        ms.append((time.time() - t1) * 1000 / len(te))
        for j, n in enumerate(names[:3]):
            oof[n][te] = pb[:, j]
            rows[n].append(metrics(y[te], pb[:, j]))
        oof[names[3]][te] = ps
        rows[names[3]].append(metrics(y[te], ps, st.thr))
        thrs.append(st.thr)
        params.append(st.prm)
        log(f"  outer fold {i + 1}/{folds}  stack acc {rows[names[3]][-1]['accuracy']:.2f} auc {rows[names[3]][-1]['auc']:.4f}  ({time.time() - t0:.0f}s)")
    res = {n: summarise(rows[n]) | dict(ms=float(np.mean(ms))) for n in names}
    return res, oof, dict(thresholds=thrs, params=params)


# ----------------------------------------------------------------------------- analysis
def calibration_curve(y, p, bins=8):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    q[-1] += 1e-9
    out = []
    for a, b in zip(q[:-1], q[1:]):
        m = (p >= a) & (p < b)
        if m.sum():
            out.append([float(p[m].mean()), float(y[m].mean()), int(m.sum())])
    return out


def threshold_curve(y, p):
    g = np.linspace(0.05, 0.95, 46)
    return dict(thr=g.round(3).tolist(), sens=[float(100 * ((p >= t) & (y == 1)).sum() / (y == 1).sum()) for t in g],
                spec=[float(100 * ((p < t) & (y == 0)).sum() / (y == 0).sum()) for t in g])


def shap_importance(X, y, cols):
    import shap
    m = build("LGBM", dict(n_estimators=250, learning_rate=0.05, num_leaves=24, min_child_samples=15, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8), y).fit(np.asarray(X), y)
    sv = shap.TreeExplainer(m).shap_values(np.asarray(X))
    sv = sv[1] if isinstance(sv, list) else sv
    sv = sv[..., 1] if sv.ndim == 3 else sv
    ma = np.abs(sv).mean(0)
    from scipy.stats import spearmanr
    direction = [float(spearmanr(np.asarray(X)[:, j], sv[:, j])[0]) for j in range(len(cols))]
    order = np.argsort(-ma)
    return dict(features=[cols[i] for i in order], mean_abs=[float(ma[i]) for i in order], direction=[direction[i] for i in order])


def export_forest(d, X, y, cols, path):
    """Compact ExtraTrees exported as JSON so the website can run it in the browser.

    `examples` carry the raw CBC values (+age/sex); the page recomputes the engineered ratios itself."""
    m = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=3, max_depth=11, class_weight="balanced_subsample", random_state=SEED, n_jobs=NJ).fit(np.asarray(X), y)
    trees = []
    for e in m.estimators_:
        t = e.tree_
        trees.append(dict(f=t.feature.tolist(), t=[round(float(v), 3) for v in t.threshold], l=t.children_left.tolist(), r=t.children_right.tolist(),
                          v=[round(float(v[0][1] / v[0].sum()), 3) for v in t.value]))
    Xa = np.asarray(X)
    raw = CBC20 + CLIN
    sev, non = d[y == 1][raw], d[y == 0][raw]
    ex = [dict(label="Severe", values=sev.sample(3, random_state=1).round(2).to_dict("records")),
          dict(label="Non-severe", values=non.sample(3, random_state=1).round(2).to_dict("records"))]
    meta = dict(features=cols, raw=raw, raw_lo=d[raw].quantile(0.01).round(2).tolist(), raw_hi=d[raw].quantile(0.99).round(2).tolist(),
                raw_med=d[raw].median().round(2).tolist(), examples=ex)
    path.write_text(json.dumps(dict(meta=meta, trees=trees), separators=(",", ":")))


def rfe_check(d, y):
    X = StandardScaler().fit_transform(d[CBC20])
    r = RFE(RandomForestClassifier(200, random_state=SEED, n_jobs=NJ), n_features_to_select=15).fit(X, y)
    sel = [c for c, k in zip(CBC20, r.support_) if k]
    return dict(selected=sel, overlap_with_paper=len(set(sel) & set(PAPER15)), dropped=[c for c in CBC20 if c not in sel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="quick")
    ap.add_argument("--trials", type=int, default=None, help="Optuna trials per model (default: profile)")
    ap.add_argument("--outer", type=int, default=None, help="outer folds for the improved models (default: profile)")
    a = ap.parse_args()
    P = dict(PROFILES[a.profile])
    P["optuna_trials"] = a.trials or P["optuna_trials"]
    P["outer_folds"] = a.outer or P["outer_folds"]
    ensure_dirs()
    log = lambda s: print(s, flush=True)
    d = load()
    y, groups = d["Severity"].values.astype(int), d["patid"].values
    log(f"{len(d)} rows, {d['patid'].nunique()} patients, severe {int(y.sum())}, non-severe {int((y == 0).sum())}")
    R = dict(profile=a.profile, device=device_name(), n=int(len(d)), patients=int(d["patid"].nunique()), severe=int(y.sum()), non_severe=int((y == 0).sum()))

    Xp = features(d, "paper15")
    R["paper_protocol"], R["smoteenn_counts"] = paper_protocol(Xp.values, y, log)

    log("honest protocol (patient-grouped CV, resampling inside training folds)")
    R["honest_smoteenn"], R["honest_none"] = {}, {}
    oof_h = {}
    for name, m in paper_models().items():
        R["honest_smoteenn"][name], oof_h[name] = grouped_cv(lambda m=m: Wrapped(m, True, "smoteenn"), Xp, y, groups, 10)
        R["honest_none"][name], _ = grouped_cv(lambda m=m: Wrapped(m, True, None), Xp, y, groups, 10)
        log(f"  {name:5s} smote-enn acc {R['honest_smoteenn'][name]['accuracy']:.2f} auc {R['honest_smoteenn'][name]['auc']:.4f} | none acc {R['honest_none'][name]['accuracy']:.2f}")

    log("feature ablation (ERT, honest protocol)")
    R["ablation"] = []
    ert = lambda: Wrapped(ExtraTreesClassifier(300, class_weight="balanced_subsample", random_state=SEED, n_jobs=NJ), False, None)
    for label, kind in [("Paper 15 blood markers", "paper15"), ("+ haematology ratios (NLR, PLR, SII, Mentzer)", "eng"), ("+ clinical context (age, sex)", "clin")]:
        r, _ = grouped_cv(ert, features(d, kind), y, groups, 10)
        R["ablation"].append(dict(name=label, **r))
        log(f"  {label:50s} acc {r['accuracy']:.2f} bacc {r['balanced_accuracy']:.2f} auc {r['auc']:.4f}")
    for label, mk in [("ERT, class-weighted (no resampling)", lambda: Wrapped(ExtraTreesClassifier(300, class_weight="balanced_subsample", random_state=SEED, n_jobs=NJ), False, None)),
                      ("ERT, SMOTE-ENN inside folds", lambda: Wrapped(ExtraTreesClassifier(300, random_state=SEED, n_jobs=NJ), False, "smoteenn")),
                      ("ERT, no balancing", lambda: Wrapped(ExtraTreesClassifier(300, random_state=SEED, n_jobs=NJ), False, None))]:
        r, _ = grouped_cv(mk, Xp, y, groups, 10)
        R.setdefault("balancing", []).append(dict(name=label, **r))

    log(f"improved models: Optuna ({P['optuna_trials']} trials) + stacking on paper15+ratios+age/sex, {P['outer_folds']} patient-grouped folds")
    Xc = features(d, "clin")
    R["improved"], oof_i, info = improved(Xc, y, groups, P["optuna_trials"], P["outer_folds"], log)
    R["improved_info"] = dict(thresholds=info["thresholds"], best_params=info["params"][0], features=list(Xc.columns))

    R["rfe"] = rfe_check(d, y)
    best = "Stacked ensemble"
    R["roc"] = {**{k: roc_pts(y, v) for k, v in oof_h.items() if k in ("ERT", "LGBM", "RF")}, **{k: roc_pts(y, v) for k, v in oof_i.items()}}
    pb = oof_i[best]
    thr = float(np.mean(info["thresholds"]))
    R["best"] = dict(name=best, threshold=thr, pooled=metrics(y, pb, thr), calibration=calibration_curve(y, pb), thresholds=threshold_curve(y, pb))
    R["shap"] = shap_importance(Xc, y, list(Xc.columns))
    export_forest(d, Xc, y, list(Xc.columns), SITE / "data" / "forest.json")
    (RESULTS / "blood.json").write_text(json.dumps(R))
    log("saved results/blood.json")


if __name__ == "__main__":
    main()
