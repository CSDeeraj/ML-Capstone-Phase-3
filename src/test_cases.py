"""Test cases for the proposed blood-severity model (the stacked ensemble) -> results/test_cases.json.

Each case asks a different question a clinician or examiner would ask, all on patients the model never saw:
  T1 unseen patients      : patient-grouped 10-fold, identical splits to blood.py
  T2 later patients       : temporal hold-out, trained on the earliest 70% of patients, tested on the rest
  T3 subgroups            : sex, age band, first visit vs follow-up, patients whose label changes (patient bootstrap CIs)
  T4 operating points     : default / screening (sens >= 90%) / confirmatory (spec >= 90%), thresholds from training folds only
  T5 measurement noise    : every lab value perturbed by 5% and 10% (analyser-to-analyser variation)
  T6 missing differential : a missing-aware model scores the 565 rows the paper discards
  T7 mortality            : does the severity score also rank the patients who died?
  T8 negative control     : shuffled labels must give AUC ~ 0.5 (the pipeline itself does not leak)
  T9 leakage mechanism    : how often a test row's nearest training neighbour is the same patient (row-wise vs grouped CV)

Usage:  python src/test_cases.py [--trials 20] [--folds 10]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from blood import CBC20, PAPER15, Stack, features, load
from config import BLOOD_CSV, PROFILES, RESULTS, SEED, ensure_dirs

RAW = CBC20 + ["Age", "Sex"]


def point(y, p, thr):
    """Threshold metrics where thr may be a scalar or a per-row array."""
    pred = p >= thr
    tp, tn = int((pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    fp, fn = int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    return dict(n=int(len(y)), positives=int(y.sum()), auc=auc, balanced_accuracy=100 * (sens + spec) / 2, sensitivity=100 * sens,
                specificity=100 * spec, ppv=100 * tp / max(tp + fp, 1), npv=100 * tn / max(tn + fn, 1), accuracy=100 * (tp + tn) / len(y))


def boot_ci(y, p, thr, groups, reps=300, seed=SEED):
    """95% CI of AUC and balanced accuracy, resampling patients (not rows)."""
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    thr = np.broadcast_to(thr, y.shape)
    aucs, baccs = [], []
    for _ in range(reps):
        ix = np.concatenate([idx_by[g] for g in rng.choice(ug, len(ug))])
        if 0 < y[ix].sum() < len(ix):
            r = point(y[ix], p[ix], thr[ix])
            aucs.append(r["auc"])
            baccs.append(r["balanced_accuracy"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(auc_ci=q(aucs), balanced_accuracy_ci=q(baccs))


def thr_for(p, y, target, kind):
    """Threshold from training data: highest thr keeping sensitivity >= target, or lowest keeping specificity >= target."""
    g = np.linspace(0.01, 0.99, 99)
    if kind == "sens":
        ok = [t for t in g if ((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) >= target]
        return float(max(ok)) if ok else 0.01
    ok = [t for t in g if ((p < t) & (y == 0)).sum() / max((y == 0).sum(), 1) >= target]
    return float(min(ok)) if ok else 0.99


def perturb(d, sigma, rng):
    d = d.copy()
    for c in CBC20:
        d[c] = np.clip(d[c] * (1 + rng.normal(0, sigma, len(d))), 0, None)
    return d


# ----------------------------------------------------------------------------- T1 + T3-T5 + T7 in one pass
def grouped_pass(d, y, groups, trials, folds, log):
    X = features(d, "clin").values
    n = len(y)
    out = {k: np.zeros(n) for k in ("p", "p5", "p10", "thr", "thr_screen", "thr_confirm")}
    rng = np.random.RandomState(SEED)
    ck = RESULTS / "test_cases_cache"  # per-fold checkpoints so an interrupted run resumes
    ck.mkdir(exist_ok=True)
    for i, (tr, te) in enumerate(StratifiedGroupKFold(folds, shuffle=True, random_state=SEED).split(X, y, groups)):
        f = ck / f"fold{i}_of{folds}_t{trials}.npz"
        if f.exists():
            z = np.load(f)
            for k in out:
                out[k][te] = z[k]
            log(f"  fold {i + 1}/{folds}  resumed from checkpoint")
            continue
        t0 = time.time()
        st = Stack(trials).fit(X[tr], y[tr], groups[tr])
        out["p"][te] = st.predict_proba(X[te])
        out["p5"][te] = st.predict_proba(features(perturb(d.iloc[te], 0.05, rng), "clin").values)
        out["p10"][te] = st.predict_proba(features(perturb(d.iloc[te], 0.10, rng), "clin").values)
        out["thr"][te] = st.thr
        out["thr_screen"][te] = thr_for(st.train_oof, y[tr], 0.90, "sens")
        out["thr_confirm"][te] = thr_for(st.train_oof, y[tr], 0.90, "spec")
        np.savez(f, **{k: v[te] for k, v in out.items()})
        log(f"  fold {i + 1}/{folds}  auc {roc_auc_score(y[te], out['p'][te]):.3f}  ({time.time() - t0:.0f}s)")
    return out


def subgroups(d, y, groups, o):
    first = d.sort_values("patdate").groupby("patid").cumcount().reindex(d.index).values == 0
    seq = d.groupby("patid")["Severity"].transform("nunique").values > 1
    defs = [("All rows", np.ones(len(y), bool)),
            ("Sex code 1", d.Sex.values == 1), ("Sex code 0", d.Sex.values == 0),
            ("Age < 50", d.Age.values < 50), ("Age 50-64", (d.Age.values >= 50) & (d.Age.values < 65)),
            ("Age 65-79", (d.Age.values >= 65) & (d.Age.values < 80)), ("Age 80+", d.Age.values >= 80),
            ("First visit (admission)", first), ("Follow-up visits", ~first),
            ("Patients whose label changes", seq)]
    rows = []
    for name, m in defs:
        r = point(y[m], o["p"][m], o["thr"][m])
        r |= boot_ci(y[m], o["p"][m], o["thr"][m], groups[m])
        rows.append(dict(name=name, patients=int(len(np.unique(groups[m]))), **r))
    return rows


# ----------------------------------------------------------------------------- T2 temporal
def temporal(d, y, groups, trials, log):
    first_seen = d.groupby("patid")["patdate"].min().sort_values()
    cut = int(0.7 * len(first_seen))
    train_pts = set(first_seen.index[:cut])
    tr = d.patid.isin(train_pts).values
    te = ~tr
    X = features(d, "clin").values
    st = Stack(trials).fit(X[tr], y[tr], groups[tr])
    p = st.predict_proba(X[te])
    r = point(y[te], p, st.thr) | boot_ci(y[te], p, st.thr, groups[te])
    r |= dict(train_patients=int(cut), test_patients=int(len(first_seen) - cut), train_rows=int(tr.sum()),
              cutoff=str(first_seen.iloc[cut]), train_until=str(first_seen.iloc[cut - 1]), test_until=str(d.patdate.max()))
    log(f"  temporal: train patients first seen <= {r['train_until']}, test from {r['cutoff']}  auc {r['auc']:.3f}")
    return r


# ----------------------------------------------------------------------------- T6 missing-aware model on the full file
def missing_aware(log):
    raw = pd.read_csv(BLOOD_CSV)
    raw = raw.rename(columns={**{c + "0": c for c in CBC20 if c != "PLT"}, "PLT10": "PLT"})
    raw = raw.dropna(subset=["Severity"]).reset_index(drop=True)
    X = features(raw, "clin").values  # NaN where the differential is missing; HistGradientBoosting routes NaN natively
    y, groups = raw.Severity.values.astype(int), raw.patid.values
    miss = np.isnan(X).any(axis=1)
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=SEED).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    r = dict(rows=int(len(y)), rows_missing=int(miss.sum()),
             complete_rows=point(y[~miss], p[~miss], 0.5), missing_rows=point(y[miss], p[miss], 0.5), all_rows=point(y, p, 0.5))
    log(f"  missing-aware: complete auc {r['complete_rows']['auc']:.3f}, rows with missing differential auc {r['missing_rows']['auc']:.3f}")
    return r


# ----------------------------------------------------------------------------- T8 + T9
def negative_control(d, y, groups):
    X = features(d, "clin").values
    ys = np.random.RandomState(SEED).permutation(y)
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=SEED).split(X, ys, groups):
        p[te] = ExtraTreesClassifier(300, class_weight="balanced_subsample", random_state=SEED, n_jobs=4).fit(X[tr], ys[tr]).predict_proba(X[te])[:, 1]
    return dict(auc=float(roc_auc_score(ys, p)))


def leakage_mechanism(d, y, groups):
    X = StandardScaler().fit_transform(d[PAPER15].values)
    out = {}
    for name, split in [("Row-wise CV (paper)", StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y)),
                        ("Patient-grouped CV (ours)", StratifiedGroupKFold(10, shuffle=True, random_state=SEED).split(X, y, groups))]:
        same, correct, total = 0, 0, 0
        for tr, te in split:
            nn_ = NearestNeighbors(n_neighbors=1).fit(X[tr])
            j = tr[nn_.kneighbors(X[te], return_distance=False)[:, 0]]
            same += int((groups[j] == groups[te]).sum())
            correct += int((y[j] == y[te]).sum())
            total += len(te)
        out[name] = dict(nearest_is_same_patient=100 * same / total, one_nn_accuracy=100 * correct / total)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="quick")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--folds", type=int, default=10)
    a = ap.parse_args()
    trials = a.trials or PROFILES[a.profile]["optuna_trials"]
    ensure_dirs()
    log = lambda s: print(s, flush=True)
    d = load()
    y, groups = d["Severity"].values.astype(int), d["patid"].values
    R = dict(profile=a.profile, trials=trials, folds=a.folds, model="Stacked ensemble (tuned ERT + LightGBM + XGBoost, logistic meta-learner)")

    log("T9 leakage mechanism / T8 negative control")
    R["leakage_mechanism"] = leakage_mechanism(d, y, groups)
    R["negative_control"] = negative_control(d, y, groups)
    log(f"  {R['leakage_mechanism']}  shuffled-label AUC {R['negative_control']['auc']:.3f}")

    log("T6 missing-aware model")
    R["missing_aware"] = missing_aware(log)

    log(f"T1/T3/T4/T5/T7 grouped pass ({a.folds} folds, {trials} Optuna trials per model)")
    o = grouped_pass(d, y, groups, trials, a.folds, log)
    R["unseen_patients"] = point(y, o["p"], o["thr"]) | boot_ci(y, o["p"], o["thr"], groups)
    R["subgroups"] = subgroups(d, y, groups, o)
    R["operating_points"] = [dict(name=k, threshold_mean=float(o[t].mean()), **point(y, o["p"], o[t]))
                             for k, t in [("Default (balanced accuracy)", "thr"), ("Screening (sens >= 90% on training)", "thr_screen"),
                                          ("Confirmatory (spec >= 90% on training)", "thr_confirm")]]
    R["noise"] = [dict(name=f"{s}% lab noise", sigma=s, **point(y, o[k], o["thr"])) for s, k in [(0, "p"), (5, "p5"), (10, "p10")]]
    dead = d["Dead"].values.astype(int)
    sev = y == 1
    R["mortality"] = dict(deaths=int(dead.sum()), auc_all_rows=float(roc_auc_score(dead, o["p"])),
                          auc_within_severe=float(roc_auc_score(dead[sev], o["p"][sev])),
                          flagged_severe=100 * float((o["p"][dead == 1] >= o["thr"][dead == 1]).mean()))

    log("T2 temporal hold-out")
    R["temporal"] = temporal(d, y, groups, trials, log)
    (RESULTS / "test_cases.json").write_text(json.dumps(R, indent=1))
    log("saved results/test_cases.json")


if __name__ == "__main__":
    main()
