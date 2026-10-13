"""0-10 severity scale for the website's blood predictor.

The browser forest (site/data/forest.js) outputs the average of its 120 trees' leaf shares of severe visits, with both
classes weighted equally in training, so "39%" is not a 39% chance and is hard to read. This script turns that output
into a 0-10 score: 10 x the share of the 4,430 real visits that the same forest scores at or below it, so a typical
visit sits near 5 and the most severe-looking visits near 10.

Each score level is then described by what really happened to patients the model never saw: the same forest recipe is
refit in patient-grouped 10-fold CV, every visit is scored by the fold that left its patient out, and for each score we
count how many of those visits the hospital labelled severe and how many carry the dataset's death flag.

The demo presets are real visits picked by fixed rules (label, adult age, every value inside the slider range, browser
score and held-out score both near the target), plus the median patient and one honest miss.

Writes site/data/severity.js (window.SEVERITY) and results/severity_scale.json.
Usage:  python src/severity_scale.py      (export_site.py also runs it after exporting the forest)
"""
import json

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from blood import CBC20, CLIN, features, load
from config import RESULTS, SEED, SITE, ensure_dirs

BANDS = [("Not severe", 0, 1), ("Mild", 2, 4), ("Typical patient", 5, 6), ("Serious", 7, 8), ("Severe", 9, 10)]
QUANTILES = 1001  # reference distribution shipped to the browser, one cut every 0.1%


def forest_predict(F, X):
    """Evaluate the exported forest exactly as site/js/main.js does (same JSON thresholds, x <= t goes left)."""
    out, rows = np.zeros(len(X)), np.arange(len(X))
    for t in F["trees"]:
        f, th, l, r, v = (np.asarray(t[k]) for k in "ftlrv")
        n = np.zeros(len(X), int)
        while (l[n] != -1).any():
            go = l[n] != -1
            x = X[rows, np.where(go, f[n], 0)]
            n = np.where(go, np.where(x <= th[n], l[n], r[n]), n)
        out += v[n]
    return out / len(F["trees"])


def browser_features(F, v):
    """The engineered ratios as the page computes them from raw slider values."""
    calc = dict(NLR=lambda: v["NE"] / (v["LY"] + .1), PLR=lambda: v["PLT"] / (v["LYT"] + .1),
                logSII=lambda: np.log1p(v["PLT"] * v["NET"] / (v["LYT"] + .1)), Mentzer=lambda: v["MCV"] / v["RBC"])
    return [calc[n]() if n in calc else v[n] for n in F["meta"]["features"]]


def score(ref_sorted, p):
    """10 x share of reference visits scoring at or below p (mid-rank for ties)."""
    lo, hi = np.searchsorted(ref_sorted, p, "left"), np.searchsorted(ref_sorted, p, "right")
    return 10 * (lo + hi) / 2 / len(ref_sorted)


def export_scale(forest_js=SITE / "data" / "forest.js", log=print):
    F = json.loads(forest_js.read_text().strip()[len("window.FOREST="):].rstrip(";"))
    d = load()
    y, dead, groups = d["Severity"].values.astype(int), d["Dead"].values.astype(int), d["patid"].values
    X = features(d, "clin")
    assert list(X.columns) == F["meta"]["features"], "forest.js was exported with a different feature set"
    Xa = X.values.astype(float)

    p = forest_predict(F, Xa)
    ref = np.sort(p)
    s = score(ref, p)

    oof = np.zeros(len(d))
    for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=SEED).split(Xa, y, groups):
        m = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=3, max_depth=11, class_weight="balanced_subsample",
                                 random_state=SEED, n_jobs=-1).fit(Xa[tr], y[tr])
        oof[te] = m.predict_proba(Xa[te])[:, 1]
    so = score(np.sort(oof), oof)
    lvl = np.clip(np.round(so), 0, 10).astype(int)
    levels = [dict(score=k, n=int((lvl == k).sum()), severe=round(100 * y[lvl == k].mean(), 1), died=round(100 * dead[lvl == k].mean(), 1))
              for k in range(11)]
    bands = []
    for name, a, b in BANDS:
        mk = (lvl >= a) & (lvl <= b)
        bands.append(dict(name=name, lo=a, hi=b, n=int(mk.sum()), severe=round(100 * y[mk].mean(), 1), died=round(100 * dead[mk].mean(), 1)))
    sev_low = int(((so < 5.5) & (y == 1)).sum())

    raw = CBC20 + CLIN
    lo, hi = np.asarray(F["meta"]["raw_lo"]), np.asarray(F["meta"]["raw_hi"])
    inside = ((d[raw].values >= lo) & (d[raw].values <= hi)).all(1)
    adult = (d["Age"].values >= 35) & (d["Age"].values <= 75)
    rec = lambda i: {**{c: round(float(d[c].iloc[i]), 2) for c in CBC20}, "Age": int(d["Age"].iloc[i]), "Sex": int(d["Sex"].iloc[i])}

    def pick(target, label):
        ok = inside & adult & (y == label)
        cost = np.abs(s - target) + np.abs(so - target)
        return int(np.where(ok)[0][np.argmin(cost[ok])])

    def preset(kind, values, truth=None, i=None):
        pv = float(forest_predict(F, np.array([browser_features(F, values)]))[0])
        return dict(kind=kind, values=values, truth=truth, p=round(pv, 4), score=round(float(score(ref, pv)), 2),
                    heldout=None if i is None else round(float(so[i]), 2), age=values["Age"])

    med = dict(zip(raw, F["meta"]["raw_med"]))
    med["Sex"] = int(med["Sex"])
    presets = [preset("real", rec(i), "non-severe", i) for i in (pick(0, 0), pick(3, 0))]
    presets.append(preset("median", med))
    presets += [preset("real", rec(i), "severe", i) for i in (pick(8, 1), pick(10, 1))]
    # the visit the old "Severe 2" button showed: really severe, scored low by this forest and by the held-out one
    miss = F["meta"]["examples"][0]["values"][1]
    mi = np.where(np.isclose(d["MCV"], miss["MCV"]) & np.isclose(d["PLT"], miss["PLT"]) & (d["Age"] == miss["Age"]))[0]
    presets.append(preset("miss", {**miss, "Age": int(miss["Age"]), "Sex": int(miss["Sex"])}, "severe", int(mi[0]) if len(mi) else None))

    out = dict(ref_n=int(len(d)), cuts=[round(float(v), 5) for v in np.quantile(p, np.linspace(0, 1, QUANTILES))],
               levels=levels, bands=bands, presets=presets, heldout_auc=round(float(roc_auc_score(y, oof)), 4),
               severe_total=int(y.sum()), severe_scored_5_or_less=sev_low)
    ensure_dirs()
    (RESULTS / "severity_scale.json").write_text(json.dumps(out, indent=1))
    (SITE / "data" / "severity.js").write_text("window.SEVERITY=" + json.dumps(out, separators=(",", ":")) + ";")
    log(f"severity scale: held-out AUC {out['heldout_auc']}, {sev_low} of {out['severe_total']} severe visits score 5 or less held-out")
    for b in bands:
        log(f"  {b['lo']}-{b['hi']} {b['name']:16s} {b['n']:5d} visits  {b['severe']:5.1f}% severe  {b['died']:4.1f}% death flag")
    for q in presets:
        log(f"  preset {q['kind']:6s} {str(q['truth']):10s} score {q['score']:5.2f}  held-out {q['heldout']}  age {q['age']}")
    return out


if __name__ == "__main__":
    export_scale()
