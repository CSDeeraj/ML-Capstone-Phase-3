"""Test cases for the imaging models -> results/imaging_tests_<xray|ct>.json (needs the image cache and saved models).

Uses the fold-0 test images that imaging.py held out, and the fold-0 models it saved in results/models/.
  clean            : the reference result
  corruptions      : Gaussian noise, blur, brightness, contrast, rotation, JPEG compression, low resolution
                     (the kind of variation between X-ray machines and hospitals; Hendrycks & Dietterich 2019)
  shortcut tests   : centre (lungs) blacked out, and border-only images. A model that still scores well without
                     the lungs is reading shortcuts, not disease (Maguolo & Nanni 2021; DeGrave et al. 2021)
  sweeps           : accuracy as noise, blur, rotation and occlusion get stronger (robustness curves)
  duplicates       : test images split into those with an identical copy in training and those without
                     (the score on images the model has never seen in any form)
  confidence       : calibration on clean images and how many errors were made with high confidence

Saves after every case, so an interrupted run keeps what it finished; reruns skip nothing (it is a few minutes).

Usage:  python src/imaging_tests.py --modality xray
"""
import argparse
import hashlib
import io
import json

import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: F401  (same split code as imaging.py)

from config import CACHE, RESULTS, SEED, ensure_dirs
from imaging import DEV, CovidNetPlus, PaperCNN, metrics, probs, to_tensor


def pil_map(X, fn):
    return np.stack([np.asarray(fn(Image.fromarray(x)).convert("RGB").resize((100, 100)), dtype=np.uint8) for x in X])


def jpeg(q):
    def f(im):
        b = io.BytesIO()
        im.save(b, "JPEG", quality=q)
        return Image.open(io.BytesIO(b.getvalue()))
    return f


def noise(X, s, rng):
    return np.clip(X / 255.0 + rng.normal(0, s, X.shape), 0, 1).__mul__(255).astype(np.uint8)


def scale(X, b=1.0, c=1.0):
    x = X / 255.0
    m = x.mean(axis=(1, 2, 3), keepdims=True)
    return np.clip(((x - m) * c + m) * b, 0, 1).__mul__(255).astype(np.uint8)


def mask(X, keep_center):
    """keep_center=False: black out the central 60% box (lungs). True: keep only it (border removed)."""
    X = X.copy()
    a, b = 20, 80
    if keep_center:
        out = np.zeros_like(X)
        out[:, a:b, a:b] = X[:, a:b, a:b]
        return out
    X[:, a:b, a:b] = 0
    return X


def occlude(X, k, rng):
    """black square of side k at a random place in each image (a cable, a label, a pacemaker)."""
    X = X.copy()
    for i in range(len(X)):
        r, c = rng.randint(0, 100 - k + 1, 2)
        X[i, r:r + k, c:c + k] = 0
    return X


def blur(X, r):
    return X if r == 0 else pil_map(X, lambda im: im.filter(ImageFilter.GaussianBlur(r)))


def rotate(X, a):
    return X if a == 0 else pil_map(X, lambda im: im.rotate(a, resample=Image.BILINEAR))


SWEEPS = {  # name: (unit, levels, function(X, level, rng))
    "Gaussian noise": ("σ", [0, 0.02, 0.05, 0.08, 0.12, 0.2], lambda X, v, rng: X if v == 0 else noise(X, v, rng)),
    "Blur": ("radius px", [0, 0.5, 1, 1.5, 2, 3], lambda X, v, rng: blur(X, v)),
    "Rotation": ("degrees", [0, 5, 10, 15, 20, 30], lambda X, v, rng: rotate(X, v)),
    "Occlusion": ("square side px", [0, 10, 20, 30, 40, 50], lambda X, v, rng: X if v == 0 else occlude(X, v, rng)),
}


def confidence(y, p):
    """errors made with >90 % confidence are the dangerous ones in a screening tool."""
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    wrong = pred != y
    return dict(mean_confidence_correct=float(conf[~wrong].mean()), mean_confidence_wrong=float(conf[wrong].mean()) if wrong.any() else None,
                errors=int(wrong.sum()), confident_errors=int((wrong & (conf > 0.9)).sum()))


def cases(X, rng):
    return [
        ("Clean", "reference", X),
        ("Gaussian noise σ=0.03", "corruption", noise(X, 0.03, rng)),
        ("Gaussian noise σ=0.08", "corruption", noise(X, 0.08, rng)),
        ("Blur (radius 1.5)", "corruption", pil_map(X, lambda im: im.filter(ImageFilter.GaussianBlur(1.5)))),
        ("Brightness −25 %", "corruption", scale(X, b=0.75)),
        ("Brightness +25 %", "corruption", scale(X, b=1.25)),
        ("Contrast −40 %", "corruption", scale(X, c=0.6)),
        ("Rotation 10°", "corruption", pil_map(X, lambda im: im.rotate(10, resample=Image.BILINEAR))),
        ("JPEG quality 20", "corruption", pil_map(X, jpeg(20))),
        ("Low resolution (50→100 px)", "corruption", pil_map(X, lambda im: im.resize((50, 50)).resize((100, 100), Image.BILINEAR))),
        ("Horizontal flip", "corruption", X[:, :, ::-1].copy()),
        ("Random 30 px occlusion", "corruption", occlude(X, 30, rng)),
        ("Lungs blacked out (centre 60 %)", "shortcut", mask(X, False)),
        ("Border removed (centre only)", "shortcut", mask(X, True)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["xray", "ct"], required=True)
    a = ap.parse_args()
    ensure_dirs()
    mod = "x-ray" if a.modality == "xray" else "ct"
    z = np.load(CACHE / f"{mod}_100.npz")
    X, y = z["X"], z["y"]
    _, te = next(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y))  # fold 0, as in imaging.run
    Xte, yte = X[te], y[te]
    tr = np.setdiff1d(np.arange(len(y)), te)
    models = {}
    for name, cls, f in [("Paper CNN", PaperCNN, f"{mod}_paper_cnn.pt"), ("CovidNet-Plus", CovidNetPlus, f"{mod}_covidnet_plus.pt")]:
        p = RESULTS / "models" / f
        if p.exists():
            m = cls()
            m.load_state_dict(torch.load(p, map_location="cpu"))
            models[name] = m.to(DEV).eval()  # probs() moves each batch to DEV, so the model must be there too
    if not models:
        raise SystemExit(f"no saved models for {mod} in results/models; run imaging.py first")
    pred = lambda name, Xc: probs(models[name], to_tensor(Xc), tta=(name == "CovidNet-Plus"))
    out = RESULTS / f"imaging_tests_{a.modality}.json"
    R = dict(modality=mod, test_images=int(len(te)), models=list(models), cases=[], sweeps={}, duplicates={}, confidence={})
    save = lambda: out.write_text(json.dumps(R, indent=1))

    rng = np.random.RandomState(SEED)
    for label, kind, Xc in cases(Xte, rng):
        row = dict(name=label, kind=kind)
        for name in models:
            r = metrics(yte, pred(name, Xc))
            row[name] = {k: r[k] for k in ("accuracy", "sensitivity", "specificity", "auc")}
        R["cases"].append(row)
        save()
        print(f"{label:34s} " + "  ".join(f"{n}: acc {row[n]['accuracy']:.1f} auc {row[n]['auc']:.3f}" for n in models), flush=True)

    for sname, (unit, levels, fn) in SWEEPS.items():
        rng = np.random.RandomState(SEED)
        R["sweeps"][sname] = dict(unit=unit, levels=levels, **{n: [] for n in models})
        for v in levels:
            Xc = fn(Xte, v, rng)
            for name in models:
                R["sweeps"][sname][name].append(round(float(metrics(yte, pred(name, Xc))["accuracy"]), 2))
        save()
        print(f"sweep {sname:16s} " + "  ".join(f"{n}: {R['sweeps'][sname][n]}" for n in models), flush=True)

    # identical copies after the 100x100 resize: is the test score inflated by images the model trained on?
    h = [hashlib.md5(x.tobytes()).hexdigest() for x in X]
    train_h = {h[i] for i in tr}
    seen = np.array([h[i] in train_h for i in te])
    R["duplicates"] = dict(test_with_copy_in_train=int(seen.sum()), test_without_copy=int((~seen).sum()))
    for name in models:
        p = pred(name, Xte)
        d = {}
        for key, m in (("all", np.ones_like(seen)), ("with_copy_in_train", seen), ("never_seen", ~seen)):
            if m.sum() and len(np.unique(yte[m])) == 2:
                r = metrics(yte[m], p[m])
                d[key] = {k: r[k] for k in ("accuracy", "sensitivity", "specificity", "auc")}
        R["duplicates"][name] = d
        R["confidence"][name] = dict(ece=metrics(yte, p)["ece"], **confidence(yte, p))
    save()
    for name in models:
        d = R["duplicates"][name]
        print(f"{name}: acc all {d['all']['accuracy']:.2f} | never-seen {d.get('never_seen', {}).get('accuracy', float('nan')):.2f} "
              f"({R['duplicates']['test_without_copy']} imgs) | confident errors {R['confidence'][name]['confident_errors']}", flush=True)
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
