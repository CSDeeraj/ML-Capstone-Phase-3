"""Test cases for the imaging models -> results/imaging_tests_<xray|ct>.json (needs the image cache and saved models).

Uses the fold-0 test images that imaging.py held out, and the fold-0 models it saved in results/models/.
  clean            : the reference result
  corruptions      : Gaussian noise, blur, brightness, contrast, rotation, JPEG compression, low resolution
                     (the kind of variation between X-ray machines and hospitals; Hendrycks & Dietterich 2019)
  shortcut tests   : centre (lungs) blacked out, and border-only images. A model that still scores well without
                     the lungs is reading shortcuts, not disease (Maguolo & Nanni 2021; DeGrave et al. 2021)

Usage:  python src/imaging_tests.py --modality xray
"""
import argparse
import io
import json

import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: F401  (same split code as imaging.py)

from config import CACHE, RESULTS, SEED, ensure_dirs
from imaging import CovidNetPlus, PaperCNN, metrics, probs, to_tensor


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
    models = {}
    for name, cls, f in [("Paper CNN", PaperCNN, f"{mod}_paper_cnn.pt"), ("CovidNet-Plus", CovidNetPlus, f"{mod}_covidnet_plus.pt")]:
        p = RESULTS / "models" / f
        if p.exists():
            m = cls()
            m.load_state_dict(torch.load(p, map_location="cpu"))
            models[name] = m.eval()
    if not models:
        raise SystemExit(f"no saved models for {mod} in results/models; run imaging.py first")
    rng = np.random.RandomState(SEED)
    R = dict(modality=mod, test_images=int(len(te)), cases=[])
    for label, kind, Xc in cases(Xte, rng):
        row = dict(name=label, kind=kind)
        for name, m in models.items():
            r = metrics(yte, probs(m, to_tensor(Xc), tta=(name == "CovidNet-Plus")))
            row[name] = {k: r[k] for k in ("accuracy", "sensitivity", "specificity", "auc")}
        R["cases"].append(row)
        print(f"{label:34s} " + "  ".join(f"{n}: acc {row[n]['accuracy']:.1f} auc {row[n]['auc']:.3f}" for n in models), flush=True)
    out = RESULTS / f"imaging_tests_{a.modality}.json"
    out.write_text(json.dumps(R, indent=1))
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
