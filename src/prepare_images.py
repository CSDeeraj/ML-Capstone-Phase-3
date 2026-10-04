"""Resize every X-ray / CT image to 100x100 RGB (as in the paper) and cache as uint8 .npz.

Normalisation (/255) is applied at train time so the cache stays small (~290 MB for X-ray).
"""
import numpy as np
from joblib import Parallel, delayed
from PIL import Image

from config import CACHE, IMG_ROOT, ensure_dirs

SIZE = 100
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif"}


def load(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)


def build(modality):
    X, y, names = [], [], []
    for label, cls in enumerate(["Non-COVID", "COVID"]):  # COVID = positive class (1)
        files = sorted(p for p in (IMG_ROOT / modality / cls).iterdir() if p.suffix.lower() in EXT)
        arrs = Parallel(n_jobs=-1)(delayed(load)(p) for p in files)
        X += arrs
        y += [label] * len(files)
        names += [f"{modality}/{cls}/{p.name}" for p in files]
        print(f"{modality:6s} {cls:10s} {len(files)}")
    np.savez_compressed(CACHE / f"{modality.lower()}_{SIZE}.npz", X=np.stack(X), y=np.array(y), names=np.array(names))


if __name__ == "__main__":
    ensure_dirs()
    for m in ("X-ray", "CT"):
        build(m)
