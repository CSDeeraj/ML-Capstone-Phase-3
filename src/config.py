"""Shared configuration: paths, device selection and run profiles.

The code is hardware-agnostic. It picks CUDA > Apple MPS > CPU automatically, and a
*profile* scales epochs / folds / tuning budget so the same commands work on a laptop
CPU ("quick") or on a GPU box ("full").
"""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CAPSTONE_DATA", ROOT / "data"))
CACHE = DATA / "cache"
RESULTS = ROOT / "results"
SITE = ROOT / "site"
SEED = 42

# image dataset layout after unzipping the Mendeley archive
IMG_ROOT = DATA / "images" / "COVID-19 Dataset"
BLOOD_CSV = DATA / "prognostic_data.csv"

PROFILES = {
    # quick: finishes on a CPU-only laptop. full: intended for a GPU, closer to the paper's protocol.
    # "full" keeps the paper's CNN hyper-parameters (lr 1e-4, batch 16, 100 epochs, 10-fold CV)
    "quick": dict(paper_epochs=50, paper_lr=1e-3, paper_batch=32, plus_epochs=30, plus_batch=64, folds=1,
                  optuna_trials=20, outer_folds=10, patience=12),
    "full": dict(paper_epochs=100, paper_lr=1e-4, paper_batch=16, plus_epochs=80, plus_batch=32, folds=10,
                 optuna_trials=80, outer_folds=10, patience=15),
}


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name():
    import torch
    d = get_device()
    if d.type == "cuda":
        return torch.cuda.get_device_name(0)
    return {"mps": "Apple Silicon (MPS)", "cpu": "CPU"}[d.type]


def ensure_dirs():
    for p in (CACHE, RESULTS, SITE / "data", SITE / "assets"):
        p.mkdir(parents=True, exist_ok=True)
