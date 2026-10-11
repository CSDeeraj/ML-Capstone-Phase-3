"""Power-cut-safe launcher for `imaging.py --profile paper`.

Runs the unchanged src/imaging.py paper protocol with three changes:
  * checkpoint every epoch instead of every 10 (the Paper CNN is ~74k params, about 1 MB per save)
  * atomic writes are fsync'd before the rename, so a power cut cannot leave a renamed-but-empty file
  * the previous checkpoint is kept as <ckpt>.bak; a corrupt checkpoint falls back to it (or restarts the fold)

    python scripts/paper_run.py --repo "G:/ML Capstone Phase 3" --modality xray
"""
import argparse
import os
import shutil
import sys
from pathlib import Path


def fsync_file(p):
    with open(p, "rb+") as fh:
        os.fsync(fh.fileno())


def loads(p, torch):
    try:
        torch.load(p, map_location="cpu", weights_only=False)["epoch"]
        return True
    except Exception:
        return False


def patch(imaging, every=1):
    torch = imaging.torch

    def atomic(path, save):
        path = Path(path)
        tmp = path.with_name(path.stem + ".tmp" + path.suffix)
        save(tmp)
        fsync_file(tmp)
        if path.name.endswith(".ckpt.pt") and path.exists():
            os.replace(path, path.with_name(path.name + ".bak"))  # keep the previous checkpoint
        os.replace(tmp, path)

    orig_train = imaging.train_paper

    def train_paper(model, Xtr, ytr, epochs, lr, batch, log, ckpt=None, every_=None):
        if ckpt is not None:
            bak = ckpt.with_name(ckpt.name + ".bak")
            if ckpt.exists() and not loads(ckpt, torch):
                log(f"  checkpoint {ckpt.name} is corrupt - removing it")
                ckpt.unlink()
            if not ckpt.exists() and bak.exists():
                if loads(bak, torch):
                    log(f"  restoring {ckpt.name} from {bak.name}")
                    shutil.copyfile(bak, ckpt)
                else:
                    bak.unlink()
        hist = orig_train(model, Xtr, ytr, epochs, lr, batch, log, ckpt=ckpt, every=every)
        if ckpt is not None:
            ckpt.with_name(ckpt.name + ".bak").unlink(missing_ok=True)
        return hist

    imaging.atomic = atomic
    imaging.train_paper = train_paper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--modality", choices=["xray", "ct"], required=True)
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    src = Path(a.repo) / "src"
    os.chdir(src)
    sys.path.insert(0, str(src))
    import imaging
    from config import PROFILES, ensure_dirs
    patch(imaging, a.every)
    ensure_dirs()
    mod = "x-ray" if a.modality == "xray" else "ct"
    imaging.run_paper(mod, PROFILES["paper"]["folds"], f"imaging_{a.modality}_paper")


if __name__ == "__main__":
    main()
