"""Run the whole pipeline end to end on whatever hardware you are on.

    python src/run_all.py --profile quick     # laptop / CPU
    python src/run_all.py --profile full      # GPU: paper protocol (100 epochs, 10-fold CV, lr 1e-4, batch 16)

Steps: download data (skipped if present) -> cache images -> imaging (X-ray, CT) -> blood -> export site data.
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent


def step(*args):
    print("\n$", "python", *args, flush=True)
    subprocess.run([sys.executable, str(SRC / args[0]), *args[1:]], check=True, cwd=SRC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick", choices=["quick", "full"])
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-images", action="store_true", help="skip the (slow) imaging models")
    a = ap.parse_args()
    if not a.skip_download:
        step("download_data.py")
    step("prepare_images.py")
    if not a.skip_images:
        for m in ("xray", "ct"):
            step("imaging.py", "--modality", m, "--profile", a.profile)
    step("blood.py", "--profile", a.profile)
    step("export_site.py")
    print("\nDone. Open site/index.html (or run: python -m http.server --directory site)")


if __name__ == "__main__":
    main()
