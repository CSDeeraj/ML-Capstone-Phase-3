"""Launcher for src/blood.py that never loads torch.

blood.py only uses torch for `device_name()` (a label stored in results/blood.json). On this machine
(torch 2.6.0+cu124, scikit-learn 1.9.1, Windows) initialising torch's CUDA check in the same process
makes later scikit-learn native code (libsvm, kNN) crash at random with an access violation (exit 139).
All blood models run on the CPU, so the label is simply "CPU".

    python scripts/blood_run.py --repo "G:/ML Capstone Phase 3" --profile full
"""
import argparse
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--profile", default="full")
a, rest = ap.parse_known_args()
src = Path(a.repo) / "src"
os.chdir(src)
sys.path.insert(0, str(src))
import blood  # noqa: E402

blood.device_name = lambda: "CPU"
sys.argv = ["blood.py", "--profile", a.profile, *rest]
blood.main()
