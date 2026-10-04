"""One-shot setup for a fresh machine (Windows / Linux / macOS).

    git clone https://github.com/CSDeeraj/ML-Capstone-Phase-3.git
    cd ML-Capstone-Phase-3
    python setup_env.py            # creates .venv, installs the right PyTorch build + requirements
    python setup_env.py --check    # just report GPU / torch status

Needs Python 3.10+ (3.14 works too). Picks a CUDA build of PyTorch if an NVIDIA GPU is visible,
otherwise the CPU build (Apple Silicon uses the default build, which includes MPS).
"""
import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def has_nvidia():
    return shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def run(*cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def check():
    code = ("import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),"
            "'| mps',bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()));"
            "print('gpu:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')")
    subprocess.run([str(PY if PY.exists() else sys.executable), "-c", code])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force the CPU build of PyTorch")
    a = ap.parse_args()
    if a.check:
        return check()
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10+ required")
    if len(str(ROOT)) > 80 and platform.system() == "Windows":
        print("WARNING: long path - clone into something short like C:\\work to avoid Windows path-length errors.")
    if not PY.exists():
        run(sys.executable, "-m", "venv", VENV)
    run(PY, "-m", "pip", "install", "--upgrade", "pip")
    req = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip() and not l.startswith("#") and not l.startswith("torch")]
    run(PY, "-m", "pip", "install", *req)
    if a.cpu or (not has_nvidia() and platform.system() != "Darwin"):
        run(PY, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu")
    elif has_nvidia():
        print("NVIDIA GPU detected -> CUDA build of PyTorch")
        run(PY, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124")
    else:  # macOS: default wheels include MPS
        run(PY, "-m", "pip", "install", "torch", "torchvision")
    check()
    act = r".venv\Scripts\activate" if platform.system() == "Windows" else "source .venv/bin/activate"
    print(f"\nNext:\n  {act}\n  python src/run_all.py --profile {'full' if has_nvidia() and not a.cpu else 'quick'}\n"
          "(read HANDOFF.md first - it lists what is already done and what remains)")


if __name__ == "__main__":
    main()
