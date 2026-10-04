"""Download and unpack the two public datasets used by the paper.

  1. Extensive COVID-19 X-ray and CT Chest Images (Mendeley Data 8h65ywd2jr v3, ~4.0 GB)
  2. BIGDATA-COVID19 blood markers (Zenodo 4686707, prognostic_data.csv, ~0.7 MB)

Usage:  python src/download_data.py            # both
        python src/download_data.py --blood-only
Set CAPSTONE_DATA to choose where data lives (default: ./data, git-ignored).
"""
import argparse
import hashlib
import urllib.request
import zipfile

from config import BLOOD_CSV, DATA, IMG_ROOT

BLOOD_URL = "https://zenodo.org/api/records/4686707/files/prognostic_data.csv/content"
IMG_URL = "https://data.mendeley.com/public-files/datasets/8h65ywd2jr/files/fad8e33f-0d94-4316-b852-5f9f4814ea48/file_downloaded"
IMG_SHA256 = "35a65604be12bc092334191de5dc2848545e1a52e4aa37c21c1760a6d84cee3b"


def fetch(url, dest):
    print(f"downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blood-only", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if not BLOOD_CSV.exists():
        fetch(BLOOD_URL, BLOOD_CSV)
    if a.blood_only or IMG_ROOT.exists():
        print("done (images already present)" if IMG_ROOT.exists() else "done (blood only)")
        return
    z = DATA / "covid_images.zip"
    if not z.exists():
        fetch(IMG_URL, z)
    if sha256(z) != IMG_SHA256:
        raise SystemExit("checksum mismatch - delete the zip and retry")
    print("extracting ...")
    with zipfile.ZipFile(z) as f:
        f.extractall(DATA / "images")
    print("done")


if __name__ == "__main__":
    main()
