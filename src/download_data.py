"""Download and unpack the two public datasets used by the paper.

  1. Extensive COVID-19 X-ray and CT Chest Images (Mendeley Data 8h65ywd2jr v3, ~4.0 GB)
  2. BIGDATA-COVID19 blood markers (Zenodo 4686707, prognostic_data.csv, ~0.7 MB; also committed in data/)

The big download is resumable: if it is interrupted, just run the script again.

Usage:  python src/download_data.py            # both
        python src/download_data.py --blood-only
Set CAPSTONE_DATA to choose where data lives (default: ./data, git-ignored except the blood CSV).
"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile

from config import BLOOD_CSV, DATA, IMG_ROOT

BLOOD_URL = "https://zenodo.org/api/records/4686707/files/prognostic_data.csv/content"
IMG_URL = "https://data.mendeley.com/public-files/datasets/8h65ywd2jr/files/fad8e33f-0d94-4316-b852-5f9f4814ea48/file_downloaded"
IMG_SHA256 = "35a65604be12bc092334191de5dc2848545e1a52e4aa37c21c1760a6d84cee3b"
IMG_BYTES = 3998246124


def fetch(url, dest, expected=None):
    """Streaming download with HTTP Range resume and a progress line."""
    have = dest.stat().st_size if dest.exists() else 0
    if expected and have == expected:
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # already complete
            return
        raise
    mode = "ab" if have and resp.status == 206 else "wb"
    if mode == "wb":
        have = 0
    total = (int(resp.headers.get("Content-Length", 0)) + have) or expected
    print(f"downloading {dest.name} ({(total or 0) / 1e9:.2f} GB){' - resuming at %.2f GB' % (have / 1e9) if have else ''}")
    with open(dest, mode) as f:
        done = have
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {100 * done / total:5.1f}%  {done / 1e9:.2f} GB", end="", flush=True)
    print()


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
    if a.blood_only:
        print("done (blood only)")
        return
    if IMG_ROOT.exists():
        print("images already present - done")
        return
    z = DATA / "covid_images.zip"
    for attempt in range(5):  # resume automatically after network hiccups
        try:
            fetch(IMG_URL, z, IMG_BYTES)
            break
        except Exception as e:  # noqa: BLE001
            print(f"\n  interrupted ({e}); retrying ({attempt + 1}/5)")
    else:
        sys.exit("download failed - run the script again to resume")
    print("verifying checksum ...")
    if sha256(z) != IMG_SHA256:
        z.unlink()
        sys.exit("checksum mismatch - file deleted, run again")
    print("extracting ...")
    with zipfile.ZipFile(z) as f:
        f.extractall(DATA / "images")
    print("done (you can delete data/covid_images.zip to save 4 GB)")


if __name__ == "__main__":
    main()
