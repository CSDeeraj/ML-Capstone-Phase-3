"""Data-quality audit for both datasets -> results/data_audit.json (shown on the website and in docs/DATA.md).

Blood (BIGDATA-COVID19): missingness and whether it depends on the label, rows per patient, labels that change
within a patient, physiologically implausible values, internal consistency of the differential count, age range.
Imaging (Menoufia X-ray / CT, only if the images are on disk): file formats, colour modes, resolution spread,
exact duplicates (MD5) and near duplicates (perceptual hash on the 100x100 cache), including near duplicates that
cross the train/test split used by imaging.py.

Usage:  python src/data_audit.py            (blood always; imaging when data/images exists)
"""
import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd

from config import BLOOD_CSV, CACHE, IMG_ROOT, RESULTS, SEED, ensure_dirs

# adult reference intervals used only to flag implausible values (not to filter the data)
PLAUSIBLE = {"MCHC0": (28, 38), "MCH0": (15, 40), "MCV0": (55, 130), "HGB0": (4, 20), "HCT0": (12, 65)}
CBC = ["MCV0", "NE0", "PLT10", "RBC0", "MPV0", "MCH0", "MOT0", "BAT0", "RDW0", "NET0", "EO0", "HGB0", "LY0", "EOT0",
       "WBC0", "BA0", "MCHC0", "HCT0", "LYT0", "MO0"]
DIFF_PCT = ["NE0", "LY0", "MO0", "EO0", "BA0"]
DIFF_ABS = ["NET0", "LYT0", "MOT0", "EOT0", "BAT0"]


def blood_audit():
    d = pd.read_csv(BLOOD_CSV)
    n = len(d)
    miss_any = d[CBC].isna().any(axis=1)
    kept = d[~miss_any]
    per_pt = d.groupby("patid").size()
    seq = d.sort_values(["patid", "patdate"]).groupby("patid")["Severity"].apply(lambda s: "".join(map(str, s)))
    mixed = seq[seq.str.contains("0") & seq.str.contains("1")]
    worsen_only = int(mixed.apply(lambda s: s == "".join(sorted(s))).sum())
    pct_sum = kept[DIFF_PCT].sum(axis=1)
    abs_gap = (kept[DIFF_ABS].sum(axis=1) - kept["WBC0"]).abs()
    implausible = {c.rstrip("0"): int(((d[c] < lo) | (d[c] > hi)).sum()) for c, (lo, hi) in PLAUSIBLE.items()}
    out = dict(
        rows=n, columns=int(d.shape[1]), patients=int(d.patid.nunique()),
        dates=[str(d.patdate.min()), str(d.patdate.max())],
        exact_duplicate_rows=int(d.drop(columns=["id"]).duplicated().sum()),
        missing_by_column={c.rstrip("0").replace("PLT1", "PLT"): int(d[c].isna().sum()) for c in CBC},
        rows_with_missing=int(miss_any.sum()),
        missing_rate_by_class={"non_severe": float(miss_any[d.Severity == 0].mean()), "severe": float(miss_any[d.Severity == 1].mean())},
        missing_pattern="differential count (NE/LY/MO/EO/BA, % and absolute) missing together" if
        int(d[DIFF_PCT + DIFF_ABS].isna().all(axis=1).sum()) == int(d["NE0"].isna().sum()) else "mixed",
        kept_rows=int(len(kept)), kept_patients=int(kept.patid.nunique()),
        class_before=[int((d.Severity == 0).sum()), int((d.Severity == 1).sum())],
        class_after=[int((kept.Severity == 0).sum()), int((kept.Severity == 1).sum())],
        rows_per_patient=dict(mean=float(per_pt.mean()), median=float(per_pt.median()), max=int(per_pt.max()),
                              single_visit=int((per_pt == 1).sum())),
        label_changes=dict(patients=int(len(mixed)), only_worsen=worsen_only, improve_or_fluctuate=int(len(mixed) - worsen_only)),
        dead_rows=int(d.Dead.sum()), dead_all_severe=bool((d[d.Dead == 1].Severity == 1).all()),
        age=dict(min=int(d.Age.min()), median=float(d.Age.median()), max=int(d.Age.max()), under18_rows=int((d.Age < 18).sum())),
        sex_rows=dict(code_1=int((d.Sex == 1).sum()), code_0=int((d.Sex == 0).sum())),  # coding not documented by the source
        implausible=implausible,
        differential_pct_sum=dict(rows_not_100=int((pct_sum.round(1) != 100).sum()), min=float(pct_sum.min())),
        differential_abs_gap=dict(rows_gap_gt_0_5=int((abs_gap > 0.5).sum()), max=float(abs_gap.max())),
        outliers_extreme=dict(WBC_gt_50=int((d.WBC0 > 50).sum()), LYT_gt_20=int((d.LYT0 > 20).sum())),
    )
    return out


# ----------------------------------------------------------------------------- imaging
def ahash(img100):
    """64-bit difference hash of a 100x100 RGB uint8 image (robust to small intensity changes)."""
    from PIL import Image
    g = Image.fromarray(img100).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())


def imaging_audit(modality):
    from PIL import Image
    from sklearn.model_selection import StratifiedKFold
    root = IMG_ROOT / modality
    files = [p for cls in ("COVID", "Non-COVID") for p in sorted((root / cls).iterdir()) if p.is_file()]
    fmt, mode, sizes, md5 = Counter(), Counter(), [], {}
    for p in files:
        with Image.open(p) as im:
            fmt[(im.format or p.suffix).upper()] += 1
            mode[im.mode] += 1
            sizes.append(im.size)
        md5.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append(p.parent.name)
    dup_groups = [v for v in md5.values() if len(v) > 1]
    sz = np.array(sizes)
    out = dict(files=len(files), formats=dict(fmt), modes=dict(mode),
               width=dict(min=int(sz[:, 0].min()), median=float(np.median(sz[:, 0])), max=int(sz[:, 0].max())),
               height=dict(min=int(sz[:, 1].min()), median=float(np.median(sz[:, 1])), max=int(sz[:, 1].max())),
               aspect_not_square=int((np.abs(sz[:, 0] / sz[:, 1] - 1) > 0.05).sum()),
               exact_duplicates=dict(groups=len(dup_groups), extra_files=int(sum(len(v) - 1 for v in dup_groups)),
                                     cross_class=int(sum(len(set(v)) > 1 for v in dup_groups))))
    cache = CACHE / f"{modality.lower()}_100.npz"
    if cache.exists():
        z = np.load(cache)
        X, y = z["X"], z["y"]
        H = np.stack([ahash(x) for x in X])
        bits = np.unpackbits(H, axis=1).astype(np.uint8)
        tr, te = next(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y))  # fold 0 = the fold imaging.py reports
        near = 0
        for i in te:  # Hamming distance <= 2 of 64 bits ~ visually the same image
            dist = (bits[tr] != bits[i]).sum(1)
            near += int((dist <= 2).any())
        out["near_duplicate_test_in_train"] = dict(test_images=int(len(te)), with_near_duplicate_in_train=near,
                                                   share=float(near / len(te)))
    return out


def main():
    ensure_dirs()
    path = RESULTS / "data_audit.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    R = dict(blood=blood_audit(), imaging=prev.get("imaging", {}))
    for m in ("X-ray", "CT"):
        if (IMG_ROOT / m).exists():
            print(f"auditing {m} images ...", flush=True)
            R["imaging"][m] = imaging_audit(m)
    path.write_text(json.dumps(R, indent=1))
    b = R["blood"]
    print(f"blood: {b['rows']} rows / {b['patients']} patients; {b['rows_with_missing']} rows with missing CBC "
          f"(severe {100 * b['missing_rate_by_class']['severe']:.1f}% vs non-severe {100 * b['missing_rate_by_class']['non_severe']:.1f}%); "
          f"{b['label_changes']['patients']} patients change label over time")
    for m, v in R["imaging"].items():
        print(f"{m}: {v['files']} files, {v['exact_duplicates']['extra_files']} exact duplicates, "
              f"near-dup leak {v.get('near_duplicate_test_in_train', {}).get('with_near_duplicate_in_train', 'n/a')}")
    print(f"saved {path.relative_to(path.parents[1])}")


if __name__ == "__main__":
    main()
