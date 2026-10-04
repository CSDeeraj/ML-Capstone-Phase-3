"""Bundle results/*.json (+ the paper's published numbers) into site/data/*.js for the static website.

The site loads plain <script> files (not fetch) so it also works when index.html is opened from disk.
"""
import json

from blood import CBC20, PAPER15, export_forest, features, load
from config import RESULTS, SITE, ensure_dirs

# Published numbers: Tables 3-5 of Tungal et al., Health Science Reports 2026;9:e71972
M = lambda acc, sens, spec, prec, fdr, ms: dict(accuracy=acc, sensitivity=sens, specificity=spec, precision=prec, fdr=fdr, ms=ms)
PAPER = dict(
    xray={"CNN": M(99.02, 99.02, 99.02, 99.00, 1.00, 2.037), "HGB": M(95.74, 95.57, 95.57, 95.85, 4.15, 2.115),
          "ERT": M(94.86, 94.71, 94.71, 94.93, 5.07, 2.116), "GB": M(92.90, 92.56, 92.56, 93.19, 6.81, 2.098),
          "DT": M(87.32, 87.19, 87.19, 87.22, 12.78, 2.097)},
    ct={"CNN": M(98.49, 98.49, 98.49, 98.50, 1.50, 2.173), "HGB": M(95.38, 95.31, 95.31, 95.47, 4.53, 2.247),
        "ERT": M(95.08, 94.95, 94.95, 95.35, 4.65, 2.264), "GB": M(92.97, 92.82, 92.82, 93.29, 6.71, 2.257),
        "DT": M(89.56, 89.53, 89.53, 89.55, 10.45, 2.234)},
    blood={"ERT": M(98.00, 97.80, 97.80, 98.10, 1.90, 0.024), "LGBM": M(97.81, 97.82, 97.82, 97.71, 2.29, 0.025),
           "RF": M(97.45, 97.16, 97.16, 97.64, 2.36, 0.018), "kNN": M(96.36, 95.81, 95.81, 96.83, 3.17, 0.103),
           "DT": M(91.07, 90.60, 90.60, 91.07, 8.93, 0.002), "SVM": M(90.16, 89.81, 89.81, 90.00, 10.00, 0.102),
           "AB": M(87.98, 87.74, 87.74, 87.66, 12.34, 0.017)},
    params={"Proposed CNN (16-layer)": 74018, "MobileNet": 4.2e6, "DenseNet121": 8e6, "VGG-16": 138e6},
)


def read(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ensure_dirs()
    data = dict(paper=PAPER, imaging=dict(xray=read("imaging_xray.json"), ct=read("imaging_ct.json")), blood=read("blood.json"),
                gallery=dict(xray=read("gallery_x-ray.json"), ct=read("gallery_ct.json")))
    (SITE / "data" / "results.js").write_text("window.RESULTS=" + json.dumps(data, separators=(",", ":")) + ";")
    d = load()
    y = d["Severity"].values.astype(int)
    X = features(d, "clin")
    export_forest(d, X, y, list(X.columns), SITE / "data" / "forest.json")
    (SITE / "data" / "forest.js").write_text("window.FOREST=" + (SITE / "data" / "forest.json").read_text() + ";")
    print("site/data/results.js and forest.js written")


if __name__ == "__main__":
    main()
