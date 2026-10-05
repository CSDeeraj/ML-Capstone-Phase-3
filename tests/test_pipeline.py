"""Fast checks of the pipeline's key claims:  python -m pytest -q   (CPU, under a minute; torch tests skip if absent)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blood import CBC20, PAPER15, Wrapped, features, load, metrics  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402


@pytest.fixture(scope="module")
def blood():
    d = load()
    return d, d["Severity"].values.astype(int), d["patid"].values


def test_blood_counts_match_paper(blood):
    d, y, g = blood
    assert len(d) == 4430 and int((y == 0).sum()) == 3342 and int(y.sum()) == 1088  # paper Section 2.4
    assert len(np.unique(g)) == 1136


def test_paper_rfe_markers_present(blood):
    d, _, _ = blood
    assert len(PAPER15) == 15 and set(PAPER15) <= set(CBC20) <= set(d.columns)


def test_grouped_folds_never_share_patients(blood):
    d, y, g = blood
    for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=42).split(d, y, g):
        assert not set(g[tr]) & set(g[te])


def test_resampling_only_touches_training_data(blood):
    d, y, _ = blood
    X = features(d, "paper15").values
    m = Wrapped(DecisionTreeClassifier(random_state=0), True, "smoteenn").fit(X[:3000], y[:3000])
    p = m.predict_proba(X[3000:])
    assert p.shape == (len(X) - 3000,)  # one prediction per real test row, none synthetic
    assert m.sc.n_samples_seen_ == 3000  # scaler fitted on training rows only


def test_engineered_features(blood):
    d, _, _ = blood
    f = features(d, "clin")
    assert list(f.columns[-6:]) == ["NLR", "PLR", "logSII", "Mentzer", "Age", "Sex"]
    np.testing.assert_allclose(f["NLR"], d["NE"] / (d["LY"] + 0.1))
    assert np.isfinite(f.values).all()


def test_metrics_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert metrics(y, np.array([.1, .2, .8, .9]))["accuracy"] == 100
    assert metrics(y, np.array([.9, .8, .2, .1]))["auc"] == 0


def test_paper_cnn_parameter_count():
    torch = pytest.importorskip("torch")
    from imaging import CovidNetPlus, PaperCNN, n_params
    assert n_params(PaperCNN()) == 74018  # paper Table 6: ~74,018
    assert n_params(CovidNetPlus()) < 4.2e6 / 10  # still >10x smaller than MobileNet
    out = PaperCNN()(torch.zeros(2, 3, 100, 100))
    assert out.shape == (2, 2)
