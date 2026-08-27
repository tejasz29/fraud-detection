"""Tests for the fraud model wrapper (consumer/model.py).

Most tests build a tiny real XGBoost model inline so they exercise the actual
scoring / SHAP code paths without depending on the large trained artifacts. The
trained artifacts are also tested when present (skipped otherwise).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Make the repo root importable so `consumer.model` resolves under pytest.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consumer.model import FEATURE_COLUMNS, SCALED_COLUMNS, FraudModel  # noqa: E402


def _make_model(threshold: float = 0.5) -> FraudModel:
    """Train a small XGBoost model on synthetic data shaped like the real features."""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, len(FEATURE_COLUMNS))).astype(float)
    # Fraud signal lives in V2 and V6 so there is something to learn/explain.
    y = (X[:, FEATURE_COLUMNS.index("V2")] + X[:, FEATURE_COLUMNS.index("V6")] > 0).astype(int)

    # The serving path scales Time/Amount via the scaler, so train on scaled values.
    scaled_idx = [FEATURE_COLUMNS.index(c) for c in SCALED_COLUMNS]
    sub = X[:, scaled_idx]
    scaler = StandardScaler().fit(sub)
    X_scaled = X.copy()
    X_scaled[:, scaled_idx] = scaler.transform(sub)

    booster = XGBClassifier(n_estimators=20, max_depth=3, random_state=42)
    booster.fit(X_scaled, y)

    try:
        import shap

        explainer = shap.TreeExplainer(
            booster, feature_perturbation="tree_path_dependent", feature_names=list(FEATURE_COLUMNS)
        )
    except ImportError:
        explainer = None

    return FraudModel(
        booster=booster,
        scaler=scaler,
        feature_columns=list(FEATURE_COLUMNS),
        threshold=threshold,
        explainer=explainer,
    )


def _sample_txn(model: FraudModel, rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng(1)
    return {c: float(rng.normal()) for c in FEATURE_COLUMNS}


@pytest.fixture
def model() -> FraudModel:
    return _make_model()


def test_score_one_returns_required_keys(model: FraudModel) -> None:
    result = model.score_one(_sample_txn(model))
    assert set(result) >= {"is_fraud", "confidence", "fraud_probability"}
    assert result["is_fraud"] in (0, 1)
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_confidence_matches_label(model: FraudModel) -> None:
    for _ in range(25):
        result = model.score_one(_sample_txn(model))
        proba = result["fraud_probability"]
        if result["is_fraud"]:
            assert result["confidence"] == pytest.approx(proba)
        else:
            assert result["confidence"] == pytest.approx(1.0 - proba)


def test_threshold_flips_label(model: FraudModel) -> None:
    txn = _sample_txn(model)
    low = model.score_one(txn)
    model.threshold = 0.999  # almost nothing is fraud
    high = model.score_one(txn)
    model.threshold = 0.001  # almost everything is fraud
    very_high = model.score_one(txn)
    assert very_high["is_fraud"] >= low["is_fraud"]
    assert high["is_fraud"] <= very_high["is_fraud"]


def test_to_array_raises_on_missing_feature(model: FraudModel) -> None:
    txn = _sample_txn(model)
    del txn["V10"]
    with pytest.raises(ValueError, match="missing required features"):
        model.to_array(txn)


def test_to_array_scales_time_and_amount(model: FraudModel) -> None:
    txn = {c: 10.0 for c in FEATURE_COLUMNS}
    row = model.to_array(txn)
    scaled_idx = [FEATURE_COLUMNS.index(c) for c in SCALED_COLUMNS]
    # scaler was fit on standard normal data, so mean ~0, scale ~1 -> ~10.
    for i in scaled_idx:
        assert row[0, i] == pytest.approx(10.0, abs=1.0)


def test_explain_one_shape(model: FraudModel) -> None:
    if model.explainer is None:
        pytest.skip("SHAP not installed")
    top = model.explain_one(_sample_txn(model), top_k=3)
    assert len(top) == 3
    for item in top:
        assert set(item) >= {"feature", "shap_value", "value", "direction"}
        assert item["direction"] in ("increases_fraud", "decreases_fraud")


def test_score_one_runs_without_explainer() -> None:
    m = _make_model()
    m.explainer = None
    result = m.score_one(_sample_txn(m))
    assert "top_features" not in result


def test_real_artifacts_when_present() -> None:
    model_path = ROOT / "consumer" / "model.pkl"
    if not model_path.exists():
        pytest.skip("trained model.pkl not present (run consumer/train.py)")
    model = FraudModel.load(model_path, model_path.parent / "explainer.pkl")
    result = model.score_one(_sample_txn(model))
    assert set(result) >= {"is_fraud", "confidence", "fraud_probability", "top_features"}
    assert 0.0 < model.threshold < 1.0
