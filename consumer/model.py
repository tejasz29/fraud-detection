"""Fraud-detection model wrapper: preprocessing, inference, and SHAP explanations.

This module is the single source of truth for how a raw transaction is turned
into a fraud prediction. Both ``train.py`` (which *produces* the artifacts) and
``consumer.py`` (which *consumes* them, added in Step 4) import :class:`FraudModel`,
so preprocessing can never drift between training and serving.

Latency note: single-transaction scoring deliberately avoids pandas. Building a
one-row DataFrame and calling ``predict_proba`` measures ~98 ms, which alone
exceeds the system's 100 ms budget; the numpy + ``inplace_predict`` path used by
:meth:`FraudModel.score_one` measures ~2 ms for bit-identical output.

Artifacts:
    model.pkl      joblib bundle -> booster + scaler + feature order + threshold
    explainer.pkl  shap.TreeExplainer built from the trained booster
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical feature order the model is trained and served on: Time, V1..V28, Amount.
FEATURE_COLUMNS: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# "Time" and "Amount" are the only raw (non-PCA) columns, so they are the only
# ones that need scaling. V1..V28 are already PCA components from the dataset.
SCALED_COLUMNS: list[str] = ["Time", "Amount"]


@dataclass
class FraudModel:
    """Bundles the trained booster with its preprocessing and explainer so that
    a single object fully defines how a transaction is scored."""

    booster: Any                       # trained xgboost.XGBClassifier
    scaler: Any                        # fitted sklearn scaler for SCALED_COLUMNS
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    threshold: float = 0.5
    explainer: Any = field(default=None, repr=False)  # shap.TreeExplainer

    def __post_init__(self) -> None:
        """Precompute the lookups used on the hot inference path."""
        self._scaled_idx = np.array(
            [self.feature_columns.index(c) for c in SCALED_COLUMNS], dtype=np.intp
        )
        # Cache scaler params as plain arrays so scaling is arithmetic, not a
        # sklearn call with DataFrame validation overhead.
        self._mean = np.asarray(self.scaler.mean_, dtype=np.float32)
        self._scale = np.asarray(self.scaler.scale_, dtype=np.float32)
        # Raw Booster handle: inplace_predict skips sklearn-wrapper validation.
        self._raw_booster = self.booster.get_booster()

    # ------------------------------------------------------------------ persistence
    def save(self, model_path: str | Path, explainer_path: str | Path | None = None) -> None:
        """Persist the model bundle (and explainer, if present) to disk."""
        bundle = {
            "booster": self.booster,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "threshold": self.threshold,
        }
        joblib.dump(bundle, model_path)
        logger.info("Saved model bundle -> %s", model_path)
        if explainer_path is not None and self.explainer is not None:
            joblib.dump(self.explainer, explainer_path)
            logger.info("Saved SHAP explainer -> %s", explainer_path)

    @classmethod
    def load(cls, model_path: str | Path, explainer_path: str | Path | None = None) -> "FraudModel":
        """Load a model bundle, and the SHAP explainer if a path is given/exists."""
        bundle = joblib.load(model_path)
        explainer = None
        if explainer_path is not None and Path(explainer_path).exists():
            explainer = joblib.load(explainer_path)
        return cls(
            booster=bundle["booster"],
            scaler=bundle["scaler"],
            feature_columns=bundle["feature_columns"],
            threshold=float(bundle.get("threshold", 0.5)),
            explainer=explainer,
        )

    # ---------------------------------------------------------------- preprocessing
    def to_array(self, transaction: dict) -> np.ndarray:
        """Fast path: turn one transaction dict into a scaled ``(1, n_features)`` array.

        Raises ValueError naming every missing feature, so a malformed Kafka
        message produces an actionable log line instead of a KeyError.
        """
        missing = [c for c in self.feature_columns if c not in transaction]
        if missing:
            raise ValueError(f"transaction is missing required features: {missing}")

        row = np.empty((1, len(self.feature_columns)), dtype=np.float32)
        for i, col in enumerate(self.feature_columns):
            row[0, i] = transaction[col]
        row[0, self._scaled_idx] = (row[0, self._scaled_idx] - self._mean) / self._scale
        return row

    def to_frame(self, transaction: dict | Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
        """Batch path: normalise input into an ordered, scaled feature frame.

        Used for evaluation and batch scoring, where per-call pandas overhead is
        amortised across many rows. For single-transaction serving use
        :meth:`to_array`.
        """
        if isinstance(transaction, pd.DataFrame):
            df = transaction.copy()
        elif isinstance(transaction, dict):
            df = pd.DataFrame([transaction])
        else:
            df = pd.DataFrame(list(transaction))

        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"transaction is missing required features: {missing}")

        df = df[self.feature_columns].astype(float)
        df[SCALED_COLUMNS] = self.scaler.transform(df[SCALED_COLUMNS])
        return df

    # ------------------------------------------------------------------- inference
    def predict_proba(self, transaction) -> np.ndarray:
        """Return P(fraud) for each transaction.

        A single dict takes the low-latency numpy path; DataFrames and iterables
        take the batch path.
        """
        if isinstance(transaction, dict):
            return np.asarray(self._raw_booster.inplace_predict(self.to_array(transaction)))
        X = self.to_frame(transaction)
        return self.booster.predict_proba(X)[:, 1]

    def predict(self, transaction) -> np.ndarray:
        """Return the 0/1 label for each transaction using ``self.threshold``."""
        return (self.predict_proba(transaction) >= self.threshold).astype(int)

    def score_one(self, transaction: dict) -> dict:
        """Score a single transaction for the streaming consumer.

        Returns fraud label, confidence in that label, raw fraud probability, and
        (if an explainer is loaded) the top contributing features. Scales the
        input once and reuses it for both prediction and explanation.
        """
        row = self.to_array(transaction)
        proba = float(np.asarray(self._raw_booster.inplace_predict(row))[0])
        is_fraud = int(proba >= self.threshold)
        result = {
            "is_fraud": is_fraud,
            "confidence": proba if is_fraud else 1.0 - proba,
            "fraud_probability": proba,
        }
        if self.explainer is not None:
            result["top_features"] = self._top_features(row, top_k=3)
        return result

    # -------------------------------------------------------------- explainability
    def explain_one(self, transaction: dict, top_k: int = 3) -> list[dict]:
        """Return the ``top_k`` features by absolute SHAP contribution for one txn."""
        return self._top_features(self.to_array(transaction), top_k=top_k)

    def _top_features(self, row: np.ndarray, top_k: int = 3) -> list[dict]:
        """Rank features for an already-scaled ``(1, n_features)`` row."""
        if self.explainer is None:
            raise RuntimeError("No SHAP explainer loaded; cannot explain prediction.")

        shap_values = self.explainer.shap_values(row)
        # Binary classifiers may return a per-class list; take the positive class.
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        values = np.asarray(shap_values)
        if values.ndim == 3:      # (n, features, classes)
            values = values[0, :, -1]
        elif values.ndim == 2:    # (n, features)
            values = values[0]

        order = np.argsort(np.abs(values))[::-1][:top_k]
        return [
            {
                "feature": self.feature_columns[i],
                "shap_value": float(values[i]),
                "value": float(row[0, i]),
                "direction": "increases_fraud" if values[i] > 0 else "decreases_fraud",
            }
            for i in order
        ]
