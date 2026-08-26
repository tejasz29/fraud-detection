"""Train the XGBoost fraud-detection model.

Pipeline:
    1. Load creditcard.csv
    2. Stratified train/test split (test set left untouched and unbalanced so that
       reported metrics reflect real-world 0.17% fraud prevalence)
    3. Fit scaler on TRAIN ONLY, then apply SMOTE to TRAIN ONLY
    4. Train XGBoost, evaluate precision / recall / F1 / AUC-ROC / PR-AUC
    5. Tune the decision threshold for best F1 on a validation slice
    6. Save model.pkl and explainer.pkl

Usage:
    python consumer/train.py
    python consumer/train.py --data data/creditcard.csv --test-size 0.2 --no-smote
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Allow `python consumer/train.py` to import the sibling model module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FEATURE_COLUMNS, SCALED_COLUMNS, FraudModel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "creditcard.csv"
DEFAULT_MODEL_OUT = Path(__file__).resolve().parent / "model.pkl"
DEFAULT_EXPLAINER_OUT = Path(__file__).resolve().parent / "explainer.pkl"
DEFAULT_METRICS_OUT = Path(__file__).resolve().parent / "metrics.json"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the XGBoost fraud detection model.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to creditcard.csv")
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--explainer-out", type=Path, default=DEFAULT_EXPLAINER_OUT)
    p.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--no-smote", action="store_true", help="Skip SMOTE; rely on scale_pos_weight")
    return p.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error("Dataset not found at %s", path)
        logger.error("Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        logger.error("and place creditcard.csv in the data/ folder.")
        raise SystemExit(1)

    logger.info("Loading %s ...", path)
    df = pd.read_csv(path)

    required = set(FEATURE_COLUMNS) | {"Class"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset is missing expected columns: {sorted(missing)}")

    n_fraud = int(df["Class"].sum())
    logger.info(
        "Loaded %s rows | fraud=%s (%.4f%%) | normal=%s",
        f"{len(df):,}", n_fraud, n_fraud / len(df) * 100, f"{len(df) - n_fraud:,}",
    )
    return df


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float, label: str) -> dict:
    """Compute the imbalance-appropriate metric set at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }

    logger.info("--- %s (threshold=%.4f) ---", label, threshold)
    logger.info(
        "precision=%.4f  recall=%.4f  f1=%.4f  roc_auc=%.4f  pr_auc=%.4f",
        metrics["precision"], metrics["recall"], metrics["f1"],
        metrics["roc_auc"], metrics["pr_auc"],
    )
    logger.info("TP=%d  FP=%d  FN=%d  TN=%d", tp, fp, fn, tn)
    return metrics


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    default: float = 0.5,
    min_gain: float = 0.01,
) -> float:
    """Pick a probability cut-off, preferring ``default`` unless tuning clearly wins.

    At 0.17% prevalence the default 0.5 is often not optimal, so it is worth
    searching. But two failure modes make naive tuning worse than no tuning:

    1. ``precision_recall_curve`` only enumerates *observed* scores, so with
       well-separated classes the best-F1 threshold is exactly the lowest fraud
       score in validation — a razor-thin cut that any slightly-lower-scoring
       fraud in production slips under.
    2. F1 is usually flat across a band of thresholds, and ``argmax`` grabs an
       arbitrary edge of that band.

    So: take the *median* of the near-optimal plateau (fixing 2), and adopt the
    tuned value only if it beats ``default`` on validation by ``min_gain``
    relative F1 (fixing 1).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best_f1 = float(f1.max()) if f1.size else 0.0
    default_f1 = float(f1_score(y_true, (y_proba >= default).astype(int), zero_division=0))

    if best_f1 <= 0:
        logger.warning("No threshold produced a positive F1; keeping default %.2f", default)
        return default

    if best_f1 <= default_f1 * (1.0 + min_gain):
        logger.info(
            "Keeping default threshold %.2f (val F1=%.4f; best tunable F1=%.4f "
            "is not a >%.0f%% improvement)",
            default, default_f1, best_f1, min_gain * 100,
        )
        return default

    plateau = thresholds[f1 >= best_f1 * 0.99]
    threshold = float(np.median(plateau))

    # Max-margin placement. `thresholds` only contains *observed* scores, so the
    # value above sits exactly ON a training-set score with zero margin — the
    # nearest unseen transaction scoring a hair lower flips its label. Recentre
    # the cut midway into the score gap beneath it. On dense real-world scores
    # the neighbour is adjacent and this barely moves; on well-separated scores
    # it moves the cut off the cliff edge and into the middle of the gap.
    below = y_proba[y_proba < threshold]
    if below.size:
        centred = (threshold + float(below.max())) / 2.0
        if centred != threshold:
            logger.info("Max-margin recentre: %.4f -> %.4f", threshold, centred)
            threshold = centred

    logger.info(
        "Tuned threshold=%.4f (val F1 %.4f -> %.4f; plateau of %d thresholds in [%.4f, %.4f])",
        threshold, default_f1, best_f1, len(plateau), float(plateau.min()), float(plateau.max()),
    )
    return threshold


def main() -> None:
    args = parse_args()
    df = load_data(args.data)

    X = df[FEATURE_COLUMNS].astype(float)
    y = df["Class"].astype(int).to_numpy()

    # Hold out a test set that keeps the natural class imbalance.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
    )
    # Carve a validation slice off TRAIN for threshold tuning, so the test set
    # stays a genuine hold-out and the reported metrics are not optimistic.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE
    )
    logger.info(
        "Split -> train=%s (fraud=%d) | val=%s (fraud=%d) | test=%s (fraud=%d)",
        f"{len(X_train):,}", y_train.sum(), f"{len(X_val):,}", y_val.sum(),
        f"{len(X_test):,}", y_test.sum(),
    )

    # Fit the scaler on TRAIN ONLY — fitting on all data would leak test
    # distribution info into the model.
    scaler = StandardScaler().fit(X_train[SCALED_COLUMNS])
    for frame in (X_train, X_val, X_test):
        frame[SCALED_COLUMNS] = scaler.transform(frame[SCALED_COLUMNS])

    # SMOTE resamples TRAIN ONLY. Synthetic minority rows must never appear in
    # val/test or the metrics become fiction.
    if args.no_smote:
        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        scale_pos_weight = neg / max(pos, 1)
        X_fit, y_fit = X_train, y_train
        logger.info("SMOTE disabled; using scale_pos_weight=%.2f", scale_pos_weight)
    else:
        scale_pos_weight = 1.0
        logger.info("Applying SMOTE to training set ...")
        t0 = time.perf_counter()
        X_fit, y_fit = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
        logger.info(
            "SMOTE: %s -> %s rows (fraud %d -> %d) in %.1fs",
            f"{len(X_train):,}", f"{len(X_fit):,}", y_train.sum(), int(y_fit.sum()),
            time.perf_counter() - t0,
        )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",   # fast, and keeps inference well under the 100ms budget
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info("Training XGBoost on %s rows ...", f"{len(X_fit):,}")
    t0 = time.perf_counter()
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    logger.info("Trained in %.1fs", time.perf_counter() - t0)

    # Tune on validation, then report honestly on the untouched test set.
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val, val_proba)

    test_proba = model.predict_proba(X_test)[:, 1]
    default_metrics = evaluate(y_test, test_proba, 0.5, "TEST @ default 0.5")
    tuned_metrics = evaluate(y_test, test_proba, threshold, "TEST @ tuned")
    logger.info(
        "\n%s", classification_report(
            y_test, (test_proba >= threshold).astype(int),
            target_names=["normal", "fraud"], digits=4, zero_division=0,
        ),
    )

    # Measure single-row latency — the project's hard requirement is <100ms.
    fraud_model = FraudModel(
        booster=model, scaler=scaler,
        feature_columns=list(FEATURE_COLUMNS), threshold=threshold,
    )
    raw_sample = df[FEATURE_COLUMNS].iloc[0].to_dict()
    fraud_model.predict_proba(raw_sample)  # warm up
    t0 = time.perf_counter()
    for _ in range(100):
        fraud_model.predict_proba(raw_sample)
    predict_ms = (time.perf_counter() - t0) / 100 * 1000
    logger.info("Mean inference latency (predict only): %.2f ms", predict_ms)

    # Build the SHAP explainer. tree_path_dependent needs no background dataset,
    # which keeps explainer.pkl small, makes it load fast in the consumer, and
    # avoids the interventional path's unsupported-categorical-split error on
    # current XGBoost builds.
    logger.info("Building SHAP explainer (tree_path_dependent) ...")
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        feature_names=list(FEATURE_COLUMNS),
    )
    fraud_model.explainer = explainer

    top = fraud_model.explain_one(raw_sample, top_k=3)
    logger.info("SHAP sanity check, top 3 features: %s",
                ", ".join(f"{f['feature']}({f['shap_value']:+.4f})" for f in top))

    # Re-measure with SHAP included: this is what the consumer actually pays
    # per transaction, so it is the number that must stay under 100 ms.
    fraud_model.score_one(raw_sample)  # warm up
    t0 = time.perf_counter()
    for _ in range(100):
        fraud_model.score_one(raw_sample)
    scoring_ms = (time.perf_counter() - t0) / 100 * 1000
    logger.info("Mean end-to-end scoring latency (predict + SHAP): %.2f ms", scoring_ms)
    if scoring_ms >= 100:
        logger.warning("Scoring latency %.2f ms exceeds the 100 ms budget!", scoring_ms)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    fraud_model.save(args.model_out, args.explainer_out)

    metrics_payload = {
        "dataset": {
            "path": str(args.data), "rows": int(len(df)), "fraud": int(df["Class"].sum()),
            "fraud_rate_pct": float(df["Class"].sum() / len(df) * 100),
        },
        "config": {
            "smote": not args.no_smote, "test_size": args.test_size,
            "scale_pos_weight": float(scale_pos_weight), "random_state": RANDOM_STATE,
        },
        "chosen_threshold": float(threshold),
        "test_metrics_default_threshold": default_metrics,
        "test_metrics_tuned_threshold": tuned_metrics,
        "latency_ms": {
            "predict_only": float(predict_ms),
            "predict_plus_shap": float(scoring_ms),
        },
    }
    args.metrics_out.write_text(json.dumps(metrics_payload, indent=2))
    logger.info("Saved metrics -> %s", args.metrics_out)
    logger.info("Done. Artifacts: %s, %s", args.model_out, args.explainer_out)


if __name__ == "__main__":
    main()
