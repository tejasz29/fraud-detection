"""Kafka consumer that scores transactions in real time using FraudModel.

Reads transactions from a Kafka topic, scores each one with the trained
XGBoost model, logs fraud alerts, and persists every result to SQLite
(``fraud_results.db``) for the dashboard to read.

Usage:
    python -m consumer
    python -m consumer --topic transactions --model consumer/model.pkl
    python -m consumer --db fraud_results.db --output results.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Allow `python -m consumer` to import the sibling model module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_settings  # noqa: E402
from model import FraudModel  # noqa: E402
from storage import ResultStore, utc_now_iso  # noqa: E402
from transport import Transport, get_transport  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "consumer/model.pkl"
DEFAULT_EXPLAINER_PATH = "consumer/explainer.pkl"


@dataclass
class FraudConsumer:
    """Consumes transactions from Kafka, scores them, and logs alerts."""

    transport: Transport
    model_path: str = DEFAULT_MODEL_PATH
    explainer_path: str = DEFAULT_EXPLAINER_PATH
    db_path: str | None = None
    output_path: str | None = None
    _model: FraudModel | None = field(default=None, repr=False)
    _store: ResultStore | None = field(default=None, repr=False)
    _output_file: object | None = field(default=None, repr=False)
    _running: bool = field(default=True, repr=False)
    _stats: dict = field(default_factory=lambda: {
        "total": 0, "fraud": 0, "legit": 0, "errors": 0,
    }, repr=False)

    def __post_init__(self) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, _signum: int, _frame: object) -> None:
        logger.info("Shutdown signal received — finishing current batch")
        self._running = False

    def _load_model(self) -> FraudModel:
        """Load the trained model and SHAP explainer from disk."""
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path} — run consumer/train.py first")

        explainer_path = Path(self.explainer_path)
        model = FraudModel.load(model_path, explainer_path)
        logger.info("Loaded model from %s (threshold=%.4f)", model_path, model.threshold)
        return model

    def _score_transaction(self, transaction: dict) -> dict:
        """Score one transaction and return the enriched result.

        Records the model's own latency (``scoring_ms``) so the dashboard can
        show live evidence against the project's 100 ms budget.
        """
        started = time.perf_counter()
        result = self._model.score_one(transaction)
        result["scoring_ms"] = (time.perf_counter() - started) * 1000.0
        # Producer supplies transaction_id; fall back so a hand-crafted or
        # replayed message without one still persists cleanly.
        result["transaction_id"] = str(transaction.get("transaction_id", "unknown"))
        result["amount"] = float(transaction.get("Amount", 0.0))
        result["timestamp"] = utc_now_iso()
        result["transaction"] = transaction
        return result

    def _log_alert(self, result: dict) -> None:
        """Log a fraud alert with relevant details."""
        proba = result["fraud_probability"]
        confidence = result["confidence"]
        top = result.get("top_features", [])

        top_str = ", ".join(
            f"{f['feature']}({f['direction']})" for f in top
        ) if top else "n/a"

        logger.warning(
            "FRAUD ALERT | %s | amount=%.2f | prob=%.4f conf=%.4f | top features: %s",
            result["transaction_id"],
            result["amount"],
            proba,
            confidence,
            top_str,
        )

    def _persist(self, result: dict) -> None:
        """Write the scored result to SQLite and, if configured, the JSONL file."""
        if self._store is not None:
            self._store.insert(
                transaction_id=result["transaction_id"],
                amount=result["amount"],
                is_fraud=result["is_fraud"],
                confidence=result["confidence"],
                fraud_probability=result["fraud_probability"],
                shap_explanation=result.get("top_features", []),
                scoring_ms=result.get("scoring_ms"),
                timestamp=result["timestamp"],
            )
        self._write_result(result)

    def _write_result(self, result: dict) -> None:
        """Append a scored result to the optional JSON-lines output file."""
        if self._output_file is None:
            return
        record = {
            "transaction_id": result["transaction_id"],
            "amount": result["amount"],
            "fraud_probability": result["fraud_probability"],
            "is_fraud": result["is_fraud"],
            "confidence": result["confidence"],
            "scoring_ms": result.get("scoring_ms"),
            "top_features": result.get("top_features", []),
            "timestamp": result["timestamp"],
        }
        self._output_file.write(json.dumps(record) + "\n")
        self._output_file.flush()

    def run(self) -> dict:
        """Main loop: consume, score, log, and persist. Returns summary stats."""
        self._model = self._load_model()
        self.transport.connect()

        if self.db_path:
            self._store = ResultStore(self.db_path)

        if self.output_path:
            self._output_file = open(self.output_path, "a")
            logger.info("Writing results to %s", self.output_path)

        start_time = time.time()

        try:
            logger.info("Listening for transactions... (Ctrl-C to stop)")

            for transaction in self.transport.receive(should_stop=lambda: not self._running):
                self._stats["total"] += 1

                try:
                    result = self._score_transaction(transaction)
                    self._persist(result)

                    if result["is_fraud"]:
                        self._stats["fraud"] += 1
                        self._log_alert(result)
                    else:
                        self._stats["legit"] += 1

                    if self._stats["total"] % 500 == 0:
                        elapsed = time.time() - start_time
                        rate = self._stats["total"] / elapsed if elapsed > 0 else 0
                        logger.info(
                            "Processed %d (%.1f txn/s) | fraud=%d legit=%d",
                            self._stats["total"],
                            rate,
                            self._stats["fraud"],
                            self._stats["legit"],
                        )

                except Exception as e:
                    self._stats["errors"] += 1
                    logger.error("Failed to score transaction: %s", e)

        finally:
            elapsed = time.time() - start_time
            # Flush buffered rows before closing so a Ctrl-C never loses the
            # last partial batch.
            if self._store:
                self._store.close()
            if self._output_file:
                self._output_file.close()
            self.transport.close()

            logger.info(
                "Consumer finished — total=%d fraud=%d legit=%d errors=%d elapsed=%.1fs",
                self._stats["total"],
                self._stats["fraud"],
                self._stats["legit"],
                self._stats["errors"],
                elapsed,
            )

        return {**self._stats, "elapsed_seconds": round(elapsed, 2)}


def parse_args(settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume and score credit-card transactions in real time",
    )
    parser.add_argument(
        "--model",
        default=settings.model_path,
        help=f"Path to model.pkl (default: {settings.model_path})",
    )
    parser.add_argument(
        "--explainer",
        default=settings.explainer_path,
        help=f"Path to explainer.pkl (default: {settings.explainer_path})",
    )
    parser.add_argument(
        "--transport",
        default=settings.transport,
        help=f"Transport backend (default: {settings.transport})",
    )
    parser.add_argument(
        "--db",
        default=settings.db_path,
        help=f"SQLite database for scored results (default: {settings.db_path})",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable SQLite persistence",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Also write scored results as JSON-lines to this path (optional)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    settings = load_settings()
    args = parse_args(settings)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Allow --transport to override the .env setting for this run.
    effective = settings
    if args.transport != settings.transport:
        from dataclasses import replace

        effective = replace(settings, transport=args.transport)

    consumer = FraudConsumer(
        transport=get_transport(effective),
        model_path=args.model,
        explainer_path=args.explainer,
        db_path=None if args.no_db else args.db,
        output_path=args.output,
    )

    summary = consumer.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
