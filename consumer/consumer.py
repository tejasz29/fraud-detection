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

from kafka import KafkaConsumer
from kafka.errors import KafkaTimeoutError

# Allow `python -m consumer` to import the sibling model module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FraudModel  # noqa: E402
from storage import DEFAULT_DB_PATH, ResultStore, utc_now_iso  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_TOPIC = "transactions"
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_MODEL_PATH = "consumer/model.pkl"
DEFAULT_EXPLAINER_PATH = "consumer/explainer.pkl"
DEFAULT_GROUP_ID = "fraud-detection-group"


@dataclass
class FraudConsumer:
    """Consumes transactions from Kafka, scores them, and logs alerts."""

    topic: str = DEFAULT_TOPIC
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS
    model_path: str = DEFAULT_MODEL_PATH
    explainer_path: str = DEFAULT_EXPLAINER_PATH
    group_id: str = DEFAULT_GROUP_ID
    db_path: str | None = DEFAULT_DB_PATH
    output_path: str | None = None
    _consumer: KafkaConsumer | None = field(default=None, repr=False)
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

    def _connect(self) -> KafkaConsumer:
        """Create and return a KafkaConsumer, retrying on connection failure."""
        while True:
            try:
                consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    auto_offset_reset="earliest",
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    enable_auto_commit=True,
                    consumer_timeout_ms=1000,
                )
                logger.info("Connected to Kafka at %s (topic=%s)", self.bootstrap_servers, self.topic)
                return consumer
            except (KafkaTimeoutError, OSError):
                logger.warning("Kafka not available — retrying in 3s")
                time.sleep(3)

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

    def _write_result(self, result: dict) -> None:
        """Append a scored result to the output file if configured."""
        if self._output_file is None:
            return
        record = {
            "fraud_probability": result["fraud_probability"],
            "is_fraud": result["is_fraud"],
            "confidence": result["confidence"],
            "top_features": result.get("top_features", []),
        }
        self._output_file.write(json.dumps(record) + "\n")
        self._output_file.flush()

    def run(self) -> dict:
        """Main loop: consume, score, log, and persist. Returns summary stats."""
        self._model = self._load_model()
        self._consumer = self._connect()

        if self.output_path:
            self._output_file = open(self.output_path, "a")
            logger.info("Writing results to %s", self.output_path)

        start_time = time.time()

        try:
            logger.info("Listening for transactions... (Ctrl-C to stop)")

            while self._running:
                batch = self._consumer.poll(timeout_ms=500)

                for _topic_partition, messages in batch.items():
                    for message in messages:
                        if not self._running:
                            break

                        self._stats["total"] += 1
                        transaction = message.value

                        try:
                            result = self._score_transaction(transaction)
                            self._write_result(result)

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
            if self._output_file:
                self._output_file.close()
            self._consumer.close()

            logger.info(
                "Consumer finished — total=%d fraud=%d legit=%d errors=%d elapsed=%.1fs",
                self._stats["total"],
                self._stats["fraud"],
                self._stats["legit"],
                self._stats["errors"],
                elapsed,
            )

        return {**self._stats, "elapsed_seconds": round(elapsed, 2)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume and score credit-card transactions in real time",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"Kafka topic to consume from (default: {DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=DEFAULT_BOOTSTRAP_SERVERS,
        help=f"Kafka broker address (default: {DEFAULT_BOOTSTRAP_SERVERS})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model.pkl (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--explainer",
        default=DEFAULT_EXPLAINER_PATH,
        help=f"Path to explainer.pkl (default: {DEFAULT_EXPLAINER_PATH})",
    )
    parser.add_argument(
        "--group-id",
        default=DEFAULT_GROUP_ID,
        help=f"Kafka consumer group ID (default: {DEFAULT_GROUP_ID})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write scored results as JSON-lines (optional)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    consumer = FraudConsumer(
        topic=args.topic,
        bootstrap_servers=args.bootstrap_servers,
        model_path=args.model,
        explainer_path=args.explainer,
        group_id=args.group_id,
        output_path=args.output,
    )

    summary = consumer.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
