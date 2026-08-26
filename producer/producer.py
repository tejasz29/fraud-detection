"""Kafka producer that streams credit-card transactions for real-time scoring.

Reads ``creditcard.csv`` row-by-row and publishes each transaction as a JSON
message to a Kafka topic.  Designed to be run alongside the consumer service
so that ``consumer/model.py`` can score transactions in real time.

Features:
    - Rate-limiting via ``--delay`` to simulate realistic transaction velocity
    - Shuffle option for stress-testing with mixed timestamps
    - Graceful shutdown on Ctrl-C with a summary of sent messages

Usage:
    python -m producer.producer
    python -m producer.producer --topic transactions --delay 0.5 --shuffle
    python -m producer.producer --data data/creditcard.csv --bootstrap-servers localhost:9092
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaTimeoutError

logger = logging.getLogger(__name__)

# Features the consumer model expects (matches consumer/model.py FEATURE_COLUMNS).
FEATURE_COLUMNS: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

DEFAULT_TOPIC = "transactions"
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_DATA_PATH = "data/creditcard.csv"


@dataclass
class TransactionProducer:
    """Streams transactions from a CSV file to a Kafka topic."""

    topic: str = DEFAULT_TOPIC
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS
    data_path: str = DEFAULT_DATA_PATH
    delay: float = 0.0
    shuffle: bool = False
    _producer: KafkaProducer | None = field(default=None, repr=False)
    _sent: int = field(default=0, repr=False)
    _running: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, _signum: int, _frame: object) -> None:
        logger.info("Shutdown signal received — finishing current batch")
        self._running = False

    def _connect(self) -> KafkaProducer:
        """Create and return a KafkaProducer, retrying on connection failure."""
        while True:
            try:
                producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    acks="all",
                    retries=3,
                )
                logger.info("Connected to Kafka at %s", self.bootstrap_servers)
                return producer
            except (KafkaTimeoutError, OSError):
                logger.warning("Kafka not available at %s — retrying in 3s", self.bootstrap_servers)
                time.sleep(3)

    def _load_transactions(self) -> pd.DataFrame:
        """Load and optionally shuffle the transaction dataset."""
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path)
        logger.info("Loaded %d transactions from %s", len(df), path)

        if self.shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            logger.info("Transactions shuffled")

        return df

    def _row_to_transaction(self, row: pd.Series) -> dict:
        """Convert a DataFrame row to a consumer-ready transaction dict."""
        return {col: float(row[col]) for col in FEATURE_COLUMNS}

    def run(self) -> dict:
        """Stream all transactions to Kafka and return a summary dict."""
        self._producer = self._connect()
        df = self._load_transactions()

        sent = 0
        errors = 0
        start_time = time.time()

        try:
            for idx, row in df.iterrows():
                if not self._running:
                    break

                transaction = self._row_to_transaction(row)

                try:
                    self._producer.send(self.topic, value=transaction)
                    sent += 1

                    if sent % 500 == 0:
                        elapsed = time.time() - start_time
                        rate = sent / elapsed if elapsed > 0 else 0
                        logger.info(
                            "Sent %d / %d transactions (%.1f txn/s)",
                            sent,
                            len(df),
                            rate,
                        )
                except Exception as e:
                    errors += 1
                    logger.error("Failed to send transaction %d: %s", idx, e)

                if self.delay > 0:
                    time.sleep(self.delay)

            self._producer.flush()
        finally:
            elapsed = time.time() - start_time
            self._producer.close()
            logger.info(
                "Producer finished — sent %d, errors %d, elapsed %.1fs",
                sent,
                errors,
                elapsed,
            )

        return {
            "sent": sent,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 2),
            "topic": self.topic,
            "total_rows": len(df),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream credit-card transactions to Kafka for real-time fraud scoring",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to creditcard.csv (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"Kafka topic to publish to (default: {DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=DEFAULT_BOOTSTRAP_SERVERS,
        help=f"Kafka broker address (default: {DEFAULT_BOOTSTRAP_SERVERS})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between messages (default: 0 = as fast as possible)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle transactions before sending",
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

    producer = TransactionProducer(
        topic=args.topic,
        bootstrap_servers=args.bootstrap_servers,
        data_path=args.data,
        delay=args.delay,
        shuffle=args.shuffle,
    )

    summary = producer.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
