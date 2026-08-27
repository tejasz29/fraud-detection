"""Transaction producer: streams credit-card transactions to a Transport.

Reads ``creditcard.csv`` row-by-row and publishes each transaction as a message
on the configured :class:`~transport.Transport` (Kafka by default). The producer
knows nothing about the broker implementation — that lives in ``transport/``.

Usage:
    python -m producer
    python -m producer --data data/creditcard.csv --delay 0.5 --shuffle
    python -m producer --transport memory
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

from config import load_settings
from transport import Transport, get_transport

logger = logging.getLogger(__name__)

# Features the consumer model expects (matches consumer/model.py FEATURE_COLUMNS).
FEATURE_COLUMNS: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

DEFAULT_DATA_PATH = "data/creditcard.csv"
# The spec calls for a 1 transaction/second stream. Pass --delay 0 to replay the
# dataset at full speed instead (useful for load-testing the consumer).
DEFAULT_DELAY = 1.0


@dataclass
class TransactionProducer:
    """Streams transactions from a CSV file to a Transport."""

    transport: Transport
    data_path: str = DEFAULT_DATA_PATH
    delay: float = DEFAULT_DELAY
    shuffle: bool = False
    _sent: int = field(default=0, repr=False)
    _running: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, _signum: int, _frame: object) -> None:
        logger.info("Shutdown signal received — finishing current batch")
        self._running = False

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

    def _row_to_transaction(self, row: pd.Series, index: int) -> dict:
        """Convert a DataFrame row to a consumer-ready transaction dict.

        ``transaction_id`` is added so a scored result can be traced back to the
        source row; the model ignores keys outside FEATURE_COLUMNS.
        """
        transaction = {col: float(row[col]) for col in FEATURE_COLUMNS}
        transaction["transaction_id"] = f"txn-{index:06d}"
        return transaction

    def run(self) -> dict:
        """Stream all transactions to the transport and return a summary dict."""
        self.transport.connect()
        df = self._load_transactions()

        sent = 0
        errors = 0
        start_time = time.time()

        try:
            for idx, row in df.iterrows():
                if not self._running:
                    break

                transaction = self._row_to_transaction(row, idx)

                try:
                    self.transport.send(transaction)
                    sent += 1

                    if sent % 500 == 0:
                        elapsed = time.time() - start_time
                        rate = sent / elapsed if elapsed > 0 else 0
                        logger.info(
                            "Sent %d / %d transactions (%.1f txn/s)",
                            sent, len(df), rate,
                        )
                except Exception as e:
                    errors += 1
                    logger.error("Failed to send transaction %d: %s", idx, e)

                if self.delay > 0:
                    time.sleep(self.delay)

            self.transport.flush()
        finally:
            elapsed = time.time() - start_time
            self.transport.close()
            logger.info(
                "Producer finished — sent %d, errors %d, elapsed %.1fs",
                sent, errors, elapsed,
            )

        return {
            "sent": sent,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 2),
            "total_rows": len(df),
        }


def parse_args(settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream credit-card transactions to the configured transport for real-time fraud scoring",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to creditcard.csv (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--transport",
        default=settings.transport,
        help=f"Transport backend (default: {settings.transport})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds to wait between messages (default: {DEFAULT_DELAY} = 1 txn/s; "
             "use 0 for full speed)",
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

    producer = TransactionProducer(
        transport=get_transport(effective),
        data_path=args.data,
        delay=args.delay,
        shuffle=args.shuffle,
    )

    summary = producer.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
