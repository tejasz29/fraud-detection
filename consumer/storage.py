"""SQLite persistence for scored transactions.

Holds the write path used by ``consumer.py`` and the read helpers used by the
Streamlit dashboard, so the schema is defined in exactly one place.

Design notes:
    - WAL journal mode lets the dashboard read while the consumer is still
      writing. Without it, readers and the writer block each other and the
      dashboard stalls mid-stream.
    - Writes are batched (by count *or* elapsed time) so throughput stays high
      without the dashboard ever waiting more than ``flush_interval`` for fresh
      rows.
    - Reads are aggregated in SQL and windowed with LIMIT. The dashboard must
      never load the whole table: that cost grows without bound as the stream
      runs.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "fraud_results.db"

# `id` gives a stable insertion order for the live feed: `timestamp` alone can
# tie when many transactions are scored within the same millisecond.
SCHEMA = """
CREATE TABLE IF NOT EXISTS fraud_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id    TEXT    NOT NULL,
    amount            REAL    NOT NULL,
    is_fraud          INTEGER NOT NULL,
    confidence        REAL    NOT NULL,
    fraud_probability REAL    NOT NULL,
    shap_explanation  TEXT    NOT NULL,
    scoring_ms        REAL,
    timestamp         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fraud_results_is_fraud ON fraud_results(is_fraud);
CREATE INDEX IF NOT EXISTS idx_fraud_results_ts       ON fraud_results(timestamp);
"""

_INSERT = """
INSERT INTO fraud_results
    (transaction_id, amount, is_fraud, confidence, fraud_probability,
     shap_explanation, scoring_ms, timestamp)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (sorts lexicographically)."""
    return datetime.now(timezone.utc).isoformat()


class ResultStore:
    """Batched SQLite writer for scored transactions.

    Usable as a context manager so a crash or Ctrl-C still flushes buffered rows:

        with ResultStore("fraud_results.db") as store:
            store.insert(...)
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        batch_size: int = 50,
        flush_interval: float = 1.0,
    ) -> None:
        self.db_path = str(db_path)
        self.batch_size = max(1, batch_size)
        self.flush_interval = flush_interval
        self._pending: list[tuple] = []
        self._last_flush = time.monotonic()
        self._written = 0

        self._conn = sqlite3.connect(self.db_path, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL trades an fsync per commit for speed. A power-loss could lose the
        # last transactions, which is an acceptable trade for a monitoring stream.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        logger.info("SQLite store ready at %s (WAL)", self.db_path)

    # ------------------------------------------------------------------ writing

    def flush(self) -> int:
        """Commit buffered rows. Returns the number written."""
        if not self._pending:
            return 0
        n = len(self._pending)
        self._conn.executemany(_INSERT, self._pending)
        self._conn.commit()
        self._pending.clear()
        self._last_flush = time.monotonic()
        self._written += n
        return n

    @property
    def written(self) -> int:
        """Total rows committed by this store instance."""
        return self._written
