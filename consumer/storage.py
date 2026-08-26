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
