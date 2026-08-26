"""Real-time fraud detection dashboard built with Streamlit.

Reads scored transactions from the SQLite database the consumer writes to, and
shows:
    - Summary stat tiles (total scored, fraud alerts, fraud rate, latency)
    - Live transaction feed (every transaction, newest first)
    - Flagged transactions table with SHAP reasons
    - Confidence score bar chart
    - Fraud probability distribution
    - Model performance from training

Usage:
    streamlit run dashboard/app.py

Two decisions shape this file:

*Bounded reads.* Every query is either a SQL aggregate or LIMIT-windowed.
Loading the whole table would make each refresh slower the longer the stream
runs, which is precisely what an always-on monitor must not do.

*Fragment-scoped refresh.* The live section is an ``st.fragment`` on a timer, so
only it redraws every few seconds. The obvious alternative — ``time.sleep()``
then ``st.rerun()`` — holds a server thread open permanently, pins the app in a
"Running" state, redraws static content needlessly, and resets scroll position
on every tick.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Reuse the consumer's storage layer so the schema lives in exactly one place.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "consumer"))
from storage import (  # noqa: E402
    read_probabilities,
    read_recent,
    read_summary,
)

DB_PATH = _PROJECT_ROOT / "fraud_results.db"
METRICS_PATH = _PROJECT_ROOT / "consumer" / "metrics.json"

REFRESH_SECONDS = 3          # spec: auto-refresh every 3 seconds
LATENCY_BUDGET_MS = 100      # spec: score each transaction in under 100 ms
FEED_ROWS = 25               # live feed window
ALERT_ROWS = 50              # flagged-transaction table window
CHART_SAMPLE = 5000          # rows charted (bounded, newest-first)

# Validated palette (dataviz skill; checked with scripts/validate_palette.js).
# Single-series charts use SERIES_BLUE. Status colours are reserved and always
# ship beside a text label, never as the only cue.
SERIES_BLUE = "#2a78d6"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"]
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡",
    layout="wide",
)


# ---------------------------------------------------------------------- data
@st.cache_data(ttl=REFRESH_SECONDS)
def get_summary() -> dict:
    return read_summary(DB_PATH)


@st.cache_data(ttl=REFRESH_SECONDS)
def get_recent(limit: int, fraud_only: bool) -> list[dict]:
    return read_recent(DB_PATH, limit=limit, fraud_only=fraud_only)


@st.cache_data(ttl=REFRESH_SECONDS)
def get_probabilities(limit: int) -> list[float]:
    return read_probabilities(DB_PATH, limit=limit)


@st.cache_data(ttl=60)
def get_model_metrics() -> dict:
    """Training metrics. Static between retrains, so cached far longer."""
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text())
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("### Settings")
st.sidebar.checkbox(
    f"Auto-refresh ({REFRESH_SECONDS}s)", value=True, key="auto_refresh",
)
st.sidebar.checkbox(
    "Feed: show legitimate too", value=True, key="show_all_feed",
)
st.sidebar.caption(f"Database\n\n`{DB_PATH.name}`")

# ------------------------------------------------------------------- header
st.title("Fraud Detection Dashboard")
st.caption("Real-time credit card fraud scoring  ·  Kafka → XGBoost → SHAP")


# ----------------------------------------------------------------- live view
# Everything stream-derived lives in this fragment; the timer redraws only it.
@st.fragment(run_every=REFRESH_SECONDS if st.session_state.auto_refresh else None)
def live_view() -> None:
    if not DB_PATH.exists():
        st.info(
            f"No database at `{DB_PATH.name}` yet. Start the consumer and producer:\n\n"
            "```\npython -m consumer\npython -m producer.producer\n```"
        )
        return

    summary = get_summary()
    if summary["total"] == 0:
        st.info("Database is empty — waiting for the producer to send transactions.")
        return

    metrics = get_model_metrics()

    # ---------------------------------------------------------- summary tiles
    # Deliberately stat tiles, not a pie: at a fraud rate this low a
    # proportional slice is invisible, so the number communicates and the
    # chart would not.
    within_budget = summary["avg_scoring_ms"] < LATENCY_BUDGET_MS
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Scored", f"{summary['total']:,}")
    c2.metric("Fraud Alerts", f"{summary['fraud']:,}")
    c3.metric("Fraud Rate", f"{summary['fraud_rate']:.3f}%")
    c4.metric("Flagged Amount", f"${summary['fraud_amount']:,.2f}")
    c5.metric(
        "Avg Scoring Latency",
        f"{summary['avg_scoring_ms']:.1f} ms",
        delta=f"within {LATENCY_BUDGET_MS} ms budget" if within_budget else "over budget",
        delta_color="normal" if within_budget else "inverse",
    )

    st.divider()
live_view()
