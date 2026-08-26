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


# --------------------------------------------------------------------- chrome
def style_axes(fig: go.Figure, x_title: str, y_title: str) -> go.Figure:
    """Recessive grid and muted axis ink, so the data carries the emphasis."""
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif'),
        showlegend=False,   # single series — the title names it
        bargap=0.08,
    )
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(color=INK_MUTED, size=12)),
        showgrid=False, zeroline=False,
        tickfont=dict(color=INK_MUTED, size=11), linecolor=GRIDLINE,
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(color=INK_MUTED, size=12)),
        gridcolor=GRIDLINE, griddash="dot", zeroline=False,
        tickfont=dict(color=INK_MUTED, size=11),
    )
    return fig


def shap_summary(explanation: list) -> str:
    """Compact reason string for a table cell, e.g. ``V14 ↑ · V3 ↓ · V10 ↑``."""
    if not explanation:
        return "—"
    parts = []
    for f in explanation[:3]:
        arrow = "↑" if f.get("direction") == "increases_fraud" else "↓"
        parts.append(f"{f.get('feature', '?')} {arrow}")
    return "  ·  ".join(parts)


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

    # ------------------------------------------------------- live transaction feed
    st.subheader("Live Transaction Feed")

    feed = get_recent(FEED_ROWS, fraud_only=not st.session_state.show_all_feed)
    if feed:
        feed_df = pd.DataFrame([{
            # Status ships as text, never colour alone.
            "Status": "FRAUD" if r["is_fraud"] else "OK",
            "Transaction": r["transaction_id"],
            "Amount": r["amount"],
            "Fraud Prob.": r["fraud_probability"],
            "Confidence": r["confidence"],
            "Latency (ms)": r["scoring_ms"],
            "Reasons": shap_summary(r["shap_explanation"]),
            "Scored At (UTC)": r["timestamp"][11:19],
        } for r in feed])

        st.dataframe(
            feed_df, width="stretch", hide_index=True, height=320,
            column_config={
                "Amount": st.column_config.NumberColumn(format="$%.2f"),
                "Fraud Prob.": st.column_config.NumberColumn(format="%.4f"),
                "Confidence": st.column_config.NumberColumn(format="%.4f"),
                "Latency (ms)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption(f"The {len(feed_df)} most recent transactions, newest first.")
    else:
        st.info("No transactions yet.")

    st.divider()

    # --------------------------------------------------- flagged transactions
    st.subheader("Flagged Transactions")

    alerts = get_recent(ALERT_ROWS, fraud_only=True)
    if alerts:
        alert_df = pd.DataFrame([{
            "Transaction": r["transaction_id"],
            "Amount": r["amount"],
            "Fraud Prob.": r["fraud_probability"],
            "Confidence": r["confidence"],
            "Top SHAP Reasons": shap_summary(r["shap_explanation"]),
            "Scored At (UTC)": r["timestamp"][:19].replace("T", " "),
        } for r in alerts])

        st.dataframe(
            alert_df, width="stretch", hide_index=True,
            column_config={
                "Amount": st.column_config.NumberColumn(format="$%.2f"),
                "Fraud Prob.": st.column_config.ProgressColumn(
                    format="%.4f", min_value=0.0, max_value=1.0,
                ),
                "Confidence": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(f"The {len(alert_df)} most recent alerts. SHAP names why each fired.")

        with st.expander("Full SHAP detail for the latest alert"):
            latest = alerts[0]
            st.markdown(
                f"**{latest['transaction_id']}**  ·  ${latest['amount']:,.2f}  ·  "
                f"fraud probability {latest['fraud_probability']:.4f}"
            )
            for f in latest["shap_explanation"]:
                arrow = (
                    "↑ pushed toward fraud" if f["direction"] == "increases_fraud"
                    else "↓ pushed toward legitimate"
                )
                st.markdown(
                    f"- **{f['feature']}** = {f['value']:.4f} "
                    f"(SHAP {f['shap_value']:+.4f}) — {arrow}"
                )
    else:
        st.info("No fraud alerts yet.")

    st.divider()
live_view()
