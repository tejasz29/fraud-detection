"""Real-time fraud detection dashboard built with Streamlit.

Displays live scoring results from the Kafka consumer, including:
    - Summary metrics (total scored, fraud count, fraud rate)
    - Fraud probability distribution
    - Live fraud alerts with SHAP explanations
    - Model performance metrics from training

Usage:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RESULTS_PATH = Path("results.jsonl")
METRICS_PATH = Path("consumer/metrics.json")

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon=" ",
    layout="wide",
)


# ---------------------------------------------------------------------- data
@st.cache_data(ttl=2)
def load_results() -> list[dict]:
    """Load scored transactions from the JSON-lines output file."""
    if not RESULTS_PATH.exists():
        return []
    results = []
    for line in RESULTS_PATH.read_text().splitlines():
        if line.strip():
            results.append(json.loads(line))
    return results


@st.cache_data(ttl=60)
def load_model_metrics() -> dict:
    """Load training metrics from the saved JSON file."""
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())


# ------------------------------------------------------------------ header
st.title("Fraud Detection Dashboard")
st.caption("Real-time credit card fraud scoring pipeline  |  Kafka + XGBoost + SHAP")

results = load_results()

if not results:
    st.info("No scored transactions yet. Start the consumer and producer to see live data.")
    st.stop()

# ------------------------------------------------------------- summary cards
fraud_count = sum(1 for r in results if r["is_fraud"])
legit_count = len(results) - fraud_count
fraud_rate = (fraud_count / len(results) * 100) if results else 0
avg_proba = sum(r["fraud_probability"] for r in results) / len(results)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Scored", f"{len(results):,}")
col2.metric("Fraud Alerts", f"{fraud_count:,}", delta=f"{fraud_rate:.2f}%")
col3.metric("Legitimate", f"{legit_count:,}")
col4.metric("Avg Fraud Probability", f"{avg_proba:.6f}")

st.divider()

# --------------------------------------------------- probability distribution
st.subheader("Fraud Probability Distribution")

probas = [r["fraud_probability"] for r in results]
fig_hist = px.histogram(
    x=probas,
    nbins=80,
    labels={"x": "Fraud Probability"},
    color_discrete_sequence=["#636EFA"],
)
fig_hist.update_layout(
    xaxis_title="Fraud Probability",
    yaxis_title="Count",
    height=350,
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_hist, use_container_width=True)

# --------------------------------------------------------- fraud vs legit pie
col_left, col_right = st.columns([1, 1])

with col_left:
    fig_pie = px.pie(
        names=["Legitimate", "Fraud"],
        values=[legit_count, fraud_count],
        color_discrete_sequence=["#636EFA", "#EF553B"],
        hole=0.4,
    )
    fig_pie.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    fraud_probs = [r["fraud_probability"] for r in results if r["is_fraud"]]
    if fraud_probs:
        fig_box = px.box(
            y=fraud_probs,
            labels={"y": "Fraud Probability"},
            color_discrete_sequence=["#EF553B"],
        )
        fig_box.update_layout(
            title="Fraud Alert Probabilities",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No fraud alerts yet.")

st.divider()

# -------------------------------------------------------- live fraud alerts
st.subheader("Live Fraud Alerts")

fraud_alerts = [r for r in results if r["is_fraud"]]

if fraud_alerts:
    for alert in reversed(fraud_alerts[-10:]):
        proba = alert["fraud_probability"]
        conf = alert["confidence"]
        top = alert.get("top_features", [])

        with st.expander(
            f"  FRAUD  |  Probability: {proba:.4f}  |  Confidence: {conf:.4f}"
        ):
            if top:
                st.markdown("**Top SHAP Features:**")
                for feat in top:
                    direction = "⬆ increases fraud" if feat["direction"] == "increases_fraud" else "⬇ decreases fraud"
                    st.markdown(
                        f"- **{feat['feature']}** = {feat['value']:.4f}  "
                        f"(SHAP: {feat['shap_value']:.4f}) {direction}"
                    )
else:
    st.info("No fraud alerts detected yet.")

st.divider()

# ---------------------------------------------------- model performance
st.subheader("Model Performance (from training)")

metrics = load_model_metrics()

if metrics:
    tuned = metrics.get("test_metrics_tuned_threshold", {})
    cm = tuned.get("confusion_matrix", {})

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Threshold", f"{metrics.get('chosen_threshold', 0):.4f}")
    mcol2.metric("Precision", f"{tuned.get('precision', 0):.3f}")
    mcol3.metric("Recall", f"{tuned.get('recall', 0):.3f}")
    mcol4.metric("F1 Score", f"{tuned.get('f1', 0):.3f}")

    acol1, acol2, acol3, acol4 = st.columns(4)
    acol1.metric("ROC AUC", f"{tuned.get('roc_auc', 0):.3f}")
    acol2.metric("PR AUC", f"{tuned.get('pr_auc', 0):.3f}")
    acol3.metric("Latency (predict)", f"{metrics.get('latency_ms', {}).get('predict_only', 0):.2f} ms")
    acol4.metric("Latency (+SHAP)", f"{metrics.get('latency_ms', {}).get('predict_plus_shap', 0):.2f} ms")

    if cm:
        st.markdown("**Confusion Matrix (tuned threshold):**")
        cm_fig = px.imshow(
            [[cm.get("true_negatives", 0), cm.get("false_positives", 0)],
             [cm.get("false_negatives", 0), cm.get("true_positives", 0)]],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Legitimate", "Fraud"],
            y=["Legitimate", "Fraud"],
            color_continuous_scale="Blues",
            text_auto=True,
        )
        cm_fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(cm_fig, use_container_width=True)
else:
    st.info("No training metrics found. Run `python -m consumer.train` first.")

# ----------------------------------------------------------- auto refresh
st.sidebar.markdown("### Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
if auto_refresh:
    st.rerun()
