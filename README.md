# Fraud Detection

A real-time credit-card fraud detection system. Transactions stream through
Kafka, get scored by a trained XGBoost model (with SHAP explanations), are
persisted to SQLite, and are visualized live in a Streamlit dashboard.

## What it does

1. **Producer** generates synthetic transactions (1/sec) and publishes them to a
   Kafka topic (`transactions`).
2. **Consumer** consumes each transaction, scores it with XGBoost + SHAP, raises
   an alert on fraud, and writes every result (probability, label, SHAP reasons,
   latency) to `fraud_results.db` (SQLite).
3. **Dashboard** reads `fraud_results.db` and shows a live feed, confidence chart,
   fraud-probability histogram, confusion matrix, and a table of flagged
   transactions with their SHAP explanations.

### Example

A `$9,500` electronics purchase at 3am scores `0.94` fraud probability. Since the
decision threshold is `~0.91`, it is flagged. The dashboard shows it in red and
explains the top drivers: *"Amount +0.31, V14 −0.12, V12 +0.08"*. A normal
`$12` coffee scores `0.02` and passes.

## Dataset

`data/creditcard.csv` is the [ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 492 frauds ≈ 0.172%). Features `V1`–`V28` are PCA-
transformed (anonymized); `Amount` and `Time` are real. The extreme class
imbalance is the central challenge, which is why training uses SMOTE and careful
threshold tuning rather than plain accuracy.

> The CSV is **gitignored** (`data/*.csv`). Download it from Kaggle and place it
> at `data/creditcard.csv` before training or producing from real data.

## Architecture

```
producer/  ──transactions──▶  Kafka (kafka_2.13-3.9.2, KRaft)
                                    │
consumer/  ◀── consumes ───────────┘
   ├─ scores with XGBoost + SHAP
   └─ writes ──▶ fraud_results.db (SQLite) ──▶ dashboard/ (Streamlit)
```

## Prerequisites

- **Python 3.10+** with the dependencies in `requirements.txt`
- **Java** (Kafka is a JVM app)
- **Local Kafka** at `kafka_2.13-3.9.2/` (already downloaded; pre-formatted,
  `kafka-logs/` has a valid `cluster.id`). Config: `config/kafka-server.properties`.

Install Python deps:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start (one command)

```powershell
.\run.ps1
```

This starts the Kafka broker, producer, consumer, and dashboard, then tears them
all down on Ctrl-C. Open the dashboard at **http://localhost:8501**.

> The Python processes run in hidden windows. If one crashes you won't see the
> traceback — check the terminal output or add log redirection.

## Manual start

Run each in its own terminal (Kafka first):

```powershell
# 1. Kafka broker (KRaft, already formatted)
kafka_2.13-3.9.2\bin\windows\kafka-server-start.bat config\kafka-server.properties

# 2. Producer
python -m producer --data data/creditcard.csv

# 3. Consumer
python -m consumer

# 4. Dashboard
streamlit run dashboard/app.py
```

## Training the model

The trained artifacts (`consumer/model.pkl`, `consumer/explainer.pkl`,
`consumer/metrics.json`) are gitignored. To (re)train:

```powershell
python consumer/train.py --data data/creditcard.csv
```

This runs the full pipeline: stratified split, train-only scaler + SMOTE,
XGBoost, validation-based threshold tuning, SHAP explainer, and a <100 ms
latency check. Metrics are written to `consumer/metrics.json`.

## Configuration

| Setting            | Default             | Where |
|--------------------|---------------------|-------|
| Kafka bootstrap    | `localhost:9092`    | `producer.py` / `consumer.py` |
| Topic              | `transactions`      | `consumer.py` |
| Model / explainer  | `consumer/model.pkl`| `consumer.py` |
| Result DB          | `fraud_results.db`  | `consumer/storage.py` |
| Broker config      | `config/kafka-server.properties` | `run.ps1` |

## Project layout

```
config/kafka-server.properties   Kafka KRaft broker config (relative log.dirs)
producer/                        Kafka producer + transaction simulator
consumer/                       Consumer, model, storage, training
  train.py  model.py  consumer.py  storage.py  metrics.json
dashboard/                      Streamlit real-time UI
data/creditcard.csv             Dataset (gitignored; download from Kaggle)
run.ps1                         One-command launcher / teardown
requirements.txt                Python dependencies
```

## Known gaps

- No automated tests or CI.
- No `.env`/config layer — values are hardcoded per module.
- Transport is hard-wired to Kafka (no pluggable `Transport` interface).
- No model drift monitoring or scheduled retraining.
- `fraud_results.db` / `kafka-logs` grow unbounded (no retention job).
