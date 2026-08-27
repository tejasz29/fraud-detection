# Fraud Detection

A real-time credit-card fraud detection system. Transactions stream through
Kafka, get scored by a trained XGBoost model (with SHAP explanations), are
persisted to SQLite, and are visualized live in a Streamlit dashboard.

## About this project

This is a **learning project**, not a production fraud system. It was built to
string together the full stack of a real-time ML service end to end and to
understand the hard parts that tutorials skip:

- **Streaming architecture** — producer/consumer over Kafka, and why a
  transport abstraction matters.
- **Imbalanced classification** — with fraud at ~0.17% of transactions, naive
  accuracy is meaningless; the project leans on SMOTE, threshold tuning, and
  precision/recall/PR-AUC instead of accuracy.
- **Serving latency** — keeping single-transaction scoring (predict + SHAP)
  under a 100 ms budget, which drove the numpy/`inplace_predict` path over a
  pandas one.
- **Explainability** — every prediction ships with SHAP reasons so a "fraud"
  call is auditable, not a black box.
- **Operational reality** — config via `.env`, pluggable transports, log capture,
  and a one-command launcher, because a demo that only runs after five manual
  terminal commands isn't a demo anyone re-runs.

Treat the numbers as illustrative. The model is trained on the public ULB
dataset, the broker runs locally, and there is no live card data, no retraining
loop, and no monitoring — the gaps a real deployment would need are listed under
[Known gaps](#known-gaps).

## What it does

1. **Producer** streams transactions from `data/creditcard.csv` (1/sec by
   default) and publishes them to a Kafka topic (`transactions`).
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

All tunables live in `config.py` and are read from the environment (or a `.env`
file via `python-dotenv`). Copy `.env.example` to `.env` to override:

```powershell
copy .env.example .env
```

| Setting                  | Default             | Env var |
|--------------------------|---------------------|---------|
| Transport backend        | `kafka`             | `TRANSPORT` |
| Kafka bootstrap          | `localhost:9092`    | `KAFKA_BOOTSTRAP_SERVERS` |
| Topic                    | `transactions`      | `KAFKA_TOPIC` |
| Model / explainer        | `consumer/model.pkl`| `MODEL_PATH` / `EXPLAINER_PATH` |
| Result DB                | `fraud_results.db`  | `DB_PATH` |
| Dashboard port           | `8501`              | `DASHBOARD_PORT` |
| Broker config            | `config/kafka-server.properties` | `run.ps1` |

CLI flags (`--transport`, `--model`, `--db`, …) still override the env for a
single run.

## Pluggable transport

Producer and consumer depend only on the `Transport` interface in `transport/`,
never on Kafka directly:

- `transport/kafka_transport.py` — `KafkaTransport` (default).
- `transport/memory_transport.py` — `MemoryTransport`, an in-process queue for
  tests and single-process demos (needs no broker).

Switch backends by setting `TRANSPORT=memory` (or `--transport memory`); no code
changes in the services. A cross-process local transport (e.g. file/SQLite-backed)
can be added later by implementing `Transport`.

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
