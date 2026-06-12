# Credit Card Fraud Detection System

> End-to-end fraud detection platform combining supervised ML, unsupervised anomaly detection, explainable AI, and a dynamic risk engine — deployed as a containerized FastAPI + Streamlit application on AWS EC2.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED?style=flat-square)
![AWS EC2](https://img.shields.io/badge/AWS-EC2-FF9900?style=flat-square)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet?style=flat-square)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Live Deployment](#live-deployment)
3. [Performance Targets](#performance-targets)
4. [System Architecture](#system-architecture)
5. [Dataset](#dataset)
6. [Machine Learning Pipeline](#machine-learning-pipeline)
7. [Models](#models)
8. [Dynamic Risk Engine](#dynamic-risk-engine)
9. [SHAP Explainability](#shap-explainability)
10. [FastAPI Backend](#fastapi-backend)
11. [Streamlit Frontend](#streamlit-frontend)
12. [Project Structure](#project-structure)
13. [Local Setup](#local-setup)
14. [Docker Deployment](#docker-deployment)
15. [AWS EC2 Deployment](#aws-ec2-deployment)
16. [Tech Stack](#tech-stack)
17. [Roadmap](#roadmap)

---

## Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem — fraudulent transactions account for less than 0.2% of all transactions in real-world datasets. A naive classifier that always predicts "not fraud" achieves 99.8% accuracy yet catches zero fraud. This project is designed to solve that correctly.

The system is built as a production-style pipeline with four core layers:

- **Detection** — a stacking ensemble of six supervised models, each tuned for imbalanced data
- **Anomaly scoring** — an Isolation Forest adds an unsupervised anomaly feature before supervised training
- **Risk decision** — a weighted formula converts ML output into a banking-style three-tier decision
- **Explainability** — SHAP values explain every individual prediction, making the system auditable

All components are served via a FastAPI backend, exposed through a Streamlit dashboard, and deployed inside a single Docker container on AWS EC2.

---

## Live Deployment

| Service | URL |
|---|---|
| Streamlit Dashboard | `http://YOUR_PUBLIC_IP:8502` |
| FastAPI Interactive Docs | `http://YOUR_PUBLIC_IP:8001/docs` |
| FastAPI ReDoc | `http://YOUR_PUBLIC_IP:8001/redoc` |

---

## Performance Targets

| Metric | Target | Notes |
|---|---|---|
| Precision | > 93% | Minimizes false fraud alerts |
| Recall | > 94% | Catches real fraud transactions |
| F1 Score | > 92% | Harmonic balance of precision and recall |
| ROC-AUC | > 0.97 | Discrimination across all thresholds |
| Threshold strategy | F1-optimal | Via precision-recall curve — not recall-only |

> Thresholds are tuned per model via precision-recall curves and not fixed at 0.5. This avoids the common pitfall of optimizing only for recall and generating excessive false positives.

---

## System Architecture

```
                        ┌─────────────────────────┐
                        │     Streamlit Frontend   │
                        │         (app.py)         │
                        │                          │
                        │  Manual │ CSV │ Batch    │
                        └──────────┬──────────────┘
                                   │  HTTP POST /predict
                                   ▼
                        ┌─────────────────────────┐
                        │     FastAPI Backend      │
                        │        (main.py)         │
                        └──────────┬──────────────┘
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
   ┌───────────────────────┐             ┌─────────────────────────┐
   │   Isolation Forest    │             │    Hybrid ML Model       │
   │  (anomaly_score)      │────────────▶│  Stacking Ensemble       │
   └───────────────────────┘             │  (fraud_probability)     │
                                         └────────────┬────────────┘
                                                      │
                                         ┌────────────▼────────────┐
                                         │   Dynamic Risk Engine    │
                                         │  Risk Score = P×70+A×30  │
                                         └────────────┬────────────┘
                                                      │
                               ┌──────────────────────┼──────────────────────┐
                               ▼                      ▼                      ▼
                          LOW (0–40)           MEDIUM (40–70)         HIGH (70–100)
                          Approve              OTP verify             Block
                               │                      │                      │
                                              ┌───────▼───────┐
                                              │ SHAP values    │
                                              │ (per feature)  │
                                              └───────────────┘
                                                      │
                                              JSON response to UI
```

---

## Dataset

The project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains real anonymized European credit card transactions from September 2013.

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.172%) |
| Features | 30 (V1–V28 + Amount + Time) |
| Label column | `Class` (0 = legitimate, 1 = fraud) |

### Feature descriptions

| Feature | Description |
|---|---|
| `V1` – `V28` | PCA-transformed components (original features anonymized for privacy) |
| `Amount` | Transaction amount in EUR |
| `Time` | Seconds elapsed since first transaction in the dataset |
| `Class` | Binary label: 0 = not fraud, 1 = fraud |

> Because the original banking attributes are privacy-protected, all feature engineering is done on the PCA components. `Amount` and `Time` are scaled separately.

---

## Machine Learning Pipeline

### 1. Data preprocessing

```python
# Key preprocessing steps
df.dropna(inplace=True)                          # Remove missing values
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time']   = scaler.fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)
```

- Missing value check and removal
- `Amount` and `Time` scaled with `StandardScaler` → `scaled_amount`, `scaled_time`
- Original `Amount` and `Time` columns dropped after scaling
- Train/test split: 80/20, stratified on `Class`

### 2. Isolation Forest anomaly scoring

An unsupervised Isolation Forest is trained on the full feature set to generate an anomaly score for each transaction. This score is then injected as a new feature `anomaly_score` before supervised training.

```python
iso_forest = IsolationForest(contamination=0.002, random_state=42)
iso_forest.fit(X_train)

# anomaly_score: lower = more anomalous
X_train['anomaly_score'] = iso_forest.decision_function(X_train)
X_test['anomaly_score']  = iso_forest.decision_function(X_test)
```

This approach gives the supervised models an additional signal that is independent of their own learned decision boundaries — improving detection on novel fraud patterns.

### 3. Handling class imbalance

Fraud datasets are severely imbalanced. A single strategy does not work well for all model types:

| Model | Strategy | Reason |
|---|---|---|
| Logistic Regression | SMOTE oversampling | Linear model benefits from balanced training distribution |
| Random Forest | `class_weight='balanced'` | Native support; SMOTE on trees can overfit |
| XGBoost | `scale_pos_weight` | Efficient built-in weighting |
| LightGBM | `is_unbalance=True` | Native reweighting for imbalanced data |
| CatBoost | `auto_class_weights='Balanced'` | Must be cast to native float to avoid type errors |
| Stacking Ensemble | Inherits from base estimators | Meta-learner trained on balanced base outputs |

> **Key lesson:** Applying SMOTE globally before tree-based model training can cause probability leakage and inflate F1 scores artificially. SMOTE is only applied for Logistic Regression in this pipeline.

### 4. Model training and threshold tuning

Each model is trained independently. Optimal decision thresholds are selected by maximizing F1 score on the validation precision-recall curve — not fixed at 0.5.

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[f1_scores.argmax()]
```

Saved thresholds are stored in `fraud_config_hybrid.pkl` and loaded at inference time.

### 5. Stacking ensemble

The final model stacks all five base classifiers using a Logistic Regression meta-learner:

```
Base estimators:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost (excluded from stacking estimators due to compatibility)

Meta-learner:
  - Logistic Regression (cross-validated on out-of-fold predictions)
```

> **Important:** CatBoost is trained and evaluated independently but excluded from the `StackingClassifier` estimators list due to probability calibration issues. Its predictions are ensemble-averaged separately if needed.

---

## Models

| Model | Imbalance strategy | Notes |
|---|---|---|
| Logistic Regression | SMOTE | Baseline linear model; threshold tuned |
| Random Forest | `class_weight='balanced'` | Strong baseline; fast inference |
| XGBoost | `scale_pos_weight` | Best single-model AUC in most runs |
| LightGBM | `is_unbalance=True` | Fastest training; competitive precision |
| CatBoost | `auto_class_weights='Balanced'` | Handles categoricals natively |
| **Stacking Ensemble** | **Inherited** | **Final deployed model** |

All models are evaluated on: Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix.

---

## Dynamic Risk Engine

Rather than returning a binary fraud/not-fraud prediction, the system computes a continuous risk score that blends the supervised model's fraud probability with the unsupervised anomaly signal.

### Formula

```
Risk Score = (Fraud Probability × 70) + (Anomaly Score × 30)
```

- `Fraud Probability` — output of the stacking ensemble (0.0 to 1.0)
- `Anomaly Score` — normalized Isolation Forest decision score (0.0 to 1.0)
- Weights 70/30 reflect that the supervised model is more reliable but anomaly score adds coverage for novel patterns

### Decision table

| Risk Score | Level | Banking Action | Use case |
|---|---|---|---|
| 0 – 40 | 🟢 Low | Approve transaction | Normal spending patterns |
| 40 – 70 | 🟡 Medium | Require OTP verification | Unusual but not definitive fraud signal |
| 70 – 100 | 🔴 High | Block transaction | High-confidence fraud detected |

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) is integrated to explain every individual prediction, making the system compliant with explainability requirements common in production ML deployments.

### What it provides

- **Feature contribution** — which features pushed the prediction toward fraud or legitimate
- **Direction** — whether each feature increased or decreased the fraud probability
- **Magnitude** — how strongly each feature contributed

### Top contributing features (typical)

```
High-risk transaction example:

Feature        SHAP value
──────────────────────────
V14            -0.82   ← strongest fraud indicator
V17            -0.61
V10            -0.43
anomaly_score  +0.38
V4             +0.21
scaled_amount  -0.14
```

Negative SHAP values push toward fraud (Class 1); positive values push toward legitimate (Class 0) — or vice versa depending on SHAP baseline orientation. Displayed in the Streamlit dashboard alongside the risk score for each prediction.

---

## FastAPI Backend

The backend is built with FastAPI and served by Uvicorn. It loads the three `.pkl` artifacts at startup and exposes inference via a single POST endpoint.

### Startup

```python
# On startup, three artifacts are loaded once into memory:
fraud_model   = joblib.load("fraud_model_hybrid.pkl")   # Stacking ensemble
iso_forest    = joblib.load("isolation_forest.pkl")      # Anomaly scorer
fraud_config  = joblib.load("fraud_config_hybrid.pkl")   # Thresholds + scaler config
```

### Endpoints

#### `GET /`

Health check.

**Response:**
```json
{ "message": "Fraud Detection API Running" }
```

#### `POST /predict`

Run inference on a single transaction.

**Request body:**
```json
{
  "features": [0.1, -1.35, 0.82, -0.25, 0.45, -0.63, 1.20, 0.07,
               0.33, -0.44, -0.19, 0.58, -0.87, 0.16, -0.32, 0.91,
               -0.14, 0.74, -0.09, 0.23, 0.11, -0.55, 0.38, -0.27,
               0.08, -0.19, 0.04, 0.62, 149.62, 0.244]
}
```

> 30 values: `scaled_time`, `V1`–`V28`, `scaled_amount`. The backend computes `anomaly_score` internally before inference.

**Response:**
```json
{
  "fraud_probability": 0.97,
  "is_fraud": true,
  "anomaly_score": 0.82,
  "risk_score": 91.5,
  "risk_level": "HIGH RISK",
  "decision": "BLOCK TRANSACTION",
  "shap_values": {
    "V14": -0.82,
    "V17": -0.61,
    "V10": -0.43,
    "anomaly_score": 0.38,
    "V4": 0.21
  }
}
```

### Running the backend directly

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Interactive API docs available at `http://localhost:8001/docs`.

---

## Streamlit Frontend

The dashboard is the primary user interface. It communicates with the FastAPI backend over HTTP and visualizes results in real time.

### Mode 1 — Manual transaction input

Enter each feature value individually via a form:
- `scaled_time` and `scaled_amount`
- `V1` through `V28` (PCA components)

On submission, displays: fraud probability, anomaly score, risk score gauge, decision badge, and SHAP feature importance bar chart.

### Mode 2 — Paste CSV row

Paste a single comma-separated row of 30 values directly into a text box. Useful for quick testing with raw data from a spreadsheet.

### Mode 3 — Batch CSV prediction

Upload a `.csv` file with multiple transactions. The app:
1. Sends each row to `/predict` sequentially
2. Aggregates results into a summary table
3. Provides a downloadable CSV with columns: `fraud_probability`, `risk_score`, `risk_level`, `decision`
4. Shows aggregate stats: total transactions, fraud detected, high-risk count

---

## Project Structure

```
Credit_Card_Fraud_Detection/
│
├── app.py                        # Streamlit frontend — UI and API calls
├── main.py                       # FastAPI backend — inference and endpoints
│
├── fraud_model_hybrid.pkl        # Trained stacking ensemble
├── isolation_forest.pkl          # Trained Isolation Forest
├── fraud_config_hybrid.pkl       # Optimal thresholds, scaler params, feature order
│
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container spec (FastAPI + Streamlit in one container)
├── .dockerignore
└── README.md
```

> Training code (`train.py` / Jupyter notebooks) is kept strictly outside this directory. The three `.pkl` files are the only interface between training and production. This prevents training-time dependencies (e.g. training data, notebook state) from entering the Docker image.

---

## Local Setup

### Prerequisites

- Python 3.10+
- pip

### Install

```bash
git clone https://github.com/deepak-kumar/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

### Run backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

### Run frontend (separate terminal)

```bash
streamlit run app.py --server.port 8502
```

Open `http://localhost:8502`.

---

## Docker Deployment

### Build

```bash
docker build -t creditcardfrauddetection .
```

### Run locally

```bash
docker run -p 8001:8001 -p 8502:8502 creditcardfrauddetection
```

Both services start inside the container:

| Service | URL |
|---|---|
| FastAPI docs | `http://localhost:8001/docs` |
| Streamlit dashboard | `http://localhost:8502` |

### Push to Docker Hub

```bash
docker tag creditcardfrauddetection username/creditcardfrauddetection:latest
docker push username/creditcardfrauddetection:latest
```

---

## AWS EC2 Deployment

### EC2 setup

1. Launch an EC2 instance (t2.medium or higher recommended)
2. Open inbound ports `8001` and `8502` in the security group
3. SSH into the instance and install Docker:

```bash
sudo apt update && sudo apt install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

### Deploy

```bash
docker pull username/creditcardfrauddetection:latest

docker run -d \
  -p 8001:8001 \
  -p 8502:8502 \
  --restart unless-stopped \
  username/creditcardfrauddetection:latest
```

### Access

| Service | URL |
|---|---|
| Streamlit | `http://EC2_PUBLIC_IP:8502` |
| FastAPI docs | `http://EC2_PUBLIC_IP:8001/docs` |

> Use `--restart unless-stopped` to keep the container running after reboots.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10 | Core runtime |
| ML framework | scikit-learn | Preprocessing, LR, RF, stacking, metrics |
| Gradient boosting | XGBoost, LightGBM, CatBoost | High-performance base estimators |
| Anomaly detection | scikit-learn IsolationForest | Unsupervised anomaly score feature |
| Imbalance handling | imbalanced-learn (SMOTE) | Selective oversampling for LR |
| Explainability | SHAP | Per-prediction feature attribution |
| Backend | FastAPI + Uvicorn | REST API, async inference |
| Frontend | Streamlit | Interactive dashboard |
| Serialization | joblib | Model artifact storage and loading |
| Containerization | Docker | Reproducible single-container deployment |
| Registry | Docker Hub | Image distribution |
| Cloud | AWS EC2 | Production hosting |

---

## Roadmap

- [ ] PostgreSQL integration for transaction logging and audit trail
- [ ] User authentication (JWT-based) for the Streamlit dashboard
- [ ] Real-time monitoring dashboard (Grafana + Prometheus)
- [ ] Apache Kafka integration for streaming transaction inference
- [ ] CI/CD pipeline (GitHub Actions → Docker Hub → EC2)
- [ ] Kubernetes deployment with horizontal pod autoscaling
- [ ] Model drift detection and automated retraining triggers
- [ ] A/B testing framework for model version comparison

---

## Author

**Deepak Kumar** — AI/ML Developer

B.Tech CSE (AI), SKIT Jaipur · CGPA 9.15 · Expected 2027

`Machine Learning` `FastAPI` `Streamlit` `Docker` `AWS` `Explainable AI` `MLOps`

---

*Built with scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, FastAPI, Streamlit, Docker, and AWS EC2.*
