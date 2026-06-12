# Credit Card Fraud Detection System

> End-to-end fraud detection combining supervised ML, unsupervised anomaly detection, explainable AI, and a dynamic risk engine — served via FastAPI and deployed on AWS EC2.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-backend-teal) ![Streamlit](https://img.shields.io/badge/Streamlit-frontend-green) ![Docker](https://img.shields.io/badge/Docker-containerized-gray) ![AWS](https://img.shields.io/badge/AWS-EC2-orange)

---

## Live Deployment

| Service | URL |
|---|---|
| Streamlit UI | `http://YOUR_PUBLIC_IP:8502` |
| FastAPI Docs | `http://YOUR_PUBLIC_IP:8001/docs` |

---

## Target Performance

| Metric | Target |
|---|---|
| Precision | > 93% |
| Recall | > 94% |
| F1 Score | > 92% |
| Explainability | SHAP (per-prediction) |

---

## Machine Learning Pipeline

Transactions are processed through a hybrid pipeline — an Isolation Forest adds an anomaly score feature, then six supervised models vote on the final fraud probability.

```
Raw features (V1–V28 + Amount + Time)
        ↓
Isolation Forest  →  anomaly_score feature
        ↓
SMOTE + feature scaling  →  class balance
        ↓
Stacking Ensemble  →  fraud probability
        ↓
Dynamic Risk Engine  →  decision
```

### Handling class imbalance

- **SMOTE** oversampling applied selectively for Logistic Regression
- **class_weight / scale_pos_weight** used for tree-based models (Random Forest, XGBoost, LightGBM)
- **CatBoost** uses `auto_class_weights` with native float casting

### Threshold tuning

Thresholds are tuned via precision-recall curves targeting F1-optimal cutoffs rather than recall-only strategies, to balance false positives and false negatives.

---

## Models

| Model | Imbalance strategy |
|---|---|
| Logistic Regression | SMOTE + threshold tuning |
| Random Forest | class_weight balanced |
| XGBoost | scale_pos_weight |
| LightGBM | is_unbalance |
| CatBoost | auto_class_weights |
| **Stacking Ensemble** | **Final hybrid model** |

---

## Dynamic Risk Engine

ML fraud probability and the Isolation Forest anomaly score are blended into a single banking risk score:

```
Risk Score = (Fraud Probability × 70) + (Anomaly Score × 30)
```

| Score | Level | Action |
|---|---|---|
| 0 – 40 | 🟢 Low | Approve transaction |
| 40 – 70 | 🟡 Medium | OTP verification |
| 70 – 100 | 🔴 High | Block transaction |

---

## SHAP Explainability

SHAP is integrated to explain individual predictions. For each transaction, the model surfaces the top contributing features — e.g. `V14`, `V17`, `V10`, `anomaly_score` — so decisions are auditable and transparent.

---

## FastAPI Backend

### `GET /`

Health check.

```json
{ "message": "Fraud Detection API Running" }
```

### `POST /predict`

**Input:** 30 transaction features as a JSON array.

```json
{ "features": [0.1, -0.5, ...] }
```

**Output:**

```json
{
  "fraud_probability": 0.97,
  "is_fraud": true,
  "anomaly_score": 0.82,
  "risk_score": 91,
  "risk_level": "HIGH RISK",
  "decision": "BLOCK TRANSACTION"
}
```

---

## Streamlit Frontend

Three input modes are supported:

1. **Manual input** — enter `scaled_amount`, `scaled_time`, and V1–V28 PCA features directly
2. **Paste CSV row** — paste transaction values as a single comma-separated row
3. **Batch prediction** — upload a CSV, get fraud probability, risk score, and decision for every row; download results

---

## Project Structure

```
Credit_Card_Fraud_Detection/
├── app.py                      # Streamlit frontend
├── main.py                     # FastAPI backend
├── fraud_model_hybrid.pkl      # Stacking ensemble
├── isolation_forest.pkl        # Anomaly scorer
├── fraud_config_hybrid.pkl     # Thresholds + config
├── Dockerfile
├── requirements.txt
└── README.md
```

Training code is kept strictly separate from the frontend. The three `.pkl` artifacts are the only shared interface between training and inference.

---

## Docker Deployment

```bash
# Build
docker build -t creditcardfrauddetection .

# Run locally
docker run -p 8001:8001 -p 8502:8502 creditcardfrauddetection
```

Both services run inside a single container:
- FastAPI: `http://localhost:8001/docs`
- Streamlit: `http://localhost:8502`

---

## AWS EC2 Deployment

```bash
docker pull username/creditcardfrauddetection:latest

docker run \
  -p 8001:8001 \
  -p 8502:8502 \
  username/creditcardfrauddetection:latest
```

Then open `http://EC2_PUBLIC_IP:8502`.

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn |
| Explainability | SHAP |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Infrastructure | Docker, Docker Hub, AWS EC2 |

---

## Roadmap

- [ ] Database integration for transaction logging
- [ ] User authentication layer
- [ ] Real-time monitoring dashboard
- [ ] CI/CD pipeline
- [ ] Kubernetes deployment

---

## Author

**Deepak Kumar** — AI/ML Developer

`Machine Learning · FastAPI · Streamlit · Docker · AWS`
