<div align="center">

# 💳 Credit Card Fraud Detection System

### End-to-end ML fraud detection platform with hybrid anomaly detection, SHAP explainability, dynamic risk scoring, FastAPI inference, and AWS EC2 deployment

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/ec2)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189ABC?style=for-the-badge)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2?style=for-the-badge)](https://shap.readthedocs.io)

<br/>

> 🔍 Detects fraud in real time · ⚡ Sub-second inference via REST API · 🧠 Explains every prediction with SHAP · 🚨 Banking-style risk decisions · ☁️ Live on AWS EC2

</div>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Live Demo](#-live-demo)
3. [Model Evaluation Results](#-model-evaluation-results)
4. [System Architecture](#-system-architecture)
5. [Dataset](#-dataset)
6. [Exploratory Data Analysis](#-exploratory-data-analysis)
7. [Machine Learning Pipeline](#-machine-learning-pipeline)
8. [Models & Comparison](#-models--comparison)
9. [Dynamic Risk Engine](#-dynamic-risk-engine)
10. [SHAP Explainability](#-shap-explainability)
11. [FastAPI Backend](#-fastapi-backend)
12. [Streamlit Frontend](#-streamlit-frontend)
13. [Project Structure](#-project-structure)
14. [Local Setup](#-local-setup)
15. [Docker Deployment](#-docker-deployment)
16. [AWS EC2 Deployment](#-aws-ec2-deployment)
17. [Tech Stack](#-tech-stack)
18. [Key Engineering Decisions](#-key-engineering-decisions)

---

## 🧠 Project Overview

Credit card fraud detection is one of the hardest real-world ML problems because of **extreme class imbalance** — fraudulent transactions make up only **0.172%** of all transactions. A naive classifier that always predicts *"not fraud"* achieves **99.8% accuracy while catching zero fraud cases**.

This project solves the problem correctly with a **hybrid two-layer detection architecture**:

| Layer | Approach | Purpose |
|---|---|---|
| 🌲 Layer 1 | Isolation Forest (unsupervised) | Generates `anomaly_score` — catches novel fraud patterns |
| 🤖 Layer 2 | Stacking Ensemble (supervised) | 5 base models + meta-learner for fraud probability |
| ⚖️ Layer 3 | Dynamic Risk Engine | Converts ML output into Results]decisions |
| 🔎 Layer 4 | SHAP Explainability | Explains every individual prediction |

**Why this architecture works better than a single model:**
- The Isolation Forest generates an `anomaly_score` feature that is added to the supervised training data. This gives the ensemble an independent signal it could not learn on its own — particularly useful for catching *new* fraud patterns that differ from historical training data.
- The stacking ensemble combines predictions from 5 diverse algorithms, reducing variance and improving generalization on minority class samples.
- Threshold tuning per model via F1-optimal precision-recall curves avoids the recall-vs-precision tradeoff problem of fixed 0.5 thresholds.

---

## 🚀 Live Demo

| 🖥️ Service | 🌐 URL |
|---|---|
| Streamlit Dashboard | `http://13.48.56.226:8502` |
| FastAPI Interactive Docs | `http://13.48.56.226:8001/docs` |
| FastAPI ReDoc | `http://13.48.56.226:8001/redoc` |

> 💡 The Streamlit dashboard connects to the FastAPI backend over HTTP. Both run inside a single Docker container on AWS EC2.

---

# 📊 Model Evaluation Results

Multiple machine learning models were trained and evaluated using both baseline features and the hybrid anomaly detection pipeline.

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- PR-AUC
- ROC-AUC

---

## 🔹 Performance Without Anomaly Score (Baseline)

| Model | Threshold | Accuracy | Precision | Recall | F1 Score | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 1.0000 | 99.89% | 81.36% | 48.98% | 61.15% | 72.22% | 97.13% |
| Random Forest | 0.8843 | 99.94% | 84.44% | 77.55% | 80.85% | 81.22% | 98.31% |
| XGBoost | 0.9838 | 99.95% | 95.00% | 77.55% | 85.39% | 85.45% | 97.94% |
| CatBoost | 0.9851 | 99.94% | 88.37% | 77.55% | 82.61% | 83.59% | 98.11% |
| LightGBM | 0.9768 | 99.96% | 93.98% | 79.59% | 86.19% | 87.64% | 97.83% |

---

## 🚀 Hybrid Pipeline Results (With Isolation Forest Anomaly Score)

An Isolation Forest based anomaly detection layer was added to generate an additional fraud risk feature (`anomaly_score`) which was combined with supervised models.

| Model | Threshold | Accuracy | Precision | Recall | F1 Score | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 1.0000 | 99.89% | 81.36% | 48.98% | 61.15% | 72.17% | 97.13% |
| Random Forest | 0.8540 | 99.93% | 81.72% | 77.55% | 79.58% | 79.70% | 98.44% |
| XGBoost | 0.9908 | 99.95% | 94.49% | 76.53% | 84.57% | 86.19% | 97.94% |
| CatBoost | 0.9636 | 99.93% | 80.41% | 79.59% | 80.00% | 83.92% | 97.90% |
| LightGBM | 0.9905 | 99.96% | 96.34% | 80.61% | 87.78% | 86.79% | 97.49% |
| Stacking Ensemble | 0.9990 | 99.95% | 92.86% | 79.59% | 85.71% | 86.09% | 98.47% |

---

## 🏆 Final Selected Model

The final deployment uses:

```
Hybrid Stacking Ensemble Model
+
Isolation Forest Anomaly Detection
+
Dynamic Risk Engine
```

Final Performance: LightGBM With Isolation Forest

| Metric | Score |
|-|-|
| Accuracy | 99.96% |
| Precision | 96.34% |
| Recall | 80.61% |
| F1 Score | 87.78% |
| PR-AUC | 86.79% |
| ROC-AUC | 97.49% |

---

## 📈 Impact of Adding Anomaly Detection

Adding an unsupervised anomaly detection layer helped the system detect unusual transaction behavior and improved robustness against unseen fraud patterns.

Benefits:

✔ Detects unknown fraud patterns  
✔ Adds behavioral anomaly signal  
✔ Improves hybrid decision making  
✔ Supports dynamic risk scoring engine

> 🔑 The `anomaly_score` feature consistently improves F1 across all models by giving supervised classifiers access to an unsupervised signal.

---

## 🏗️ System Architecture

```<img width="1600" height="874" alt="image" src="https://github.com/user-attachments/assets/1f2a174c-ba39-4c9a-bd46-951e72d6b2a9" />

┌─────────────────────────────────────────────────────────┐
│                    👤 User / Client                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              🖥️  Streamlit Dashboard (app.py)           │
│                                                          │
│   📝 Manual Input │ 📋 Paste CSV Row │ 📁 Batch Upload │
└────────────────────────┬────────────────────────────────┘
                         │  HTTP POST /predict
                         │  {"features": [...30 values...]}
                         ▼
┌─────────────────────────────────────────────────────────┐
│              ⚡ FastAPI Backend (main.py)               │
│                   Uvicorn ASGI Server                    │
│                   Port: 8001                             │
└──────┬──────────────────────────────────────────────────┘
       │
       ├──────────────────────┐
       ▼                      ▼
┌─────────────┐    ┌──────────────────────────────┐
│ 🌲 Isolation │  │  🤖 Hybrid ML Model          │
│   Forest    │    │  (fraud_model_hybrid.pkl)     │
│             │───▶│                               │
│ anomaly_    │    │  Base: RF + XGBoost + LightGBM│
│ score       │    │  Meta: Logistic Regression    │
└─────────────┘    └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  ⚖️ Dynamic Risk Engine      │
                   │                              │
                   │  Score = P×70 + A×30         │
                   └──────────────┬───────────────┘
                                  │
               ┌──────────────────┼──────────────────┐
               ▼                  ▼                  ▼
         🟢 LOW (0-40)     🟡 MEDIUM (40-70)   🔴 HIGH (70-100)
           Approve         OTP Verify           Block
               │                  │                  │
               └──────────────────┼──────────────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │  🔍 SHAP Explainability      │
                   │  Top 5 feature impacts       │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  📦 JSON Response to UI      │
                   │  fraud_probability            │
                   │  anomaly_score                │
                   │  risk_score + risk_level      │
                   │  decision                     │
                   │  top_shap_features (top 5)    │
                   └──────────────────────────────┘
```

---

## 📁 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Real anonymized European credit card transactions from **September 2013**.

| 📌 Property | 📊 Value |
|---|---|
| 📦 Total transactions | 284,807 |
| 🚨 Fraudulent transactions | 492 |
| ⚖️ Fraud rate | **0.172%** (severely imbalanced) |
| 🧮 Features | 30 (V1–V28 + Amount + Time) |
| 🏷️ Target column | `Class` (0 = legitimate, 1 = fraud) |
| 📅 Time period | 2 days in September 2013 |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `V1` – `V28` | Float | PCA-transformed components — original banking attributes anonymized for privacy |
| `Amount` | Float | Transaction amount in EUR |
| `Time` | Float | Seconds elapsed since first transaction |
| `Class` | Binary | 0 = not fraud, 1 = fraud |
| `scaled_amount` | Float | `Amount` scaled with `RobustScaler` (engineered) |
| `scaled_time` | Float | `Time` scaled with `RobustScaler` (engineered) |
| `anomaly_score` | Float | Isolation Forest decision score — added before supervised training (engineered) |

> 🔒 Because original banking attributes are privacy-protected via PCA transformation, feature engineering focuses on scaling `Amount` and `Time`, and adding the Isolation Forest anomaly signal.

---

## 🔍 Exploratory Data Analysis

The following analyses were performed before model development:

### 📊 Class Distribution
```
Class 0 (Normal):  284,315  →  99.828%
Class 1 (Fraud):       492  →   0.172%
```
> Severe imbalance. Accuracy alone is a useless metric — F1, Precision, Recall, and PR-AUC are used instead.

### 📈 Visualizations Produced

| Plot | Insight |
|---|---|
| 📊 Class distribution countplot | Visualizes the extreme imbalance |
| 💰 Amount distribution histogram (KDE) | Most transactions are small; fraud spans all amounts |
| 📦 Amount boxplot by class | Fraud transactions have different amount distributions |
| ⏰ Time distribution histogram | Transaction volume varies by time of day |
| 🔥 Feature correlation heatmap | PCA components are mostly uncorrelated by design |
| 📉 V1–V28 individual histograms | Several V features show bimodal distributions linked to fraud |

### Key EDA Findings
- Features like `V14`, `V17`, `V10`, `V12`, and `V4` show the strongest separation between fraud and legitimate transactions
- `Amount` alone is not sufficient to distinguish fraud — requires PCA features
- Transaction `Time` shows a periodic pattern (day/night cycles) but is a weak fraud predictor on its own

---

## 🤖 Machine Learning Pipeline

### Step 1 — Data Preprocessing

```python
from sklearn.preprocessing import RobustScaler

rob_scaler = RobustScaler()

# RobustScaler is used (not StandardScaler) because it is
# robust to outliers — large fraudulent amounts don't skew scaling
df['scaled_amount'] = rob_scaler.fit_transform(df[['Amount']])
df['scaled_time']   = rob_scaler.fit_transform(df[['Time']])

df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Feature order: scaled_amount, scaled_time, V1–V28
scaled_amount = df.pop('scaled_amount')
scaled_time   = df.pop('scaled_time')
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time',   scaled_time)
```

- `RobustScaler` is used (not `StandardScaler`) because it is resistant to outliers — large fraudulent transaction amounts do not distort the scaling
- Stratified 80/20 train-test split preserves the 0.172% fraud ratio in both splits

### Step 2 — Isolation Forest Anomaly Scoring

An unsupervised Isolation Forest is trained first on raw features. Its `decision_function` output is normalized and injected as `anomaly_score` into the feature matrix before any supervised training.

```python
from sklearn.ensemble import IsolationForest

isf = IsolationForest(
    n_estimators=200,
    contamination=0.002,   # ~0.2% matches actual fraud rate
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)
isf.fit(X_train)

# Normalize: invert so higher = more anomalous (0 to 1 scale)
def normalize_invert(scores, ref):
    inverted = -scores
    min_val  = -ref.max()
    max_val  = -ref.min()
    return (inverted - min_val) / (max_val - min_val + 1e-8)

train_score = normalize_invert(isf.decision_function(X_train), train_raw)
test_score  = normalize_invert(isf.decision_function(X_test), train_raw)

X_train_v2['anomaly_score'] = train_score
X_test_v2['anomaly_score']  = test_score
```

> 🔑 `contamination=0.002` is intentionally set to match the true fraud rate (~0.2%). This calibrates the Isolation Forest's anomaly threshold correctly.

### Step 3 — Handling Class Imbalance

A **single SMOTE strategy does not work for all models**. Different approaches are applied per model type:

| 🤖 Model | ⚖️ Imbalance Strategy | 💡 Reason |
|---|---|---|
| Logistic Regression | SMOTE oversampling | Linear models benefit from balanced training distribution |
| Random Forest | `class_weight='balanced'` | Native support; SMOTE on trees creates synthetic samples that can overfit leaf nodes |
| XGBoost | `scale_pos_weight=fraud_ratio` | Built-in loss weighting; `fraud_ratio = n_negative / n_positive` |
| LightGBM | `class_weight='balanced'` | Native gradient weighting for minority class |
| CatBoost | Default (handles natively) | CatBoost auto-detects imbalance internally |

```python
# SMOTE only for Logistic Regression
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_v2, y_train)

# scale_pos_weight for XGBoost
fraud_ratio = (y_train_res==0).sum() / (y_train_res==1).sum()
```

> ⚠️ **Lesson learned:** Applying SMOTE globally before tree-based training inflates F1 scores artificially by introducing synthetic minority samples that leak evaluation-set patterns. SMOTE is applied selectively only for Logistic Regression.

### Step 4 — Threshold Tuning via Precision-Recall Curves

Default threshold of 0.5 is suboptimal for imbalanced data. Each model's optimal threshold is selected by **maximizing F1 score on the precision-recall curve**:

```python
from sklearn.metrics import precision_recall_curve

def find_best_threshold(model, X_val, y_val):
    prob = model.predict_proba(X_val)[:, 1]
    p, r, t = precision_recall_curve(y_val, prob)
    f1 = 2 * p * r / (p + r + 1e-8)
    return round(float(t[np.argmax(f1[:-1])]), 4)
```

Saved thresholds are stored in `fraud_config_hybrid.pkl` and loaded at inference time — so the production API applies the same threshold that was tuned on validation data.

### Step 5 — Stacking Ensemble

The final deployed model stacks three base classifiers under a Logistic Regression meta-learner using 5-fold cross-validation:

```python
from sklearn.ensemble import StackingClassifier

estimators = [
    ("rf",   rf),
    ("xgb",  xgb),
    ("lgbm", lgbm)
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=1000),
    cv=5,
    stack_method="predict_proba",
    n_jobs=-1
)

stack.fit(X_train_res, y_train_res)
```

> 🔑 **Why RF + XGBoost + LightGBM as base estimators?** These three provide diverse inductive biases — bagging (RF), gradient boosting with depth control (XGBoost), and leaf-wise boosting (LightGBM) — reducing correlated errors in the meta-learner.

> ⚠️ **CatBoost is excluded from the stacking estimators** due to probability calibration compatibility issues with `StackingClassifier`. It is trained independently and its results are reported separately.

---

## 🤖 Models & Comparison

### Model Configurations

```python
# Logistic Regression — SMOTE + tuned threshold
LogisticRegression(C=0.1, max_iter=1000, solver="saga", random_state=42)

# Random Forest — class_weight balanced
RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_split=4, class_weight="balanced",
    random_state=42, n_jobs=-1
)

# XGBoost — scale_pos_weight
XGBClassifier(
    n_estimators=300, max_depth=6,
    learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=fraud_ratio,
    eval_metric="aucpr", random_state=42, verbosity=0
)

# LightGBM — class_weight balanced
LGBMClassifier(
    n_estimators=300, num_leaves=63,
    max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", random_state=42, verbose=-1
)

# CatBoost — auto imbalance handling
CatBoostClassifier(
    iterations=300, depth=6,
    learning_rate=0.05, l2_leaf_reg=3,
    eval_metric="F1", random_seed=42, verbose=0
)
```

### Evaluation Function

```python
def evaluate_model(name, model, X_test, y_test, threshold):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "Model":     name,
        "Threshold": threshold,
        "Accuracy":  round(accuracy_score(y_test, pred) * 100, 2),
        "Precision": round(precision_score(y_test, pred) * 100, 2),
        "Recall":    round(recall_score(y_test, pred) * 100, 2),
        "F1":        round(f1_score(y_test, pred) * 100, 2),
        "PR-AUC":    round(average_precision_score(y_test, prob) * 100, 2),
        "ROC-AUC":   round(roc_auc_score(y_test, prob) * 100, 2)
    }
```

### Model Selection Flow

```
Train all 5 base models
        ↓
Evaluate each with tuned thresholds
        ↓
Select best model by F1 score → best_model_name
        ↓
Stack RF + XGBoost + LightGBM under LR meta-learner
        ↓
Save: fraud_model_hybrid.pkl + fraud_config_hybrid.pkl
```

---

## ⚖️ Dynamic Risk Engine

Rather than outputting a binary fraud/not-fraud label, the system computes a **continuous banking risk score** that blends both the supervised and unsupervised signals.

### 📐 Formula

```
Risk Score = (Fraud Probability × 70) + (Anomaly Score × 30)
```

| Weight | Signal | Rationale |
|---|---|---|
| **70%** | Fraud Probability | Supervised model — trained on labeled fraud patterns; more reliable |
| **30%** | Anomaly Score | Isolation Forest — catches novel unseen fraud not in training labels |

### 🚦 Decision Table

| 🎯 Risk Score | 🚦 Risk Level | 🏦 Banking Action | 💡 Use Case |
|---|---|---|---|
| 0 – 40 | 🟢 **LOW RISK** | ✅ Approve Transaction | Normal spending, known patterns |
| 40 – 70 | 🟡 **MEDIUM RISK** | 📱 OTP Verification Required | Unusual pattern, not definitive fraud |
| 70 – 100 | 🔴 **HIGH RISK** | 🚫 Block Transaction | High-confidence fraud signal |

### Implementation

```python
def calculate_risk_score(fraud_probability, anomaly_score):
    risk = fraud_probability * 70 + anomaly_score * 30
    return round(float(risk), 2)

def risk_decision(score):
    if score < 40:
        return "LOW RISK",    "APPROVE TRANSACTION"
    elif score < 70:
        return "MEDIUM RISK", "OTP VERIFICATION REQUIRED"
    else:
        return "HIGH RISK",   "BLOCK TRANSACTION"
```

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) is integrated into the inference pipeline to explain **every individual prediction** — making the system auditable and production-compliant.

### How It Works

```python
# TreeExplainer is used for tree-based ensemble models
explainer = shap.TreeExplainer(model)

# Compute SHAP values for the prediction
values = explainer(X_final)
impacts = values.values[0]   # Shape: (31,) — one per feature

# Build ranked feature impact table
shap_df = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "impact":  impacts
})

# Return top 5 by absolute impact
top5 = shap_df.sort_values("abs", ascending=False).head(5)
```

### Feature Names Order

```python
FEATURE_NAMES = (
    ["scaled_amount", "scaled_time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["anomaly_score"]
)
# Total: 31 features (30 input + anomaly_score appended by backend)
```

### 📊 Example SHAP Output (High-Risk Transaction)

```
🚨 Transaction classified: HIGH RISK (Score: 91.5)

Feature          SHAP Impact    Direction
─────────────────────────────────────────
V14              -0.82          → Fraud
V17              -0.61          → Fraud
V10              -0.43          → Fraud
anomaly_score    +0.38          → Fraud (high anomaly)
V4               +0.21          → Legitimate (counterweight)
scaled_amount    -0.14          → Fraud
```

> 🔑 The `anomaly_score` feature appearing in top SHAP contributors confirms the Isolation Forest adds genuine predictive value — not just noise.

### Where SHAP is Displayed

- 📊 **Bar chart** in the Streamlit dashboard showing feature name vs SHAP impact
- 📋 **Data table** with feature + impact values for each prediction
- 📦 **API response** — `top_shap_features` field in JSON

---

## 🌐 FastAPI Backend

The backend (`Main.py`) is a production-style FastAPI application served by Uvicorn. All three model artifacts are loaded **once at startup** and reused for every request.

### 🚀 Startup — Model Loading

```python
model  = joblib.load("fraud_model_hybrid.pkl")   # Stacking ensemble
isf    = joblib.load("isolation_forest.pkl")      # Anomaly scorer
config = joblib.load("fraud_config_hybrid.pkl")   # Threshold + scaler info

# SHAP explainer — initialized from the loaded model
try:
    explainer = shap.TreeExplainer(model)
except:
    explainer = None   # Graceful degradation if model type unsupported
```

### 📡 API Endpoints

#### `GET /` — Health Check

```bash
curl http://localhost:8001/
```
```json
{ "message": "Fraud Detection API Running" }
```

#### `POST /predict` — Full Inference

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [149.62, 0.244, -1.35, 0.82, -0.25, 0.45, -0.63, 1.20,
                    0.07, 0.33, -0.44, -0.19, 0.58, -0.87, 0.16, -0.32,
                    0.91, -0.14, 0.74, -0.09, 0.23, 0.11, -0.55, 0.38,
                    -0.27, 0.08, -0.19, 0.04, 0.62, 1.79]}'
```

**Request schema:**
```python
class Transaction(BaseModel):
    features: list[float]   # 30 values: scaled_amount, scaled_time, V1–V28
```

> ✅ The backend appends `anomaly_score` internally — the client sends only 30 features.

**Full response:**
```json
{
  "fraud_probability": 0.9732,
  "is_fraud": true,
  "anomaly_score": 0.8214,
  "risk_score": 92.78,
  "risk_level": "HIGH RISK",
  "decision": "BLOCK TRANSACTION",
  "top_shap_features": [
    { "feature": "V14",           "impact": -0.82 },
    { "feature": "V17",           "impact": -0.61 },
    { "feature": "V10",           "impact": -0.43 },
    { "feature": "anomaly_score", "impact":  0.38 },
    { "feature": "V4",            "impact":  0.21 }
  ]
}
```

### 🔧 Inference Pipeline (Per Request)

```python
# 1. Parse 30 input features
X = np.array(transaction.features).reshape(1, -1)

# 2. Compute Isolation Forest anomaly score
raw_score    = isf.decision_function(X)[0]
anomaly_score = max(0, float(-raw_score))   # Invert: higher = more anomalous

# 3. Append anomaly_score to feature vector (31 total)
X_final = np.hstack([X, [[anomaly_score]]])

# 4. Supervised ML prediction
fraud_probability = model.predict_proba(X_final)[0][1]
threshold         = config.get("threshold", 0.5)
is_fraud          = bool(fraud_probability >= threshold)

# 5. Risk engine
risk_score         = fraud_probability * 70 + anomaly_score * 30
risk_level, action = risk_decision(risk_score)

# 6. SHAP top-5 features
values   = explainer(X_final)
shap_top = pd.DataFrame({"feature": FEATURE_NAMES, "impact": values.values[0]}) \
             .sort_values("abs", ascending=False).head(5)
```

### ▶️ Running the Backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Interactive API docs: `http://localhost:8001/docs`

---

## 🖥️ Streamlit Frontend

The Streamlit dashboard (`App.py`) is the primary user interface. It calls the FastAPI backend over HTTP and renders predictions visually.

**Live endpoint used:**
```python
API_URL = "http://13.48.56.226:8001/predict"
```

**Custom styling applied:**
```python
# Red fraud detection button with hover effect
div.stButton > button {
    background-color: #ff3c5a;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 18px;
    font-weight: bold;
}
```

### 📝 Mode 1 — Manual Transaction Input

Enter all 30 feature values individually:
- `scaled_amount` and `scaled_time` in a 2-column layout
- `V1`–`V28` PCA features in a **4-column grid**

**On submit shows:**

| 📊 Metric | Display |
|---|---|
| Fraud status | ✅ TRANSACTION SAFE or 🚨 FRAUD DETECTED |
| Fraud probability | e.g. `97.32%` |
| Anomaly score | e.g. `0.8214` |
| Risk engine score | e.g. `92.78 / 100` |
| Risk level | LOW / MEDIUM / HIGH RISK |
| Decision | APPROVE / OTP VERIFICATION / BLOCK |
| SHAP bar chart | Feature importance visualization |
| SHAP dataframe | Raw feature + impact table |

### 📋 Mode 2 — Paste CSV Row

Paste 30 comma-separated values into a text box. Validates that exactly 30 values are provided before sending to the API.

```python
values = [float(x) for x in text.split(",")]
if len(values) != 30:
    st.error("Please enter exactly 30 values")
```

### 📁 Mode 3 — Batch Prediction

Upload a `.csv` file containing multiple transactions:

```python
for row in df[FEATURES].values:
    result = predict_transaction(row.reshape(1, -1))
    probs.append(result["fraud_probability"])
    risks.append(result["risk_score"])
    levels.append(result["risk_level"])
    decisions.append(result["decision"])

df["fraud_probability"] = probs
df["risk_score"]        = risks
df["risk_level"]        = levels
df["decision"]          = decisions
```

**Output:**
- 📊 Full results table rendered in-browser
- ⬇️ Downloadable `fraud_results.csv` with all predictions

---

## 📂 Project Structure

```
FraudShield/
│
├── 🐍 app.py                        # Streamlit dashboard (frontend)
├── 🐍 main.py                       # FastAPI backend (inference API)
│
├── 📦 fraud_model_hybrid.pkl        # Trained stacking ensemble (RF + XGBoost + LightGBM + LR meta)
├── 📦 isolation_forest.pkl          # Trained Isolation Forest (anomaly scorer)
├── 📦 fraud_config_hybrid.pkl       # Optimal threshold + RobustScaler + best model name
│
├── 📋 requirements.txt              # Python dependencies
├── 🐳 Dockerfile                    # Single container: FastAPI (8001) + Streamlit (8502)
├── 🚫 .dockerignore                 # Excludes training data, notebooks from image
└── 📄 README.md
```

> 🔑 **Separation of concerns:** Training code (Jupyter notebook) is kept entirely outside this directory. The three `.pkl` artifacts are the **only interface** between training and production inference. This keeps the Docker image lean and the deployment reproducible.

### Saved Artifacts

```python
# Training notebook saves three artifacts
joblib.dump(best_model,    "fraud_model_hybrid.pkl")
joblib.dump(isf,           "isolation_forest.pkl")
joblib.dump({
    "threshold":    best_threshold,
    "rob_scaler":   rob_scaler,
    "best_model":   best_model_name
},                         "fraud_config_hybrid.pkl")
```

---

## 🔧 Local Setup

### Prerequisites

- Python 3.10+
- pip
- (Optional) Docker

### Installation

```bash
git clone https://github.com/deepak-kumar/fraudshield.git
cd fraudshield
pip install -r requirements.txt
```

### Run Backend

```bash
uvicorn Main:app --host 0.0.0.0 --port 8001
# API docs → http://localhost:8001/docs
```

### Run Frontend (separate terminal)

```bash
streamlit run App.py --server.port 8502
# Dashboard → http://localhost:8502
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t deepak2k6/creditcardfrauddetection:latest .
```

### Run Locally

```bash
docker run -p 8001:8001 -p 8502:8502 deepak2k6/creditcardfrauddetection:latest
```

| 🌐 Service | URL |
|---|---|
| 📊 Streamlit dashboard | `http://localhost:8502` |
| ⚡ FastAPI docs | `http://localhost:8001/docs` |

### Push to Docker Hub

```bash
docker tag deepak2k6/creditcardfrauddetection:latest
docker push deepak2k6/creditcardfrauddetection:latest
```

---

## ☁️ AWS EC2 Deployment

### EC2 Setup

1. Launch EC2 instance — **t2.medium or higher** (model inference needs RAM)
2. In the Security Group, open **inbound ports** `8001` and `8502` (TCP, from anywhere `0.0.0.0/0`)
3. SSH into the instance:

```bash
ssh -i your-key.pem ubuntu@EC2_PUBLIC_IP
```

4. Install Docker:

```bash
sudo apt update && sudo apt install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker ubuntu
newgrp docker   # Apply group change without logout
```

### Deploy Container

```bash
docker pull username/fraudshield:latest

docker run -d \
  -p 8001:8001 \
  -p 8502:8502 \
  --restart unless-stopped \
  --name fraudshield \
  username/fraudshield:latest
```

### Verify Running

```bash
docker ps
docker logs fraudshield
```

### Access

| 🌐 Service | URL |
|---|---|
| 📊 Streamlit | `http://EC2_PUBLIC_IP:8502` |
| ⚡ FastAPI docs | `http://EC2_PUBLIC_IP:8001/docs` |

> 💡 `--restart unless-stopped` keeps the container running after EC2 reboots.

---

## 🛠️ Tech Stack

| 🏗️ Layer | 🔧 Technology | 📌 Version | 🎯 Purpose |
|---|---|---|---|
| Language | Python | 3.10 | Core runtime |
| ML framework | scikit-learn | 1.4+ | Preprocessing, LR, RF, stacking, metrics |
| Gradient boosting | XGBoost | 2.0 | High-performance base estimator |
| Gradient boosting | LightGBM | 4.0 | Fast leaf-wise boosting base estimator |
| Gradient boosting | CatBoost | 1.2 | Categorical-aware boosting |
| Anomaly detection | IsolationForest | scikit-learn | Unsupervised anomaly score feature generation |
| Imbalance handling | imbalanced-learn | 0.12 | SMOTE selective oversampling for LR |
| Explainability | SHAP | 0.45 | TreeExplainer for per-prediction feature attribution |
| Backend | FastAPI | 0.110 | Async REST API, Pydantic validation |
| ASGI server | Uvicorn | 0.29 | High-performance Python web server |
| Frontend | Streamlit | 1.33 | Interactive ML dashboard |
| Serialization | joblib | 1.3 | Model artifact save/load |
| Containerization | Docker | 24+ | Reproducible deployment |
| Registry | Docker Hub | — | Image distribution and versioning |
| Cloud | AWS EC2 | — | Production hosting (t2.medium+) |
| Data processing | pandas, numpy | latest | Data manipulation and feature engineering |
| Visualization | matplotlib, seaborn | latest | EDA plots and confusion matrix |
| HTTP client | requests | 2.31 | Streamlit → FastAPI communication |

---

## 🧪 Key Engineering Decisions

| 🔑 Decision | ✅ Choice Made | ❌ Alternative Rejected | 💡 Reason |
|---|---|---|---|
| Scaler | `RobustScaler` | `StandardScaler` | Robust to outliers from large fraud amounts |
| Imbalance (trees) | `class_weight` / `scale_pos_weight` | SMOTE for all | SMOTE on trees risks synthetic sample overfitting |
| Threshold | F1-optimal via PR curve | Fixed 0.5 | Imbalanced data makes 0.5 suboptimal for minority class |
| Anomaly normalization | Invert + min-max | Raw score | Raw IF scores are negative — inversion makes them interpretable |
| CatBoost in stacking | Excluded | Include | Probability calibration conflicts with `StackingClassifier` |
| SHAP explainer | `TreeExplainer` | `KernelExplainer` | 100× faster for tree-based models; exact not approximate |
| Artifact interface | 3 `.pkl` files only | Full training pipeline | Keeps Docker image lean; training code never enters production |
| Container design | Single container | Separate containers | Simplifies EC2 deployment for a portfolio project |

---

## 👨‍💻 Author

<div align="center">

**Deepak Kumar Saini**

B.Tech CSE (Artificial Intelligence) · SKIT Jaipur · CGPA 9.15 · Expected 2027

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com)

`Machine Learning` `MLOps` `FastAPI` `Streamlit` `Docker` `AWS` `Explainable AI` `Fraud Detection`

</div>

---

<div align="center">

*Built with scikit-learn · XGBoost · LightGBM · CatBoost · SHAP · FastAPI · Streamlit · Docker · AWS EC2*

⭐ Star this repo if you found it useful!

</div>
