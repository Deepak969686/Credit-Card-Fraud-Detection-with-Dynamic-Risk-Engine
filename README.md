# 💳 Credit Card Fraud Detection System

An end-to-end production-ready **Credit Card Fraud Detection System** using Machine Learning, FastAPI backend, Streamlit frontend, SHAP Explainable AI, Dynamic Risk Engine, Docker containerization, and AWS EC2 deployment.

The system detects fraudulent transactions in real time, explains predictions using SHAP, and provides banking-style risk decisions such as transaction approval, OTP verification, or blocking.

---

## 🚀 Live Deployment

### Streamlit Application

```
http://YOUR_PUBLIC_IP:8502
```

### FastAPI Documentation

```
http://YOUR_PUBLIC_IP:8001/docs
```

---

# 📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions are very rare compared to normal transactions.

This project combines:

- Supervised Machine Learning
- Unsupervised Anomaly Detection
- Explainable AI
- Risk Scoring System
- API-based ML Deployment


---

# 🏗️ System Architecture


```
                    User
                     |
                     |
                     ↓

          Streamlit Frontend
              (app.py)

     Manual Input | CSV Row | Batch CSV

                     |
                     |
              HTTP API Request

                     ↓

             FastAPI Backend
                (main.py)

                     |
        --------------------------------

        Isolation Forest
              |
              ↓
        Anomaly Score Feature

              +

        Hybrid ML Model

              |
              ↓

        Fraud Probability

              |
              ↓

        Dynamic Risk Engine

              |
              ↓

     LOW / MEDIUM / HIGH Risk

              |
              ↓

        SHAP Explainability

              |
              ↓

        JSON Response to UI

```

---

# 🧠 Machine Learning Pipeline


## Data Processing

- Dataset loading
- Data cleaning
- Missing value checking
- Feature scaling
- Train-test splitting


---

## Handling Class Imbalance

Fraud datasets are highly imbalanced.

Implemented:

- SMOTE oversampling technique

to improve fraud detection capability.

---

# 🤖 Models Used


Multiple ML algorithms were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Ensemble Learning


Final model:

```
Hybrid Ensemble Fraud Detection Model
```

---

# 🔍 Isolation Forest Anomaly Detection


An unsupervised anomaly detection layer was added.

Process:

```
Transaction Features

        ↓

Isolation Forest

        ↓

Generate anomaly_score

        ↓

Add as new ML feature
```

This improves fraud pattern detection.

---

# 📊 Model Evaluation

Models evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix


Threshold tuning was applied for better fraud classification.

---

# ⚡ Dynamic Risk Engine


The ML probability is converted into a banking risk score.

Formula:

```
Risk Score =
(Fraud Probability × 70)
+
(Anomaly Score × 30)
```


Decision:

| Risk Score | Level | Action |
|----------|-------|---------|
| 0-40 | LOW | Approve Transaction |
| 40-70 | MEDIUM | OTP Verification |
| 70-100 | HIGH | Block Transaction |

---

# 🔎 SHAP Explainable AI


SHAP is integrated to explain:

- Why transaction is fraud
- Feature contribution
- Model decision transparency


Example Output:

```
Top Important Features

V14
V17
V10
anomaly_score
```

---

# 🌐 FastAPI Backend


FastAPI provides ML inference API.

## Endpoints


### Health Check

```
GET /
```

Response:

```json
{
 "message":"Fraud Detection API Running"
}
```


---

### Prediction API

```
POST /predict
```


Input:

```json
{
 "features":[
    0.1,
    -0.5,
    ...
 ]
}
```


Output:


```json
{
 "fraud_probability":0.97,

 "is_fraud":true,

 "anomaly_score":0.82,

 "risk_score":91,

 "risk_level":"HIGH RISK",

 "decision":"BLOCK TRANSACTION"
}
```

---

# 🖥️ Streamlit Frontend Features


## 1. Manual Transaction Testing

Enter:

- scaled_amount
- scaled_time
- V1-V28 PCA features


---

## 2. Paste CSV Row


Paste transaction values directly.


---

## 3. Batch Prediction


Upload CSV file and get:

- Fraud probability
- Risk score
- Decision


Download results.

---

# 📂 Project Structure


```
Credit_Card_Fraud_Detection/


│
├── app.py
│
│   Streamlit Frontend
│

├── main.py
│
│   FastAPI Backend
│

├── fraud_model_hybrid.pkl

├── isolation_forest.pkl

├── fraud_config_hybrid.pkl


├── requirements.txt

├── Dockerfile

├── .dockerignore

└── README.md
```

---

# 🐳 Docker Deployment


## Build Docker Image


```bash
docker build -t deepak2k6/creditcardfrauddetection .
```


---

## Run Container


```bash
docker run -p 8001:8001 -p 8502:8502 creditcardfrauddetection
```


Services:


FastAPI:

```
http://localhost:8001/docs
```


Streamlit:


```
http://localhost:8502
```


---

# ☁️ AWS EC2 Deployment


Pull Docker image:


```bash
docker pull deepak2k6/creditcardfrauddetection:latest
```


Run:


```bash
docker run \
-p 8001:8001 \
-p 8502:8502 \
deepak2k6/creditcardfrauddetection:latest
```


Open:


```
http://EC2_PUBLIC_IP:8502
```


---

# 🛠️ Tech Stack


## Programming

- Python


## Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Isolation Forest
- SMOTE


## Explainable AI

- SHAP


## Backend

- FastAPI
- Uvicorn


## Frontend

- Streamlit


## Deployment

- Docker
- Docker Hub
- AWS EC2


---

# 📌 Key Features


✔ Real-time fraud prediction

✔ Hybrid ML architecture

✔ Anomaly detection

✔ Dynamic risk scoring

✔ Explainable AI

✔ REST API backend

✔ Interactive frontend

✔ Batch prediction support

✔ Dockerized deployment

✔ Cloud hosted


---

# Future Improvements


- Database integration
- User authentication
- Monitoring dashboard
- CI/CD pipeline
- Kubernetes deployment


---

# Author

**Deepak Kumar Saini**

AI/ML Developer

```
Machine Learning | FastAPI | Streamlit | Docker | AWS
```
