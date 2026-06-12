# ==========================================================
# Fraud Detection FastAPI Backend
# Hybrid ML + Isolation Forest + SHAP + Risk Engine
# ==========================================================

from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import pandas as pd
import joblib
import shap


# ----------------------------------------------------------
# FastAPI App
# ----------------------------------------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Hybrid ML Fraud Detection Backend",
    version="1.0"
)


# ----------------------------------------------------------
# Load Models
# ----------------------------------------------------------

model = joblib.load(
    "fraud_model_hybrid.pkl"
)

isf = joblib.load(
    "isolation_forest.pkl"
)

config = joblib.load(
    "fraud_config_hybrid.pkl"
)


# SHAP Explainer

try:
    explainer = shap.TreeExplainer(
        model
    )

except:

    explainer = None



FEATURE_NAMES = (
    ["scaled_amount",
     "scaled_time"]
    +
    [f"V{i}" for i in range(1,29)]
    +
    ["anomaly_score"]
)


# ----------------------------------------------------------
# Input Schema
# ----------------------------------------------------------

class Transaction(BaseModel):

    features: list[float]



# ----------------------------------------------------------
# Risk Engine
# ----------------------------------------------------------

def calculate_risk_score(
        fraud_probability,
        anomaly_score
):

    risk = (
        fraud_probability * 70
        +
        anomaly_score * 30
    )

    return round(
        float(risk),
        2
    )



def risk_decision(score):


    if score < 40:

        return (
            "LOW RISK",
            "APPROVE TRANSACTION"
        )


    elif score < 70:

        return (
            "MEDIUM RISK",
            "OTP VERIFICATION REQUIRED"
        )


    else:

        return (
            "HIGH RISK",
            "BLOCK TRANSACTION"
        )



# ----------------------------------------------------------
# Home Route
# ----------------------------------------------------------

@app.get("/")
def home():

    return {
        "message":
        "Fraud Detection API Running"
    }



# ----------------------------------------------------------
# Prediction API
# ----------------------------------------------------------

@app.post("/predict")
def predict(
    transaction: Transaction
):


    # ----------------------------
    # Input
    # ----------------------------

    X = np.array(
        transaction.features
    ).reshape(
        1,-1
    )


    # ----------------------------
    # Isolation Forest
    # ----------------------------

    raw_score = (
        isf
        .decision_function(X)[0]
    )


    anomaly_score = max(0,float(-raw_score))



    # Add anomaly feature

    X_final = np.hstack(
        [
            X,
            [[anomaly_score]]
        ]
    )


    # ----------------------------
    # ML Prediction
    # ----------------------------


    fraud_probability = (
        model
        .predict_proba(
            X_final
        )[0][1]
    )



    threshold = config.get(
        "threshold",
        0.5
    )


    prediction = bool(
        fraud_probability
        >=
        threshold
    )


    # ----------------------------
    # Risk Engine
    # ----------------------------


    risk_score = calculate_risk_score(
        fraud_probability,
        anomaly_score
    )


    risk_level, action = risk_decision(
        risk_score
    )



    # ----------------------------
    # SHAP Explainability
    # ----------------------------

    shap_result = []


    if explainer:

        try:

            values = explainer(
                X_final
            )


            impacts = (
                values.values[0]
            )


            shap_df = pd.DataFrame(
                {
                    "feature":
                        FEATURE_NAMES,

                    "impact":
                        impacts
                }
            )


            shap_df["abs"] = (
                shap_df["impact"]
                .abs()
            )


            top = (
                shap_df
                .sort_values(
                    "abs",
                    ascending=False
                )
                .head(5)
            )


            shap_result = (
                top[
                    [
                    "feature",
                    "impact"
                    ]
                ]
                .to_dict(
                    "records"
                )
            )


        except:

            shap_result=[]



    # ----------------------------
    # Response
    # ----------------------------


    return {


        "fraud_probability":
            round(
                float(
                    fraud_probability
                ),
                4
            ),


        "is_fraud":
            prediction,


        "anomaly_score":
            round(
                float(
                    anomaly_score
                ),
                4
            ),


        "risk_score":
            risk_score,


        "risk_level":
            risk_level,


        "decision":
            action,


        "top_shap_features":
            shap_result
    }
# python -m uvicorn Main:app --reload --port 8001