
# =============================================================================
# FraudShield Frontend
# Streamlit + FastAPI Integration
# Manual + CSV Row + Batch Prediction
# =============================================================================
import os
import streamlit as st
import pandas as pd
import numpy as np
import requests


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")


##---------------------CSS---------------
st.markdown("""
<style>

/* Detect Fraud Button */
div.stButton > button {
    background-color: #ff3c5a;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 18px;
    font-weight: bold;
    border: none;
}


/* Hover Effect */
div.stButton > button:hover {
    background-color: #ff0033;
    color: white;
    border: 1px solid #ff0033;
}

</style>
""", unsafe_allow_html=True)


# ---------------- API ----------------

API_URL = "http://13.48.56.226:8001/predict"

FEATURES = (
    ["scaled_amount", "scaled_time"]
    +
    [f"V{i}" for i in range(1,29)]
)


# ---------------- API CALL ----------------

def predict_transaction(X):

    try:

        response = requests.post(
            API_URL,
            json={
                "features":
                X.flatten().tolist()
            }
        )


        if response.status_code == 200:

            return response.json()


        else:

            st.error(
                "FastAPI Prediction Error"
            )

            return None


    except Exception as e:

        st.error(
            f"Backend not connected: {e}"
        )

        return None



# ---------------- SHOW RESULT ----------------


def show_result(result):


    if result is None:

        return


    c1,c2,c3 = st.columns(3)


    with c1:

        if result["is_fraud"]:

            st.error(
                "FRAUD DETECTED"
            )

        else:

            st.success(
                "TRANSACTION SAFE"
            )


    with c2:

        st.metric(
            "Fraud Probability",
            f'{result["fraud_probability"]*100:.2f}%'
        )


        st.metric(
            "Anomaly Score",
            result["anomaly_score"]
        )



    with c3:


        st.metric(
            "Risk Engine Score",
            f'{result["risk_score"]}/100'
        )


        st.write(
            result["risk_level"]
        )


        st.write(
            result["decision"]
        )


    # ---------------- SHAP ----------------

    st.divider()

    st.subheader(
        "SHAP Explainability"
    )


    shap_data = result.get(
        "top_shap_features",
        []
    )


    if len(shap_data)>0:


        shap_df = pd.DataFrame(
            shap_data
        )


        st.bar_chart(
            shap_df.set_index(
                "feature"
            )["impact"]
        )


        st.dataframe(
            shap_df,
            use_container_width=True
        )


    else:

        st.info(
            "No SHAP values returned"
        )



# ---------------- INPUT METHOD ----------------


mode = st.radio(
    "Input method",
    [
        "Manual",
        "Paste CSV row",
        "Batch Prediction"
    ],
    horizontal=True
)



# ---------------- MANUAL ----------------


if mode=="Manual":


    values={}


    st.subheader(
        "Transaction Features"
    )


    # keep old layout size

    c1,c2 = st.columns(2)


    with c1:

        values["scaled_amount"] = st.number_input(
            "scaled_amount",
            value=0.0,
            format="%.6f"
        )


    with c2:

        values["scaled_time"] = st.number_input(
            "scaled_time",
            value=0.0,
            format="%.6f"
        )



    st.markdown(
        "### V1 - V28 (PCA Components)"
    )


    cols = st.columns(4)


    for i,f in enumerate(
        [f"V{x}" for x in range(1,29)]
    ):


        with cols[i%4]:


            values[f] = st.number_input(
                f,
                value=0.0,
                format="%.6f"
            )



    if st.button(
        "Detect Fraud",
        use_container_width=True
    ):


        X = np.array(
            [
                [
                    values[f]
                    for f in FEATURES
                ]
            ]
        )


        result = predict_transaction(
            X
        )


        show_result(
            result
        )



# ---------------- CSV ROW ----------------


elif mode=="Paste CSV row":


    text = st.text_area(
        "Paste 30 comma separated values"
    )


    if st.button(
        "🚀 Detect Fraud"
    ):


        try:


            values = [
                float(x)
                for x in text.split(",")
            ]


            if len(values)!=30:


                st.error(
                    "Please enter exactly 30 values"
                )


            else:


                X=np.array(
                    [values]
                )


                show_result(
                    predict_transaction(X)
                )



        except Exception as e:


            st.error(e)



# ---------------- BATCH ----------------


else:


    file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


    if file:


        df = pd.read_csv(
            file
        )


        st.dataframe(
            df.head()
        )


        if st.button(
            "Run Batch Prediction"
        ):


            probs=[]
            risks=[]
            levels=[]
            decisions=[]


            for row in df[FEATURES].values:


                result = predict_transaction(
                    row.reshape(1,-1)
                )


                if result:


                    probs.append(
                        result["fraud_probability"]
                    )

                    risks.append(
                        result["risk_score"]
                    )

                    levels.append(
                        result["risk_level"]
                    )

                    decisions.append(
                        result["decision"]
                    )


            df["fraud_probability"] = probs
            df["risk_score"] = risks
            df["risk_level"] = levels
            df["decision"] = decisions


            st.success(
                "Batch Prediction Completed"
            )


            st.dataframe(
                df,
                use_container_width=True
            )


            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "fraud_results.csv"
            )
# python -m streamlit run App.py --server.port 8502