"""
Bitcoin Trading Analytics — deployed app for the AADM End Term Project
(Group 1, Finance, Dataset DS1).

Two tools, backed by models trained in train_and_save_models.py and shipped
as .pkl files (no retraining happens on the server):

1. Trade Profitability Predictor  -> Random Forest classifier
2. Investor Segment Explorer      -> K-Means clustering

A third tab shows the underlying EDA/model charts from the report, so the
whole analysis is visible from one live link rather than needing the Word
doc open side-by-side.
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Bitcoin Trading Analytics", layout="wide")


@st.cache_resource
def load_artifacts():
    rf_model = joblib.load(os.path.join(APP_DIR, "rf_model.pkl"))
    cluster_scaler = joblib.load(os.path.join(APP_DIR, "cluster_scaler.pkl"))
    kmeans_model = joblib.load(os.path.join(APP_DIR, "kmeans_model.pkl"))
    with open(os.path.join(APP_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return rf_model, cluster_scaler, kmeans_model, meta


rf_model, cluster_scaler, kmeans_model, meta = load_artifacts()

st.title("Bitcoin Trading Analytics")
st.caption(
    "AADM End Term Project — Group 1, Finance Specialization, Dataset DS1  |  "
    "Rishabh Rohra, Aryendra Singh, Chandrakant Sahu"
)

tab1, tab2, tab3 = st.tabs([
    "Trade Profitability Predictor",
    "Investor Segment Explorer",
    "Project Dashboard",
])

# ============================================================
# TAB 1 — Trade Profitability Predictor (Random Forest)
# ============================================================
with tab1:
    st.subheader("Predict whether a trade will be profitable")

    st.warning(
        f"**Read this before you trust the output.** This model was tested "
        f"against 600 held-out trades and scored {meta['rf_test_accuracy']*100:.1f}% "
        f"accuracy with an ROC-AUC of {meta['rf_test_roc_auc']:.2f} — a ROC-AUC of 0.50 "
        f"means pure guessing. Across four different modeling techniques "
        f"(Logistic Regression, Decision Tree, Random Forest, Linear Regression), none "
        f"found a meaningful relationship between these features and trade outcome. "
        f"This tool is deployed to demonstrate the full modeling pipeline end-to-end, "
        f"not as a trading signal. Do not use this to make real trading decisions."
    )

    col1, col2, col3 = st.columns(3)
    inputs = {}

    with col1:
        st.markdown("**Investor & Account**")
        inputs["Investor_Type"] = st.selectbox("Investor Type", meta["category_options"]["Investor_Type"])
        inputs["KYC_Status"] = st.selectbox("KYC Status", meta["category_options"]["KYC_Status"])
        inputs["Two_FA_Status"] = st.selectbox("Two-Factor Auth", meta["category_options"]["Two_FA_Status"])
        inputs["Device_Type"] = st.selectbox("Device Type", meta["category_options"]["Device_Type"])
        r = meta["numeric_ranges"]["Wallet_Age_Days"]
        inputs["Wallet_Age_Days"] = st.slider("Wallet Age (days)", int(r["min"]), int(r["max"]), int(r["median"]))
        r = meta["numeric_ranges"]["Wallet_Balance_BTC"]
        inputs["Wallet_Balance_BTC"] = st.number_input("Wallet Balance (BTC)", r["min"], r["max"], r["median"])

    with col2:
        st.markdown("**Trade Details**")
        r = meta["numeric_ranges"]["BTC_Price_USD"]
        inputs["BTC_Price_USD"] = st.number_input("BTC Price (USD)", r["min"], r["max"], r["median"])
        r = meta["numeric_ranges"]["BTC_Quantity"]
        inputs["BTC_Quantity"] = st.number_input("BTC Quantity", r["min"], r["max"], r["median"])
        r = meta["numeric_ranges"]["Transaction_Value_USD"]
        inputs["Transaction_Value_USD"] = st.number_input("Transaction Value (USD)", r["min"], r["max"], r["median"])
        r = meta["numeric_ranges"]["Transaction_Fee_USD"]
        inputs["Transaction_Fee_USD"] = st.number_input("Transaction Fee (USD)", r["min"], r["max"], r["median"])
        r = meta["numeric_ranges"]["Leverage_Ratio"]
        inputs["Leverage_Ratio"] = st.slider("Leverage Ratio", r["min"], r["max"], r["median"])

    with col3:
        st.markdown("**Market Conditions**")
        r = meta["numeric_ranges"]["Price_24h_Change_Percent"]
        inputs["Price_24h_Change_Percent"] = st.slider("24h Price Change (%)", r["min"], r["max"], r["median"])
        r = meta["numeric_ranges"]["Price_7d_Change_Percent"]
        inputs["Price_7d_Change_Percent"] = st.slider("7d Price Change (%)", r["min"], r["max"], r["median"])
        r = meta["numeric_ranges"]["Volatility_30d_Percent"]
        inputs["Volatility_30d_Percent"] = st.slider("30d Volatility (%)", r["min"], r["max"], r["median"])
        inputs["Market_Sentiment"] = st.selectbox("Market Sentiment", meta["category_options"]["Market_Sentiment"])
        r = meta["numeric_ranges"]["RSI_14"]
        inputs["RSI_14"] = st.slider("RSI (14-day)", r["min"], r["max"], r["median"])
        r = meta["numeric_ranges"]["Fear_Greed_Index"]
        inputs["Fear_Greed_Index"] = st.slider("Fear & Greed Index", r["min"], r["max"], r["median"])
        r = meta["numeric_ranges"]["Portfolio_Risk_Score"]
        inputs["Portfolio_Risk_Score"] = st.slider("Portfolio Risk Score", r["min"], r["max"], r["median"])
        inputs["Credit_Risk_Score"] = inputs["Portfolio_Risk_Score"]  # identical column, see report's data-quality note

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([inputs])
        row_encoded = pd.get_dummies(row, drop_first=True)
        row_aligned = row_encoded.reindex(columns=meta["trained_columns"], fill_value=0)

        pred = rf_model.predict(row_aligned)[0]
        prob = rf_model.predict_proba(row_aligned)[0][1]

        c1, c2 = st.columns(2)
        with c1:
            if pred == 1:
                st.success(f"Prediction: **Profitable**")
            else:
                st.error(f"Prediction: **Loss**")
        with c2:
            st.metric("Model's estimated probability of profit", f"{prob*100:.1f}%")

        st.caption(
            "Note the probability above will rarely stray far from ~50% for the reason "
            "stated in the warning box — the model has not found strong signal in this data."
        )

# ============================================================
# TAB 2 — Investor Segment Explorer (K-Means)
# ============================================================
with tab2:
    st.subheader("Which behavioural segment does this account fall into?")
    st.info(
        "Unlike the profitability model, this clustering **did** find genuine structure "
        "in the data (silhouette score ~0.22 — weak-to-moderate but real separation). "
        "The key finding from the project: these four behavioural segments do **not** "
        "line up with the platform's existing Investor_Type label (Retail/HNI/Institutional/Trader) "
        "— all four clusters are 55-63% Retail. Behaviour and the current label are two "
        "different things."
    )

    c1, c2 = st.columns(2)
    with c1:
        r = meta["numeric_ranges"]["Wallet_Age_Days"]
        wallet_age = st.slider("Wallet Age (days)", int(r["min"]), int(r["max"]), int(r["median"]), key="c_age")
        r = meta["numeric_ranges"]["Wallet_Balance_BTC"]
        wallet_balance = st.number_input("Wallet Balance (BTC)", r["min"], r["max"], r["median"], key="c_bal")
    with c2:
        r = meta["numeric_ranges"]["Portfolio_Risk_Score"]
        risk_score = st.slider("Portfolio Risk Score", r["min"], r["max"], r["median"], key="c_risk")
        velocity = st.slider("Transaction Velocity Score", 0.0, 100.0, 50.0, key="c_vel")

    if st.button("Find Segment", type="primary"):
        row = pd.DataFrame([{
            "Wallet_Age_Days": wallet_age,
            "Wallet_Balance_BTC": wallet_balance,
            "Portfolio_Risk_Score": risk_score,
            "Transaction_Velocity_Score": velocity,
        }])[meta["cluster_features"]]

        row_scaled = cluster_scaler.transform(row)
        cluster_id = int(kmeans_model.predict(row_scaled)[0])
        label = meta["cluster_labels"][str(cluster_id)]

        st.success(f"Segment: **{label}** (Cluster {cluster_id})")

        profile = meta["cluster_profile"][str(cluster_id)]
        crosstab = meta["cluster_investor_crosstab"][str(cluster_id)]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Typical profile of this cluster (dataset average)**")
            st.table(pd.DataFrame(profile, index=["Average"]).T.rename(columns={"Average": "Value"}))
        with c2:
            st.markdown("**Investor_Type mix within this cluster**")
            st.table(pd.DataFrame(crosstab, index=["% of cluster"]).T.rename(columns={"% of cluster": "%"}))

# ============================================================
# TAB 3 — Project Dashboard (static charts from the report)
# ============================================================
with tab3:
    st.subheader("Full analysis at a glance")
    charts = [
        ("chart1_target_dist.png", "Distribution of Trade Profitability"),
        ("chart2_txn_value_dist.png", "Distribution of Transaction Value"),
        ("chart3_corr_heatmap.png", "Correlation Heatmap of Key Numeric Variables"),
        ("chart4_boxplot.png", "Transaction Value by Profitability Outcome"),
        ("chart5_rf_importance.png", "Random Forest — Top 10 Feature Importances"),
        ("chart6_silhouette.png", "Silhouette Score by Number of Clusters"),
        ("chart7_cluster_profile.png", "Cluster Profiles (Normalized)"),
    ]
    cols = st.columns(2)
    for i, (fname, caption) in enumerate(charts):
        path = os.path.join(APP_DIR, "charts", fname)
        with cols[i % 2]:
            if os.path.exists(path):
                st.image(path, caption=caption, use_container_width=True)
