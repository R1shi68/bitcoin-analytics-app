"""
Re-trains the Random Forest classifier and K-Means clustering model used by
app.py, and saves them plus metadata.json (form ranges, cluster labels,
cluster profile stats). Run this once if you ever need to regenerate the
.pkl files from the raw dataset — app.py itself does NOT retrain anything,
it only loads these saved files.
"""
import pandas as pd
import numpy as np
import joblib
import json

DATA_PATH = "bitcoin_advanced_financial_analytics_dataset_3000.csv"  # place the dataset CSV next to this script to re-run

df = pd.read_csv(DATA_PATH)

# ---------- Impute (same as report) ----------
df['RSI_14'] = df['RSI_14'].fillna(df['RSI_14'].median())
df['Wallet_Balance_BTC'] = df['Wallet_Balance_BTC'].fillna(df['Wallet_Balance_BTC'].median())

# ---------- Target ----------
y = df['Target_Profitability'].map({'Profitable': 1, 'Loss': 0})

numeric_features = ['Wallet_Age_Days', 'Wallet_Balance_BTC', 'BTC_Price_USD', 'BTC_Quantity',
                     'Transaction_Value_USD', 'Transaction_Fee_USD', 'Leverage_Ratio',
                     'Price_24h_Change_Percent', 'Price_7d_Change_Percent', 'Volatility_30d_Percent',
                     'RSI_14', 'Fear_Greed_Index', 'Portfolio_Risk_Score', 'Credit_Risk_Score']
categorical_features = ['Investor_Type', 'KYC_Status', 'Two_FA_Status', 'Device_Type', 'Market_Sentiment']
features = numeric_features + categorical_features

X = pd.get_dummies(df[features], drop_first=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
rf_test_accuracy = accuracy_score(y_test, y_pred)
rf_test_roc_auc = roc_auc_score(y_test, y_prob)
print("RF accuracy:", rf_test_accuracy, "| RF ROC-AUC:", rf_test_roc_auc)

joblib.dump(rf_model, "rf_model.pkl")

# ---------- K-Means clustering ----------
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

cluster_features = ['Wallet_Age_Days', 'Wallet_Balance_BTC', 'Portfolio_Risk_Score', 'Transaction_Velocity_Score']
X_cluster = df[cluster_features]
cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans_model.fit_predict(X_cluster_scaled)

joblib.dump(cluster_scaler, "cluster_scaler.pkl")
joblib.dump(kmeans_model, "kmeans_model.pkl")

cluster_profile = df.groupby('Cluster')[cluster_features].mean().round(2)
crosstab = (pd.crosstab(df['Cluster'], df['Investor_Type'], normalize='index') * 100).round(1)

# Descriptive labels — assign by hand after inspecting cluster_profile.
# (see report Section E/F: high Wallet_Balance_BTC = "Whales", high
# Wallet_Age_Days + high velocity = "Veteran active traders", low velocity =
# "Dormant holders", low wallet age + high velocity = "New active entrants")
cluster_labels = {
    "0": "Whales",
    "1": "Veteran active traders",
    "2": "Dormant holders",
    "3": "New active entrants",
}

metadata = {
    "trained_columns": X.columns.tolist(),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "category_options": {c: sorted(df[c].dropna().unique().tolist()) for c in categorical_features},
    "numeric_ranges": {c: {"min": float(df[c].min()), "max": float(df[c].max()), "median": float(df[c].median())} for c in numeric_features},
    "cluster_features": cluster_features,
    "cluster_profile": cluster_profile.to_dict(orient="index"),
    "cluster_investor_crosstab": crosstab.to_dict(orient="index"),
    "cluster_labels": cluster_labels,
    "rf_test_accuracy": rf_test_accuracy,
    "rf_test_roc_auc": rf_test_roc_auc,
}

with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved: rf_model.pkl, cluster_scaler.pkl, kmeans_model.pkl, metadata.json")
