# Bitcoin Trading Analytics — Live Demo

MBA Advanced Analytics for Decision-Making — End Term Project
Group 1 | Finance Specialization | Dataset: DS1 (Bitcoin Advanced Financial Analytics)

Submitted by: Rishabh Rohra (Roll No. EBIZ 91), Aryendra Singh (Roll No. 64), Chandrakant Sahu

## What this is

A Streamlit web app deploying two of the models built for the project report, in three tabs:

1. **Trade Profitability Predictor** (Random Forest) — enter a hypothetical trade/investor
   profile, get a Profitable/Loss prediction. The app shows the honest caveat next to the
   result: test-set accuracy 55.5%, ROC-AUC 0.48 — no better than a coin flip. Across four
   independent techniques (Logistic Regression, Decision Tree, Random Forest, Linear
   Regression), none found a meaningful relationship between the available features and
   trade profitability in this dataset. That is the project's actual finding (report Section
   H), not a deployment defect — the app is a demonstration of the working pipeline, not a
   trading signal.
2. **Investor Segment Explorer** (K-Means) — assigns an investor profile to one of four
   behavioural segments the clustering actually found: Whales, Veteran active traders,
   Dormant holders, New active entrants. This is the technique that surfaced a real insight:
   none of these segments line up with the platform's existing `Investor_Type` label — all
   four are 55–63% Retail (report Section F).
3. **Project Dashboard** — the seven EDA/model charts from the report, viewable from the
   live link.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (3 tabs) |
| `train_and_save_models.py` | Reproduces training from the raw CSV if the `.pkl` files ever need regenerating (not used at runtime) |
| `rf_model.pkl` | Trained Random Forest classifier |
| `cluster_scaler.pkl`, `kmeans_model.pkl` | Trained K-Means clustering model and its feature scaler |
| `metadata.json` | Form input ranges, category options, cluster profiles/labels, and reported metrics — all baked in so the app never needs the raw CSV |
| `charts/` | The seven report charts, shown in the Project Dashboard tab |
| `requirements.txt` | Pinned package versions — must match what trained the `.pkl` files (scikit-learn 1.8.0), or loading the model can silently break |
| `startup_command.txt` | Command to paste into Azure App Service → Configuration → Startup Command |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Azure App Service (free tier)

1. Push this folder to a GitHub repository.
2. Create an Azure App Service: Linux, Python 3.11 runtime, F1 (Free) pricing tier.
3. In the App Service → **Deployment Center**, connect it to the GitHub repo (this
   auto-redeploys whenever you push).
4. In the App Service → **Configuration** → **General settings**, set **Startup Command**
   to the single line in `startup_command.txt`.
5. Save, wait for the deployment to finish, then open the app's URL.
