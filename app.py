from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np
import os
import joblib
import json

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt


# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMPLATES_DIR = os.path.join(BASE_DIR, "template")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# -----------------------------
# FEATURES
# -----------------------------
NUMERIC_FEATURES = [
    "rpm", "motor_power", "torque", "outlet_pressure_bar", "air_flow",
    "noise_db", "outlet_temp", "wpump_outlet_press", "water_inlet_temp",
    "water_outlet_temp", "wpump_power", "water_flow", "oilpump_power",
    "oil_tank_temp", "gaccx", "gaccy", "gaccz", "haccx", "haccy", "haccz"
]

LABEL_COLS = ["bearings", "wpump", "oilpump", "filter", "exvalve", "acmotor"]
TRUE_FAILURE_COL = "true_failure"


# -----------------------------
# RISK CATEGORY UTILS
# -----------------------------
def get_risk_category(score: float) -> str:
    if score >= 0.80:
        return "HIGH"
    elif score >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def estimate_ttf(category: str) -> str:
    if category == "HIGH":
        return "0–72 hours"
    elif category == "MEDIUM":
        return "3–10 days"
    else:
        return ">10 days"


# -----------------------------
# BUILD/ENSURE TRUE FAILURE LABEL
# -----------------------------
def ensure_true_failure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TRUE_FAILURE_COL not in df.columns:
        available = [c for c in LABEL_COLS if c in df.columns]
        if not available:
            raise ValueError("No component-level failure columns found to build 'true_failure'.")
        df[TRUE_FAILURE_COL] = df[available].max(axis=1).astype(int)
    return df


# -----------------------------
# SUPERVISED TRAIN/TEST EVALUATION
# -----------------------------
def evaluate_supervised_model(df: pd.DataFrame) -> dict:
    """
    Split data into train/test and evaluate a RandomForest classifier.
    Returns test-set metrics and confusion matrix.
    """
    df = ensure_true_failure(df)
    X = df[NUMERIC_FEATURES]
    y = df[TRUE_FAILURE_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # Save eval metrics separately if you want
    eval_path = os.path.join(OUTPUT_DIR, "supervised_test_metrics.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# -----------------------------
# NEIGHBOR-BASED VERIFICATION
# -----------------------------
def verify_prediction_with_neighbors(
    new_sample: dict,
    dataset_path: str = None,
    n_neighbors: int = 5
) -> dict:
    """
    Compare new sample with historical data using nearest neighbors.
    Uses the health-scored dataset if available, otherwise uploaded.csv.
    """
    if dataset_path is None:
        scored_path = os.path.join(OUTPUT_DIR, "health_scored_dataset.csv")
        uploaded_path = os.path.join(OUTPUT_DIR, "uploaded.csv")
        if os.path.exists(scored_path):
            dataset_path = scored_path
        elif os.path.exists(uploaded_path):
            dataset_path = uploaded_path
        else:
            raise FileNotFoundError("No dataset found for verification. Train the model first.")

    df = pd.read_csv(dataset_path)
    df = ensure_true_failure(df)

    # Keep only rows with all numeric features present
    missing_features = [f for f in NUMERIC_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing required features: {missing_features}")

    X = df[NUMERIC_FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sample_df = pd.DataFrame([new_sample])[NUMERIC_FEATURES]
    sample_scaled = scaler.transform(sample_df)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(X_scaled)

    distances, indices = nbrs.kneighbors(sample_scaled)
    matches = df.iloc[indices[0]].copy()
    matches["distance"] = distances[0]

    failure_rate = matches[TRUE_FAILURE_COL].mean()

    verification = (
        "LIKELY FAILURE" if failure_rate >= 0.6 else
        "POSSIBLE FAILURE" if failure_rate >= 0.3 else
        "LIKELY NORMAL"
    )

    return {
        "failure_rate_of_neighbors": round(float(failure_rate), 4),
        "verification_result": verification,
        "nearest_samples": matches.head(n_neighbors).to_dict(orient="records")
    }


# -----------------------------
# TRAINING PIPELINE
# -----------------------------
def train_pipeline(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> dict:

    df = ensure_true_failure(df)
    df = df.copy()

    X = df[NUMERIC_FEATURES]
    y = df[TRUE_FAILURE_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Isolation Forest Tuning
    # -----------------------------
    contamination_grid = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    tuning_results = []

    for c in contamination_grid:
        iso_tmp = IsolationForest(n_estimators=200, contamination=c, random_state=42)
        iso_tmp.fit(X_scaled)
        y_pred = (iso_tmp.predict(X_scaled) == -1).astype(int)
        iso_raw = -iso_tmp.decision_function(X_scaled)

        f1 = f1_score(y, y_pred, zero_division=0)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)

        try:
            roc = roc_auc_score(y, iso_raw)
        except:
            roc = None

        ap = average_precision_score(y, iso_raw)

        tuning_results.append({
            "contamination": c,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "roc_auc": roc,
            "avg_precision": ap
        })

    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(os.path.join(output_dir, "iso_tuning_results.csv"), index=False)

    best_c = tuning_df.sort_values("f1", ascending=False).iloc[0]["contamination"]

    # -----------------------------
    # Final Isolation Forest
    # -----------------------------
    iso = IsolationForest(n_estimators=300, contamination=best_c, random_state=42)
    iso.fit(X_scaled)

    # -----------------------------
    # PCA Reconstruction Error
    # -----------------------------
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    X_rec = pca.inverse_transform(X_pca)
    recon_error = np.mean((X_scaled - X_rec) ** 2, axis=1)

    # Normalize anomaly scores
    iso_mm = MinMaxScaler()
    pca_mm = MinMaxScaler()

    iso_norm = iso_mm.fit_transform((-iso.decision_function(X_scaled)).reshape(-1, 1)).ravel()
    pca_norm = pca_mm.fit_transform(recon_error.reshape(-1, 1)).ravel()

    df["failure_risk_score"] = 0.6 * iso_norm + 0.4 * pca_norm
    df["risk_category"] = df["failure_risk_score"].apply(get_risk_category)
    df["pred_failure"] = (df["risk_category"] == "HIGH").astype(int)
    df["estimated_time_to_failure"] = df["risk_category"].apply(estimate_ttf)

    # -----------------------------
    # Unsupervised Metrics
    # -----------------------------
    unsup_acc = accuracy_score(y, df["pred_failure"])
    unsup_ap = average_precision_score(y, df["failure_risk_score"])

    try:
        unsup_roc = roc_auc_score(y, df["failure_risk_score"])
    except:
        unsup_roc = None

    # -----------------------------
    # SUPERVISED RandomForest MODEL (FULL DATA)
    # -----------------------------
    rf_sup = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42
    )
    rf_sup.fit(X_scaled, y)

    y_sup_pred = rf_sup.predict(X_scaled)

    supervised_acc = accuracy_score(y, y_sup_pred)
    supervised_f1 = f1_score(y, y_sup_pred, zero_division=0)
    supervised_precision = precision_score(y, y_sup_pred, zero_division=0)
    supervised_recall = recall_score(y, y_sup_pred, zero_division=0)

    # SAVE SUPERVISED MODEL (FOR LIVE PREDICTION)
    joblib.dump(rf_sup, os.path.join(output_dir, "rf_supervised.pkl"))

    # Feature importance model
    rf_feat = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    rf_feat.fit(X_scaled, y)

    feat_imp = pd.DataFrame({
        "feature": NUMERIC_FEATURES,
        "importance": rf_feat.feature_importances_
    }).sort_values("importance", ascending=False)

    feat_imp.to_csv(os.path.join(output_dir, "feature_importances.csv"), index=False)

    # Save unsupervised models and scalers
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(iso, os.path.join(output_dir, "isolation_forest.pkl"))
    joblib.dump(pca, os.path.join(output_dir, "pca_model.pkl"))
    joblib.dump(iso_mm, os.path.join(output_dir, "iso_mm.pkl"))
    joblib.dump(pca_mm, os.path.join(output_dir, "pca_mm.pkl"))

    # Save full health-scored dataset
    df.to_csv(os.path.join(output_dir, "health_scored_dataset.csv"), index=False)

    # -----------------------------
    # SUPERVISED TEST-SET EVALUATION
    # -----------------------------
    test_metrics = evaluate_supervised_model(df)

    # -----------------------------
    # SUMMARY FOR DASHBOARD
    # -----------------------------
    summary = {
        # Unsupervised
        "unsupervised_accuracy": float(unsup_acc),
        "unsupervised_roc_auc": None if unsup_roc is None else float(unsup_roc),
        "unsupervised_ap": float(unsup_ap),

        # Supervised (trained on full data, in-sample)
        "supervised_train_accuracy": float(supervised_acc),
        "supervised_train_f1": float(supervised_f1),
        "supervised_train_precision": float(supervised_precision),
        "supervised_train_recall": float(supervised_recall),

        # Supervised test metrics (more important)
        "supervised_test": test_metrics,

        # Feature importance (top 10)
        "top_features": feat_imp.head(10).to_dict(orient="records")
    }

    return summary


# -----------------------------
# PREDICTION FROM SAVED MODELS
# -----------------------------
def predict_failure_from_saved(new_data: dict, output_dir: str = OUTPUT_DIR) -> dict:

    scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
    iso = joblib.load(os.path.join(output_dir, "isolation_forest.pkl"))
    pca = joblib.load(os.path.join(output_dir, "pca_model.pkl"))
    iso_mm = joblib.load(os.path.join(output_dir, "iso_mm.pkl"))
    pca_mm = joblib.load(os.path.join(output_dir, "pca_mm.pkl"))
    rf_sup = joblib.load(os.path.join(output_dir, "rf_supervised.pkl"))

    new_df = pd.DataFrame([new_data])[NUMERIC_FEATURES]
    scaled = scaler.transform(new_df)

    # Unsupervised risk score
    iso_raw = -iso.decision_function(scaled).reshape(-1, 1)
    iso_norm = iso_mm.transform(iso_raw).ravel()[0]

    X_pca = pca.transform(scaled)
    X_rec = pca.inverse_transform(X_pca)
    recon_error = np.mean((scaled - X_rec) ** 2, axis=1).reshape(-1, 1)
    pca_norm = pca_mm.transform(recon_error).ravel()[0]

    risk_score = 0.6 * iso_norm + 0.4 * pca_norm
    risk_category = get_risk_category(risk_score)
    ttf = estimate_ttf(risk_category)

    # Supervised prediction
    supervised_pred = int(rf_sup.predict(scaled)[0])

    return {
        "risk_score": round(float(risk_score), 4),
        "risk_category": risk_category,
        "estimated_time_to_failure": ttf,
        "supervised_prediction": supervised_pred
    }


# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload CSV, train full pipeline, compute metrics, save summary.
    """
    contents = await file.read()
    save_path = os.path.join(OUTPUT_DIR, "uploaded.csv")

    with open(save_path, "wb") as f:
        f.write(contents)

    df = pd.read_csv(save_path)
    summary = train_pipeline(df)

    joblib.dump(summary, os.path.join(OUTPUT_DIR, "last_summary.pkl"))

    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    """
    Simple endpoint for HTML dashboard.
    Template can read 'summary' dict keys:
      - unsupervised_accuracy, unsupervised_roc_auc, unsupervised_ap
      - supervised_train_*
      - supervised_test (dict)
      - top_features (list of {feature, importance})
    """
    path = os.path.join(OUTPUT_DIR, "last_summary.pkl")
    summary = joblib.load(path) if os.path.exists(path) else None
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "summary": summary}
    )


@app.post("/predict")
async def predict(request: Request):
    """
    JSON input with all numeric sensor fields → returns prediction.
    """
    payload = await request.json()

    try:
        result = predict_failure_from_saved(payload)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return JSONResponse(result)


@app.post("/verify")
async def verify(request: Request):
    """
    JSON input (same as /predict) → returns:
      - model prediction
      - neighbor-based verification
    """
    payload = await request.json()

    try:
        prediction = predict_failure_from_saved(payload)
        verification = verify_prediction_with_neighbors(payload)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return JSONResponse({
        "prediction": prediction,
        "verification": verification
    })
