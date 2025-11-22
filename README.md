
# 🚀 Compressor Health Monitoring Using Machine Learning


Real-Time Fault Detection • Anomaly Monitoring • Predictive Maintenance

## 📌 Overview

This project builds a machine learning–powered compressor health monitoring system that analyzes real-time sensor data to:

• Detect anomalies (unsupervised learning)

• Predict hardware failures (supervised learning)

• Visualize compressor health

• Provide interactive predictions via FastAPI web UI

The application combines Isolation Forest, PCA, Random Forest, and a full FastAPI backend to deliver industrial-grade monitoring.


## 📂 Project Structure
```
App/
 ├── main.py
 ├── compressor.csv
 ├── templates/
 ├── static/
 ├── models/
 ├── outputs/
 └── README.md
 ```

## 📦 Installation Guide
1️⃣ Install Python

Make sure Python 3.8+ is installed:
```
python --version
```
2️⃣ Create Virtual Environment (Recommended)
Windows:
```
python -m venv venv
venv\Scripts\activate
```
Linux / macOS:
```
python3 -m venv venv
source venv/bin/activate
```

3️⃣ Install Required Libraries

Install all dependencies at once:
```
pip install -r requirements.txt
```

If you don't have a requirements.txt, use this combined installation command:

```
pip install fastapi uvicorn pandas numpy scikit-learn matplotlib joblib python-multipart jinja2
```

Optional libraries used:
```
pip install xgboost
pip install seaborn
pip install scipy
```
## 🧠 Machine Learning Models Used

🔹 1. Isolation Forest (Unsupervised Anomaly Detection)

- Detects unusual sensor patterns
- Suitable for high-dimensional industrial data
- Robust to noise

🔹 2. PCA (Dimensionality Reduction)

- Reduces sensor noise
- Helps visualize anomalies
- Improves clustering

🔹 3. Random Forest Classifier (Supervised Failure Prediction)

- Predicts:
    - Bearing failures
    - Water pump failures
    - Oil pump failures
    - Radiator faults
    - AC motor issues
    - Exhaust valve faults

- Reason for choosing:
    - High accuracy for tabular data
    - Handles correlated features well
    - Resistant to overfitting
## 🖥️ Running the FastAPI Application
🔹 Start the Backend Server
```
uvicorn main:app --reload
```
🔹 Open in Browser
```
http://127.0.0.1:8000/
```
## 📊 Dataset Overview
The system uses 20+ continuous sensor features:
- rpm
- motor_power
- torque
- outlet_pressure_bar
- air_flow
- noise_db
- outlet_temp
- wpump_outlet_press
- water_inlet_temp
- water_outlet_temp
- wpump_power
- water_flow
- oilpump_power
- oil_tank_temp
- gaccx, gaccy, gaccz
- haccx, haccy, haccz

Failure label created:
```
true_failure = max([bearings, wpump, oilpump, radiator, exvalve, acmotor])
```
##  🧪 Features of the Web App
✔️ Upload CSV & Predict Failures

✔️ Real-Time Anomaly Detection

✔️ Visual Metric Dashboard

✔️ PCA & Threshold Visualizations

✔️ Side-by-Side Sample Predictions

✔️ Auto-generated Confusion Matrix

## 📈 Results Summary
- Isolation Forest detects anomalies with high separation

- Random Forest provides accurate failure classification

- PCA visualizes health clusters clearly

- Dashboard improves interpretability
