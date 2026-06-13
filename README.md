🌪️ TerraNova FEMA Disaster Cost Prediction


🎯 Overview


TerraNova is an end-to-end machine learning system designed to forecast FEMA disaster recovery costs using historical disaster declarations, public assistance funding records, and disaster impact indicators.


This project demonstrates production-grade data science practices including:

✅ Disaster Cost Forecasting using Machine Learning

✅ Feature Engineering (Temporal, Funding, Severity Features)

✅ Model Comparison (Linear Regression, Random Forest, XGBoost)

✅ Real-time Scoring API built with FastAPI

✅ Interactive Dashboard using Streamlit

✅ Automated Model Tracking with MLflow
✅ Production-Ready Project Structure


## 📊 Project Scope

| Metric | Value |
|---------|---------|
| Industry | Emergency Management |
| Domain | Disaster Recovery |
| Data Source | FEMA Open Data |
| Records | 69,000+ Disaster Events |
| Geographic Coverage | United States |
| Time Period | Multiple Fiscal Years |
| Deployment | FastAPI + Streamlit |
| Best Model | XGBoost |

🏢 Business Context


The Challenge


Federal agencies often struggle to estimate disaster recovery costs during the early stages of emergency response.



🏗️ Feature Engineering

Temporal Features

- Disaster Duration
- Declaration Delay
- Declaration Month
- Declaration Quarter
- Declaration Season

Funding Features

- Federal Share Metrics
- Project Size Indicators
- Public Assistance Aggregations

Disaster Severity Features

- Registration Counts
- Assistance Indicators
- Disaster Scale Features

DSF Score
- Custom Disaster Severity Framework (DSF) Score developed to quantify disaster impact.

🤖 Model Development

Models Evaluated

M## 🤖 Model Performance

| Model | RMSE (Log Scale) | R² Score | MAE (Original Scale) |
|---------|---------|---------|---------|
| Linear Regression | 5.3925 | 0.5008 | $26.17 Billion |
| Random Forest | 3.0367 | 0.8417 | $61.90 Million |
| XGBoost | 2.9988 | 0.8456 | $65.63 Million |


Best Model

🏆 XGBoost Regressor

Key findings:

- XGBoost produced the lowest prediction error.
- Approximately 50% of disasters received no FEMA funding.
- Cost distributions were highly right-skewed.
- Log transformation improved model stability and predictive performance.

📡 FastAPI Deployment

Features
 
- Real-time prediction endpoint
- Swagger UI documentation
- JSON request/response interface
- Production-ready API structure


## 📡 FastAPI Deployment

### Endpoint

```http
POST /predict

{
  "predicted_log_cost": 17.6118,
  "predicted_recovery_cost": 44536528
}

assets/
├── swagger_api.png
├── prediction_example.png
├── actual_vs_predicted.png
└── model_comparison.png

🏗️ System Architecture


## The Solution

A machine learning system capable of:

* Predicting disaster recovery costs
* Identifying high-cost disaster events
* Supporting resource allocation decisions
* Providing real-time cost estimates
* Enabling proactive disaster planning


# ✨ Key Features

## 🌪️ Disaster Cost Forecasting

* Linear Regression Baseline
* Random Forest Regressor
* XGBoost Regressor
* Log-transformed target modeling

### Temporal Features

* Disaster Duration
* Declaration Delay
* Declaration Month
* Declaration Quarter
* Declaration Season

### Funding Features

* Federal Share Metrics
* Public Assistance Aggregations
* Project Size Indicators

### Disaster Severity Features

* Registration Statistics
* Assistance Indicators
* Disaster Scale Metrics

### DSF Score

Custom Disaster Severity Framework (DSF) score developed to quantify disaster impact and scale.



# 🏗️ System Architecture

(Complete FEMA disaster cost prediction project)
Raw FEMA Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
FastAPI Deployment
        │
        ▼
Streamlit Dashboard





# 🤖 Model Performance

| Model             | RMSE (Log Scale) | R² Score   | MAE (Original Scale) |
| ----------------- | ---------------- | ---------- | -------------------- |
| Linear Regression | 5.3925           | 0.5008     | $26.17 Billion       |
| Random Forest     | 3.0367           | 0.8417     | $61.90 Million       |
| XGBoost           | **2.9988**       | **0.8456** | $65.63 Million       |

## Best Model

🏆 XGBoost Regressor

* RMSE (Log): 2.9988
* R² Score: 0.8456
* Target Variable: log_totalobligated
* Deployment: FastAPI + Streamlit

Approximately 50% of disasters received no FEMA funding, resulting in a highly skewed target distribution. Log transformation significantly improved model stability and predictive performance.

---

# 📸 Application Screenshots

### Streamlit Dashboard

![Streamlit Dashboard](assets/streamlit_dashboard.png)

### Prediction Example

![Prediction Example](assets/prediction_example.png)

### FastAPI Swagger Documentation

![Swagger API](assets/swagger_api.png)

---

# 🚀 API Endpoint

### POST /predict

Example Request

```json
{
  "fydeclared": 2024,
  "state": "FL",
  "declarationtype": "DR",
  "incidenttype": "Hurricane",
  "designatedarea": "Statewide",
  "disaster_duration_days": 120,
  "declaration_delay_days": 30
}
```

Example Response

```json
{
  "predicted_log_cost": 18.0219,
  "predicted_recovery_cost": 67110464
}
```

---

# 📁 Project Structure


TerraNova_project/
│
├── assets/
│   ├── streamlit_dashboard.png
│   ├── prediction_example.png
│   └── swagger_api.png
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── fema_cost_model.pkl
  (Complete FEMA disaster cost prediction project)
│
├── Notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/

│   │   ├── build_features.py
│   │   └── dsf.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   └── evaluate.py
│   │
│   ├── api/
│   │   └── main.py
│   │
│   └── config.py
│
├── models/
│   └── fema_cost_model.pkl
│
├── streamlit_app/
│   └── app.py
│
├── assets/
│   ├── swagger_api.png
│   ├── prediction_example.png
│   ├── actual_vs_predicted.png
│   └── model_comparison.png
│
├── requirements.txt
├── .gitignore
└── README.md


Technologies Used
- Python
- Pandas
- Scikit-Learn
- XGBoost
- FastAPI
- Joblib
- VS Code



# 🛠️ Technology Stack

| Category                | Technologies          |
| ----------------------- | --------------------- |
| Machine Learning        | Scikit-Learn, XGBoost |
| Data Processing         | Pandas, NumPy         |
| Visualization           | Matplotlib, Seaborn   |
| API Development         | FastAPI               |
| Dashboard               | Streamlit             |
| Model Serialization     | Joblib                |
| Development Environment | VS Code               |
| Version Control         | Git, GitHub           |


# 🚀 Quick Start

### Clone Repository

```bash
git clone https://github.com/yourusername/TerraNova_project.git
cd TerraNova_project
```

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run FastAPI

```bash
uvicorn src.api.main:app --reload
```

### Run Streamlit

```bash
streamlit run streamlit_app/app.py
```

---

# 📌 Key Findings

* XGBoost achieved the strongest predictive performance.
* Disaster duration and declaration timing were important predictors.
* Funding distributions were highly skewed.
* Approximately 50% of disasters received no FEMA funding.
* Log transformation significantly improved model stability.
* The final model explained approximately 84.6% of the variation in disaster recovery costs.

---

# 👨‍💻 Author

Ransom Chukwu


Data Science | Machine Learning | 
 (Complete FEMA disaster cost prediction project)
