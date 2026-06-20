# 🌪️ TerraNova FEMA Disaster Cost Prediction

## 🎯 Overview

TerraNova is an end-to-end machine learning system developed to forecast FEMA disaster recovery costs using historical disaster declarations, public assistance funding records, and disaster impact indicators.

The project supports evidence-based disaster planning by helping emergency management agencies estimate recovery costs earlier in the disaster lifecycle. It demonstrates the complete machine learning workflow from exploratory analysis and feature engineering to model deployment through FastAPI and Streamlit.

### Key Highlights

✅ Disaster Cost Forecasting using Machine Learning

✅ Advanced Feature Engineering (Temporal, Funding & Severity Features)

✅ Model Comparison (Linear Regression, Random Forest, XGBoost)

✅ Real-Time Prediction API using FastAPI

✅ Interactive Streamlit Dashboard

✅ Automated Experiment Tracking using MLflow

✅ Production-Ready Project Structure

---

## 📊 Project Scope

| Metric              | Value                   |
| ------------------- | ----------------------- |
| Industry            | Emergency Management    |
| Domain              | Disaster Recovery       |
| Data Source         | FEMA Open Data          |
| Records             | 69,000+ Disaster Events |
| Geographic Coverage | United States           |
| Time Period         | Multiple Fiscal Years   |
| Deployment          | FastAPI + Streamlit     |
| Best Model          | XGBoost (R² = 0.8456)   |

---

## 🏢 Business Context

### The Challenge

Federal agencies often struggle to estimate disaster recovery costs during the early stages of emergency response.

Key challenges include:

* Uncertain disaster funding requirements
* Inefficient resource allocation
* Delayed recovery planning
* Limited forecasting capability
* Increasing disaster frequency and severity

### The Solution

TerraNova provides a machine learning solution capable of:

* Predicting disaster recovery costs
* Identifying high-cost disaster events
* Supporting resource allocation decisions
* Providing real-time cost estimates
* Enabling proactive disaster planning

---

## 📈 Business Impact

Accurate disaster cost forecasting can help decision-makers:

* Improve emergency funding allocation
* Prioritize high-impact disaster events
* Support recovery planning and budgeting
* Reduce uncertainty during emergency response
* Enable data-driven resource deployment

---

## 🏗️ Feature Engineering

### Temporal Features

* Disaster Duration
* Declaration Delay
* Declaration Month
* Declaration Quarter
* Declaration Season

### Funding Features

* Federal Share Metrics
* Project Size Indicators
* Public Assistance Aggregations

### Disaster Severity Features

* Registration Counts
* Assistance Indicators
* Disaster Scale Metrics

### DSF Score

A custom Disaster Severity Framework (DSF) score was developed to quantify disaster impact and severity.

---

## 🏗️ System Architecture

```text
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
MLflow Experiment Tracking
        │
        ▼
FastAPI Deployment
        │
        ▼
Streamlit Dashboard
```

---

## 🤖 Model Development

### Models Evaluated

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

The objective was to predict FEMA recovery costs using engineered disaster-level features while avoiding target leakage and preserving real-world prediction capability.

---

## 🤖 Model Performance

| Model             | RMSE (Log Scale) | R² Score   | MAE (Original Scale) |
| ----------------- | ---------------- | ---------- | -------------------- |
| Linear Regression | 5.3925           | 0.5008     | $26.17 Billion       |
| Random Forest     | 3.0367           | 0.8417     | $61.90 Million       |
| XGBoost           | **2.9988**       | **0.8456** | $65.63 Million       |

### 🏆 Best Model: XGBoost Regressor

#### Performance Summary

* RMSE (Log Scale): **2.9988**
* R² Score: **0.8456**
* Target Variable: **log_totalobligated**
* Deployment: **FastAPI + Streamlit**

### Model Insights

The XGBoost model achieved the strongest overall performance with an R² score of 0.8456, explaining approximately 84.6% of the variation in FEMA recovery costs.

Several important observations emerged during modeling:

* Approximately 50% of disasters received no FEMA funding.
* Disaster recovery costs exhibited a highly right-skewed distribution.
* Log transformation significantly improved model stability.
* Temporal disaster characteristics were strong predictors of recovery cost.
* Ensemble tree-based models substantially outperformed linear regression.

---

## 🔍 Experiment Tracking

MLflow was used to:

* Track model experiments
* Compare model performance
* Log hyperparameters
* Store evaluation metrics
* Manage model artifacts

This ensured reproducible model development and transparent model selection.

---

## 📸 Application Screenshots

### Streamlit Dashboard

![Streamlit Dashboard](./assets/streamlit_dashboard.png)

Interactive dashboard for disaster scenario analysis and real-time FEMA recovery cost prediction.

---

### Prediction Example

![Prediction Example](./assets/prediction_example.png)

Example prediction generated by the deployed XGBoost model showing estimated disaster recovery costs.

---

### FastAPI Swagger Documentation

![Swagger API](./assets/swagger_api.png)

Interactive API documentation generated automatically by FastAPI.

---

## 🚀 API Endpoint

### POST /predict

#### Example Request

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

#### Example Response

```json
{
  "predicted_log_cost": 18.0219,
  "predicted_recovery_cost": 67110464
}
```

---

## 📁 Project Structure

```text
TerraNova_project/
│
├── assets/
│   ├── streamlit_dashboard.png
│   ├── prediction_example.png
│   └── swagger_api.png
│
├── data/
│   ├── raw/
│   │   ├── declarations.csv
│   │   ├── public_assistance.csv
│   │   └── disaster_summaries.csv
│   │
│   └── processed/
│       └── features_fema.csv
│
├── models/
│   └── fema_cost_model.pkl
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
├── streamlit_app/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Technology Stack

| Category                | Technologies          |
| ----------------------- | --------------------- |
| Machine Learning        | Scikit-Learn, XGBoost |
| Data Processing         | Pandas, NumPy         |
| Visualization           | Matplotlib, Seaborn   |
| Experiment Tracking     | MLflow                |
| API Development         | FastAPI               |
| Dashboard               | Streamlit             |
| Model Serialization     | Joblib                |
| Development Environment | VS Code               |
| Version Control         | Git, GitHub           |

---

## 🚀 Quick Start

### Clone Repository

```bash
git clone https://github.com/Alpha-rammy/TerraNova_project.git
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

### Launch MLflow

```bash
mlflow ui
```

---

## 📌 Key Findings

* XGBoost achieved the strongest predictive performance.
* Disaster duration and declaration timing were important predictors.
* Funding distributions were highly skewed.
* Approximately 50% of disasters received no FEMA funding.
* Log transformation significantly improved model stability.
* Ensemble models significantly outperformed linear regression.
* The final model explained approximately 84.6% of the variation in disaster recovery costs.

---

## 👨‍💻 Author

**Ransom Chukwu**

Medical Doctor (MD) | Master of Public Health (MPH)

Data Science • Machine Learning • Health Informatics • Public Health Analytics

GitHub: https://github.com/Alpha-rammy
