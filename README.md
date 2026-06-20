# 🌪️ TerraNova FEMA Disaster Cost Prediction

## 🎯 Overview

TerraNova is an end-to-end machine learning system designed to forecast FEMA disaster recovery costs using historical disaster declarations, public assistance funding records, and disaster impact indicators.

The project demonstrates a complete machine learning workflow from data acquisition and feature engineering to model deployment through FastAPI and Streamlit.

### Key Highlights

✅ Disaster Cost Forecasting using Machine Learning

✅ Advanced Feature Engineering (Temporal, Funding & Severity Features)

✅ Model Comparison (Linear Regression, Random Forest, XGBoost)

✅ Real-Time Prediction API using FastAPI

✅ Interactive Streamlit Dashboard

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
FastAPI Deployment
        │
        ▼
Streamlit Dashboard
```

---

## 🤖 Model Performance

| Model             | RMSE (Log Scale) | R² Score   | MAE (Original Scale) |
| ----------------- | ---------------- | ---------- | -------------------- |
| Linear Regression | 5.3925           | 0.5008     | $26.17 Billion       |
| Random Forest     | 3.0367           | 0.8417     | $61.90 Million       |
| XGBoost           | **2.9988**       | **0.8456** | $65.63 Million       |

### 🏆 Best Model: XGBoost Regressor

#### Key Findings

* Lowest RMSE on the transformed target
* Highest R² score (0.8456)
* Explained approximately 84.6% of the variation in disaster recovery costs
* Selected as the final production model

#### Modeling Insights

* Approximately 50% of disasters received no FEMA funding
* Disaster funding distributions were highly right-skewed
* Log transformation significantly improved model stability and predictive performance

---

## 📸 Application Screenshots

### Streamlit Dashboard

![Streamlit Dashboard](assets/streamlit_dashboard.png)

### Prediction Example

![Prediction Example](assets/prediction_example.png)

### FastAPI Swagger Documentation

![Swagger API](assets/swagger_api.png)

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

---

## 📌 Key Findings

* XGBoost achieved the strongest predictive performance.
* Disaster duration and declaration timing were important predictors.
* Funding distributions were highly skewed.
* Approximately 50% of disasters received no FEMA funding.
* Log transformation significantly improved model stability.
* The final model explained approximately 84.6% of the variation in disaster recovery costs.

---

## 👨‍💻 Author

**Ransom Chukwu**

Data Science | Machine Learning | Health Informatics
