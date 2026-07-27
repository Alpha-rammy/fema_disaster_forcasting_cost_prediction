# TerraNova Disaster Cost Recovery Forecasting

## Predicting Federal Disaster Recovery Costs Using Early Disaster Declaration Information

## Project Overview

Natural disasters such as hurricanes, floods, wildfires, and severe storms often require significant financial support for response and recovery. However, estimating recovery costs immediately after a disaster declaration is challenging because detailed damage assessments and funding applications can take weeks or months to complete.

The goal of this project is to develop a machine learning model that estimates the final federal disaster recovery cost using only information available at the time of the disaster declaration. This provides an early cost forecast that can support emergency planning, resource allocation, and budgeting before the formal funding process begins.

Unlike many disaster recovery models, this project deliberately excludes Public Assistance funding information from the predictor variables to avoid data leakage. Instead, Public Assistance data is used only to calculate the target variable (final obligated recovery cost), ensuring that the model reflects a realistic real-world deployment scenario.

---

## Business Problem

Emergency management agencies must make critical decisions immediately after a disaster declaration, often before detailed project assessments and funding requests are available.

Without an early estimate of recovery costs, decision-makers may face challenges in:

- Allocating emergency resources effectively.
- Planning recovery budgets.
- Prioritising high-impact disasters.
- Supporting early strategic decision-making.

This project addresses that challenge by providing an early prediction of disaster recovery costs using only declaration-stage information.

---

## Project Objective

The objective of this project is to build an interpretable machine learning model that predicts the final federal disaster recovery cost immediately after a disaster declaration.

The project focuses on three key principles:

- Using only information available at the declaration stage.
- Preventing target leakage by excluding funding-related variables from the predictors.
- Producing an accurate and explainable cost forecasting model suitable for operational use.

---

## Dataset

This project uses disaster data consisting of three related datasets.

### 1. Disaster Declarations

Used to engineer the predictor variables.

Examples include:

- State
- Disaster type
- Declaration type
- Declaration date
- Designated areas
- Historical disaster information

### 2. Public Assistance

Used only to calculate the modelling target:

- Total Federal Recovery Cost
- Log-transformed Recovery Cost

No Public Assistance funding variables were used as predictor features.

### 3. Disaster Summaries

Used for exploratory data analysis and understanding overall disaster trends but excluded from the prediction model.

---

## Machine Learning Workflow

```text
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Model Explainability
        │
        ▼
FastAPI Deployment
```

---

# Project Structure

```
TerraNova_Disaster_Cost_Forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── terranova_disaster_cost_model.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_explainability.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── api/
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

## Programming

- Python 3.x

## Data Processing

- Pandas
- NumPy

## Data Visualisation

- Matplotlib
- Seaborn

## Machine Learning

- Scikit-learn
- XGBoost

## Model Explainability

- SHAP

## Model Deployment

- FastAPI
- Uvicorn

---

# Feature Engineering

The objective of feature engineering was to create meaningful predictors using only information available immediately after a disaster declaration.

The following features were engineered:

### Temporal Features

- Declaration delay
- Declaration month
- Declaration year
- Declaration weekday
- Declaration season

### Geographic Features

- State
- Number of designated disaster areas

### Historical Disaster Features

- Previous disasters in the state
- Previous disasters of the same incident type
- Previous disasters of the same incident type within the state
- Days since the previous state disaster
- Days since the previous disaster of the same incident type

### Operational Feature

- Ongoing disaster indicator at the time of declaration

These engineered features simulate the information that would realistically be available during the early stages of disaster response.

---

# Machine Learning Models

Three regression models were developed and evaluated.

| Model | Purpose |
|-------|----------|
| Linear Regression | Baseline model |
| Random Forest Regressor | Ensemble tree model |
| XGBoost Regressor | Gradient boosting model |

The models were evaluated using a chronological train-test split to simulate real-world forecasting, where future disasters are predicted using knowledge from past events.

---

# Model Performance

The models were evaluated using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

| Model | RMSE (Log Scale) | R² Score |
|--------|-----------------:|----------:|
| Linear Regression | 1.383 | 0.516 |
| Random Forest | 1.125 | 0.679 |
| XGBoost | 1.112 | **0.688** |

XGBoost achieved the best overall predictive performance and was selected as the final production model.

---

# Model Explainability

To improve transparency and support decision-making, the final XGBoost model was interpreted using multiple explainability techniques.

The analysis included:

- Feature Importance
- SHAP Summary Plot
- SHAP Waterfall Plot

These techniques helped identify the declaration-stage factors that contributed most to the predicted recovery costs and provided both global and individual prediction explanations.

---

---

# Business Impact

Accurately estimating disaster recovery costs immediately after a disaster declaration can help emergency management agencies make faster and more informed decisions.

Potential applications include:

- Early disaster recovery budgeting
- Resource allocation and planning
- Prioritisation of high-impact disasters
- Financial risk assessment
- Decision support for emergency management agencies

By relying only on declaration-stage information, the model can be deployed before detailed damage assessments or funding applications become available.

---

# FastAPI Deployment

The trained XGBoost model was deployed using **FastAPI** to provide real-time disaster recovery cost predictions.

### Available Endpoint

**POST** `/predict`

The API accepts declaration-stage information and returns:

- Predicted recovery cost (log scale)
- Estimated federal recovery cost (original dollar scale)

This demonstrates how a machine learning model can be integrated into operational decision-support systems.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your_username>/TerraNova_Disaster_Cost_Forecasting.git

cd TerraNova_Disaster_Cost_Forecasting
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Running the Project

### 1. Perform Exploratory Data Analysis

Open:

```
notebooks/01_eda.ipynb
```

---

### 2. Preprocess the Data

Run:

```
notebooks/02_preprocessing.ipynb
```

---

### 3. Engineer Features

Run:

```
notebooks/03_feature_engineering.ipynb
```

---

### 4. Train the Models

Run:

```
notebooks/04_modeling.ipynb
```

---

### 5. Explain Model Predictions

Run:

```
notebooks/05_explainability.ipynb
```

---

### 6. Launch the FastAPI Application

```bash
uvicorn src.api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

to access the interactive Swagger API documentation.

---

# Future Improvements

Possible enhancements include:

- Incorporating weather and climate data
- Integrating satellite imagery for damage assessment
- Including socioeconomic and demographic indicators
- Hyperparameter optimisation using Optuna
- Continuous model retraining as new disaster data becomes available
- Cloud deployment using Azure or AWS
- Building an interactive dashboard for operational monitoring

---

# Key Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Regression Modeling
- Time-Aware Train-Test Splitting
- Model Evaluation
- Model Explainability with SHAP
- Pipeline Development
- FastAPI Deployment
- Git & GitHub Version Control

---

# Author

**Ransom Chukwu**

Medical Doctor | Public Health Professional | Data Scientist

Passionate about applying machine learning and analytics to support evidence-based decision-making in healthcare, disaster management, and public sector operations.

---

# Acknowledgements

This project uses publicly available disaster data provided by the **Federal Emergency Management Agency (FEMA)**. The data was used solely for educational and portfolio purposes to demonstrate end-to-end data science and machine learning workflows.