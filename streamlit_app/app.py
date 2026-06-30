import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st



# PATHS


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fema_cost_model.pkl")



# LOAD MODEL


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# PAGE CONFIG


st.set_page_config(
    page_title="TerraNova | FEMA Cost Predictor",
    page_icon="🌪️",
    layout="wide"
)



# HEADER


st.title("🌪️ TerraNova FEMA Disaster Recovery Cost Prediction")

st.markdown(
    """
    ### Predict FEMA Disaster Recovery Costs Using Machine Learning

    TerraNova estimates FEMA disaster recovery costs using **declaration-stage disaster information**
    including disaster timing, location, and incident characteristics.

    The model was designed to minimize **target leakage** by excluding post-disaster funding variables,
    making predictions more realistic for operational decision-making.
    """
)

st.success(
    """
    🏆 **Best Model:** XGBoost Regressor  
    📈 **R² Score:** 0.8404  
    🎯 **Target Variable:** log_totalobligated
    """
)



# SIDEBAR INPUTS


st.sidebar.header("🌪️ Disaster Input Features")

fydeclared = st.sidebar.number_input(
    "Fiscal Year Declared",
    min_value=1950,
    max_value=2030,
    value=2024
)

state = st.sidebar.selectbox(
    "State",
    ["FL", "TX", "CA", "LA", "NY", "NC", "GA", "PR", "OR", "WA"]
)

declarationtype = st.sidebar.selectbox(
    "Declaration Type",
    ["DR", "EM", "FM"]
)

incidenttype = st.sidebar.selectbox(
    "Incident Type",
    [
        "Hurricane",
        "Fire",
        "Flood",
        "Severe Storm",
        "Snowstorm",
        "Tornado",
        "Drought",
        "Earthquake",
        "Coastal Storm",
        "Other"
    ]
)

designatedarea = st.sidebar.text_input(
    "Designated Area",
    value="Statewide"
)

avg_delay_days = st.sidebar.number_input(
    "Average Declaration Delay Days",
    min_value=0,
    max_value=365,
    value=30
)

declaration_delay_days = st.sidebar.number_input(
    "Declaration Delay Days",
    min_value=0,
    max_value=365,
    value=30
)

declaration_year = st.sidebar.number_input(
    "Declaration Year",
    min_value=1950,
    max_value=2030,
    value=2024
)

declaration_month = st.sidebar.slider(
    "Declaration Month",
    min_value=1,
    max_value=12,
    value=9
)

declaration_quarter = st.sidebar.selectbox(
    "Declaration Quarter",
    [1, 2, 3, 4],
    index=2
)

declaration_season = st.sidebar.selectbox(
    "Declaration Season",
    ["Winter", "Spring", "Summer", "Autumn"],
    index=3
)



# CREATE INPUT DATA


input_data = pd.DataFrame(
    [{
        "fydeclared": fydeclared,
        "state": state,
        "declarationtype": declarationtype,
        "incidenttype": incidenttype,
        "designatedarea": designatedarea,
        "avg_delay_days": avg_delay_days,
        "declaration_delay_days": declaration_delay_days,
        "declaration_year": declaration_year,
        "declaration_month": declaration_month,
        "declaration_quarter": declaration_quarter,
        "declaration_season": declaration_season
    }]
)


# DISPLAY INPUT PROFILE


st.subheader("📋 Input Disaster Profile")
st.dataframe(input_data, use_container_width=True)



# PREDICTION


st.markdown("---")

if st.button("🚀 Predict FEMA Recovery Cost"):

    predicted_log_cost = model.predict(input_data)[0]
    predicted_cost = np.expm1(predicted_log_cost)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Log Cost",
            value=round(float(predicted_log_cost), 4)
        )

    with col2:
        st.metric(
            label="Predicted Recovery Cost",
            value=f"${predicted_cost:,.2f}"
        )

    st.success("✅ Prediction completed successfully.")



# MODEL INFORMATION


st.markdown("---")

st.subheader("📊 Model Information")

col1, col2, col3 = st.columns(3)

col1.metric("Model", "XGBoost")
col2.metric("R²", "0.8404")
col3.metric("Target", "log_totalobligated")

st.info(
    "This model predicts FEMA disaster recovery costs using declaration-stage information while avoiding target leakage."
)



# FOOTER

st.markdown("---")

st.markdown(
    """
    **TerraNova FEMA Disaster Recovery Cost Prediction**  
    Built with Python, Scikit-Learn, XGBoost, Streamlit, and FastAPI.
    """
)