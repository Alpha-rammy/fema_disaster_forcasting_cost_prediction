import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st



# PATHS


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fema_cost_model.pkl")



# LOAD MODE

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()



# PAGE CONFIG


st.set_page_config(
    page_title="TerraNova FEMA Cost Predictor",
    page_icon="🌪️",
    layout="wide"
)



# APP HEADER

st.title("🌪️ TerraNova FEMA Disaster Cost Prediction")
st.markdown(
    """
    This app predicts estimated FEMA disaster recovery cost using disaster declaration,
    timing, location, and incident characteristics.
    """
)



# SIDEBAR INPUTS


st.sidebar.header("Disaster Input Features")

fydeclared = st.sidebar.number_input(
    "Fiscal Year Declared",
    min_value=1950,
    max_value=2030,
    value=2024
)

state = st.sidebar.selectbox(
    "State",
    ["Fl", "Tx", "Ca", "La", "Ny", "Nc", "Ga", "Pr", "Or", "Wa"]
)

declarationtype = st.sidebar.selectbox(
    "Declaration Type",
    ["Dr", "Em", "Fm"]
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
    "avg_delay_days",
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
    index=2
)



# CREATE INPUT DATA


input_data = pd.DataFrame(
    [{
        "fydeclared": fydeclared,
        "state": state,
        "declarationtype": declarationtype,
        "incidenttype": incidenttype,
        "designatedarea": designatedarea,
        "Average Declaration Delay": "average declaration delay",
        "declaration_delay_days": declaration_delay_days,
        "declaration_year": declaration_year,
        "declaration_month": declaration_month,
        "declaration_quarter": declaration_quarter,
        "declaration_season": declaration_season
    }]
)



# DISPLAY INPUTS


st.subheader("Input Disaster Profile")
st.dataframe(input_data, use_container_width=True)



# PREDICTION


if st.button("Predict FEMA Recovery Cost"):

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

    st.success("Prediction completed successfully.")



# MODEL INFO


st.markdown("---")

st.subheader("Model Information")

st.markdown(
    """
    **Final Model:** XGBoost Regressor  
    **Target Variable:** log_totalobligated  
    **Best R² Score:** 0.8456  
    **Deployment:** Streamlit + FastAPI-ready  
    """
)