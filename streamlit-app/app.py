"""
AIC354 Assignment 2: End-to-End Machine Learning System
Zain Aftab (FA24-BSE-155), COMSATS University Islamabad, Lahore Campus
-----------------------------------------------------------------------
One Streamlit web app that serves both trained models:
  Tab 1: Titanic Passenger Survival Prediction (Support Vector Classifier)
  Tab 2: Heart Disease Prediction (Support Vector Classifier)

Each tab loads the .pkl model saved by the notebook and recreates the
same LabelEncoders (fitted on the same fixed category lists), so the
predictions match the notebook exactly.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AIC354 ML Prediction Systems", page_icon="🩺", layout="centered")

APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model(file_name):
    with open(APP_DIR / file_name, "rb") as f:
        return pickle.load(f)


# Same category lists used to train the LabelEncoders in the notebook
TITANIC_CATEGORIES = {
    "PClass": ["First", "Second", "Third"],
    "Gender": ["Male", "Female"],
    "Sibling": ["Zero", "One", "Two", "Three"],
    "Embarked": ["Southampton", "Cherbourg", "Queenstown"],
}

HEART_CATEGORIES = {
    "ChestPain": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    "Slope": ["Upsloping", "Flat", "Downsloping"],
    "MajorVessels": ["Zero", "One", "Two", "Three"],
    "Thal": ["Normal", "Fixed Defect", "Reversible Defect"],
}


@st.cache_resource
def build_encoders(categories):
    encoders = {}
    for column, values in categories.items():
        encoders[column] = LabelEncoder().fit(values)
    return encoders


def encode(user_input, encoders):
    encoded = user_input.copy()
    for column in user_input.columns:
        encoded[column] = encoders[column].transform(user_input[column])
    return encoded


st.title("Machine Learning Prediction Systems")
st.markdown(
    "**AIC354 Machine Learning Fundamentals, Assignment 2**  \n"
    "Zain Aftab (FA24-BSE-155), COMSATS University Islamabad, Lahore Campus"
)

titanic_tab, heart_tab = st.tabs(["🚢 Titanic Survival", "🩺 Heart Disease"])

# ---------------------------------------------------------------------------
# Tab 1: Titanic Passenger Survival Prediction
# ---------------------------------------------------------------------------
with titanic_tab:
    st.subheader("Titanic Passenger Survival Prediction")
    st.write("Pick a passenger's details. The trained Support Vector Classifier predicts whether they survived.")

    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class (PClass)", TITANIC_CATEGORIES["PClass"], index=2)
        gender = st.selectbox("Gender", TITANIC_CATEGORIES["Gender"])
    with col2:
        sibling = st.selectbox("Siblings/Spouses Aboard", TITANIC_CATEGORIES["Sibling"])
        embarked = st.selectbox("Port of Embarkation", TITANIC_CATEGORIES["Embarked"])

    if st.button("Predict Survival", type="primary", use_container_width=True):
        user_input = pd.DataFrame({
            "PClass": [pclass], "Gender": [gender], "Sibling": [sibling], "Embarked": [embarked],
        })
        encoded = encode(user_input, build_encoders(TITANIC_CATEGORIES))
        prediction = load_model("titanic_svc_trained_model.pkl").predict(encoded)[0]
        if prediction == 1:
            st.success("### ✅ SURVIVED")
        else:
            st.error("### ❌ NOT SURVIVED")
        with st.expander("Encoded feature vector sent to the model"):
            st.dataframe(encoded, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2: Heart Disease Prediction
# ---------------------------------------------------------------------------
with heart_tab:
    st.subheader("Heart Disease Prediction")
    st.write("Enter a patient's details. The trained Support Vector Classifier predicts whether heart disease is present.")

    col1, col2 = st.columns(2)
    with col1:
        chest_pain = st.selectbox("Chest Pain Type", HEART_CATEGORIES["ChestPain"], index=3)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", HEART_CATEGORIES["Slope"], index=1)
    with col2:
        major_vessels = st.selectbox("Major Vessels Coloured by Fluoroscopy", HEART_CATEGORIES["MajorVessels"])
        thal = st.selectbox("Thal (Thallium Stress Test)", HEART_CATEGORIES["Thal"])

    if st.button("Predict Heart Disease", type="primary", use_container_width=True):
        user_input = pd.DataFrame({
            "ChestPain": [chest_pain], "Slope": [slope],
            "MajorVessels": [major_vessels], "Thal": [thal],
        })
        encoded = encode(user_input, build_encoders(HEART_CATEGORIES))
        prediction = load_model("heart_svc_trained_model.pkl").predict(encoded)[0]
        if prediction == 1:
            st.error("### ⚠️ HEART DISEASE PRESENT")
        else:
            st.success("### ✅ NO HEART DISEASE")
        with st.expander("Encoded feature vector sent to the model"):
            st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for teaching purposes. Titanic: 100-row simplified dataset. "
    "Heart Disease: UCI Cleveland dataset (297 complete records, 4 input attributes). "
    "Not a medical diagnosis tool."
)
