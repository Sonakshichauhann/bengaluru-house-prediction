import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bengaluru House Price Predictor", layout="centered")
st.title("🏠 Bengaluru House Price Predictor")

@st.cache_data(show_spinner=False)
def load_artifacts():
    model = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    model_locations = joblib.load("model_locations.pkl")
    return model, model_columns, model_locations

model, model_columns, model_locations = load_artifacts()

# ------------------------------
# User Inputs
# ------------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        # Dropdown for location
        location = st.selectbox("Select Location", options=sorted(model_locations))
        total_sqft = st.number_input("Total sqft", min_value=100.0, value=1200.0, step=1.0)
    with col2:
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)
        bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)

    submitted = st.form_submit_button("Predict")

# ------------------------------
# Prediction
# ------------------------------
if submitted:
    bath_per_bhk = (bath / bhk) if bhk != 0 else 0.0

    x = pd.DataFrame([[total_sqft, bhk, bath, bath_per_bhk, location]],
                     columns=['total_sqft','BHK','bath','bath_per_bhk','location'])

    # One-hot encode location
    x = pd.get_dummies(x, columns=['location'], drop_first=True)
    x = x.reindex(columns=model_columns, fill_value=0)

    pred = model.predict(x)[0]
    st.metric("Estimated Price (in Lakhs)", f"{pred:.2f}")

    with st.expander("Model input vector"):
        st.write(x.iloc[0])
