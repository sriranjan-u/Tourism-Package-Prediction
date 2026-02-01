import streamlit as st
import pandas as pd
import os
import joblib
from huggingface_hub import hf_hub_download

# --- Configuration ---
# Updated to your specific repository and model filename
MODEL_REPO_ID = "Sriranjan/Tourism-Package-Pred"
MODEL_FILENAME = "models/tourism_xgb_model.joblib"

# --- Load Model ---
@st.cache_resource
def load_model():
    """Loads the trained model pipeline from Hugging Face Hub."""
    try:
        # Download from the dataset-type repo where your Action uploads it
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID, 
            filename=MODEL_FILENAME,
            repo_type="dataset"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Tourism Predictor", layout="wide")
st.title("✈️ Wellness Tourism Package Prediction")
st.write("""
This app predicts if a customer will purchase a package based on their profile.
Enter the details below to generate a prediction.
""")

if model is None:
    st.warning("Model not found. Please check your Hugging Face repository.")
    st.stop()

# --- Input UI ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Personal Details")
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    income = st.number_input("Monthly Income", 0, 100000, 25000)
    passport = st.selectbox("Has Passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns Car?", ["No", "Yes"])

with col2:
    st.header("🏨 Travel Preferences")
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
    prop_star = st.slider("Property Rating Preference", 3, 5, 3)
    num_trips = st.number_input("Number of Trips", 0, 20, 1)
    num_visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
    children = st.number_input("Number of Children", 0, 5, 0)

with col3:
    st.header("📞 Interaction")
    contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
    pitch_dur = st.number_input("Duration of Pitch (min)", 1, 120, 15)
    satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    followups = st.number_input("Number of Follow-ups", 0, 10, 3)

# --- Prediction Logic ---
if st.button("Predict Purchase Probability", type="primary"):
    # Create DataFrame matching the raw feature names used in your preprocess.py
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": contact,
        "CityTier": city_tier,
        "DurationOfPitch": pitch_dur,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_visiting,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": prop_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income,
        "PitchSatisfactionScore": satisfaction
    }])

    # The pipeline handles scaling and one-hot encoding automatically
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.success(f"### Likely to Purchase! (Confidence: {probability:.2%})")
    else:
        st.info(f"### Unlikely to Purchase. (Confidence: {1-probability:.2%})")
