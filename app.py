import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Page config
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏡",
    layout="wide"
)


# Load saved components
model = joblib.load('linear_regression_pca.pkl')
pca = joblib.load('pca_transformer.pkl')
scaler = joblib.load('scaler.pkl')


# Sidebar
st.sidebar.title("⚙️ App Controls")
st.sidebar.info("Adjust the house features and predict the price in real time.")


# Main title
st.markdown(
    """
    <h1 style='text-align: center;'>🏠 California Housing Price Predictor</h1>
    <p style='text-align: center;'>Interactive ML app using PCA + Linear Regression</p>
    """,
    unsafe_allow_html=True
)

st.divider()


# Layout
col1, col2 = st.columns(2)


with col1:
    st.subheader("📥 Input House Features")
    MedInc = st.slider("Median Income", 0.0, 15.0, 3.0)
    HouseAge = st.slider("House Age", 1.0, 60.0, 20.0)
    AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
    AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
    Population = st.slider("Population", 1.0, 5000.0, 1000.0)
    AveOccup = st.slider("Average Occupancy", 1.0, 6.0, 3.0)
    Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
    Longitude = st.slider("Longitude", -124.0, -114.0, -119.0)


with col2:
    st.subheader("📊 Feature Overview")
    input_df = pd.DataFrame({
        "Feature": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
        "Value": [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    })
    st.dataframe(input_df, use_container_width=True)

st.divider()


# Prediction section
st.subheader("🔮 Prediction")


if st.button("🚀 Predict House Value", use_container_width=True):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
