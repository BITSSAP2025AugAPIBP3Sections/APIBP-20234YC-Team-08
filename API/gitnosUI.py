import streamlit as st
import requests
from PIL import Image

API_URL = "http://localhost:8000/api/v1/predict"
st.title("Multi-Image Uploader & Predictor")

uploaded_files = st.file_uploader(
    "Upload one or more .PNG or .JPEG files",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("Uploaded Image Preview")

    num_cols = 10 
    for i in range(0, len(uploaded_files), num_cols):
        cols = st.columns(num_cols)
        for j, file in enumerate(uploaded_files[i:i+num_cols]):
            img = Image.open(file)
            cols[j].image(img, caption=file.name, use_container_width=True, width=100) 

    if st.button("Run Predictions"):
        files = [("images", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            st.subheader("Prediction Results from API")
            st.json(response.json())
        else:
            st.error(f"Prediction failed. Status code: {response.status_code}")




