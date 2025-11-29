import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

API_BASE = "http://localhost:8000/api/v1"
PREDICT_URL = f"{API_BASE}/predict"
PREDICT_THRESHOLD_URL = f"{API_BASE}/predict/threshold"
HEALTH_URL = f"{API_BASE}/health"
INFO_URL = f"{API_BASE}/model/info"

st.set_page_config(
    page_title="GitNos API UI",
    layout="wide",
)

if "show_menu" not in st.session_state:
    st.session_state.show_menu = False

if "current_page" not in st.session_state:
    st.session_state.current_page = "Welcome"

menu_col1, menu_col2 = st.columns([0.05, 0.95])
with menu_col1:
    if st.button("☰", key="menu_toggle"):
        st.session_state.show_menu = not st.session_state.show_menu
with menu_col2:
    st.markdown("Menu" if st.session_state.show_menu else "")

if st.session_state.show_menu:
    st.markdown("Navigation")
    col1, col2, col3 = st.columns(3)
    st.write("")
    if st.button("Home"):
        st.session_state.current_page = "Home"
    if st.button("Info"):
        st.session_state.current_page = "Info"
    if st.button("Welcome"):
        st.session_state.current_page = "Welcome"

if st.session_state.current_page == "Welcome":
    st.title("Welcome to GitNos API")
    st.markdown("""
        This interface allows you to Upload one or more images for digit prediction
    """)

elif st.session_state.current_page == "Home":
    st.title("GitNos Predictor")
    st.markdown("Upload one or more images for prediction using the trained MNIST model.")

    threshold = st.text_input("Optional Confidence Threshold (0.0 - 1.0):", "")

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
            for j, file in enumerate(uploaded_files[i:i + num_cols]):
                img = Image.open(file)
                cols[j].image(img, caption=file.name, use_container_width=True, width=100)

        if st.button("Run Predictions"):
        
            use_threshold = False
            if threshold.strip():
                try:
                    threshold_value = float(threshold)
                    if not (0.0 <= threshold_value <= 1.0):
                        st.error("Threshold must be between 0.0 and 1.0")
                        st.stop()
                    use_threshold = True
                except ValueError:
                    st.error("Invalid threshold value. Enter a number between 0.0 and 1.0.")
                    st.stop()

            try:
                if use_threshold:
                    st.info(f"Using threshold: {threshold_value}")
                    results = []
                    for file in uploaded_files:
                        buffered = BytesIO()
                        img = Image.open(file)
                        img.save(buffered, format="PNG")
                        img_b64 = base64.b64encode(buffered.getvalue()).decode()

                        payload = {"image": img_b64, "threshold": threshold_value}
                        response = requests.post(PREDICT_THRESHOLD_URL, json=payload)
                        if response.status_code == 200:
                            results.append(response.json())
                        else:
                            st.error(f"Error for {file.name}: {response.text}")
                    st.subheader("Threshold Prediction Results")
                    st.json(results)
                else:
                    files = [("images", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
                    response = requests.post(PREDICT_URL, files=files)
                    if response.status_code == 200:
                        st.subheader("Prediction Results from API")
                        st.json(response.json())
                    else:
                        st.error(f"Prediction failed. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

elif st.session_state.current_page == "Info":
    st.title("Model Information & Health")

    col_buttons, col_result = st.columns([1, 3])

    with col_buttons:
        st.markdown("### Actions")
        st.markdown("Click to fetch details:")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🩺 Check Model Health", key="health"):
            try:
                response = requests.get(HEALTH_URL)
                if response.status_code == 200:
                    st.session_state.result_data = response.json()
                    st.session_state.result_source = "health"
                else:
                    st.session_state.result_data = {"error": f"Status code: {response.status_code}"}
            except Exception as e:
                st.session_state.result_data = {"error": str(e)}

        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Get Model Info", key="info"):
            try:
                response = requests.get(INFO_URL)
                if response.status_code == 200:
                    st.session_state.result_data = response.json()
                    st.session_state.result_source = "info"
                else:
                    st.session_state.result_data = {"error": f"Status code: {response.status_code}"}
            except Exception as e:
                st.session_state.result_data = {"error": str(e)}

    with col_result:
        st.markdown("### Result Display")
        st.markdown("---")
        if "result_data" in st.session_state:
            st.json(st.session_state.result_data)
        else:
            st.info("Results will appear here after clicking a button.")






