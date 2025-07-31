import streamlit as st
import requests

API_URL = "http://backend:8000/generate"


st.title("GAN ONNX Image Generator")

seed = st.number_input("Random Seed", min_value=0, step=1)

if st.button("Generate Image"):
    try:
        res = requests.post(API_URL, json={"seed": seed})
        if res.status_code == 200:
            data = res.json()
            filename = data["image"]

            # Construct URL to image served by FastAPI
            image_url = f"http://localhost:8000/outputs/{filename}"
            st.image(image_url, caption="Generated Image")
        else:
            st.error(f"Error from API: {res.status_code}")
    except Exception as e:
        st.error(f"Request failed: {e}")
