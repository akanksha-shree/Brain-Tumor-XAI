"""
Streamlit dashboard for the Brain Tumor XAI classifier.

This is a CLIENT — it does not load the model itself. It calls the
FastAPI backend (api.py) over HTTP, the same way a real production
frontend would. Run/deploy the API first, then this.

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000 &
    streamlit run streamlit_app.py

Cloud deployment (Streamlit Community Cloud):
    Set the BRAIN_TUMOR_API_URL secret/env var to your deployed FastAPI
    URL (e.g. https://your-app.onrender.com). See README.md for the full
    deployment walkthrough.

In Colab: see the notebook, which handles both processes + tunneling.
"""
import base64
import io
import os
import time

import requests
import streamlit as st
from PIL import Image

from config import CLASS_NAMES

API_URL = os.environ.get("BRAIN_TUMOR_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="BrainTumorXAI Dashboard",
    page_icon="🧠",
    layout="wide",
)


with st.sidebar:
    st.title("🧠 BrainTumorXAI")
    st.markdown(
        "A custom CNN brain tumor MRI classifier with **Grad-CAM** "
        "explainability, served via a FastAPI backend."
    )
    st.divider()
    st.markdown("**Classes detected:**")
    for c in CLASS_NAMES:
        st.markdown(f"- {c}")
    st.divider()

    st.markdown("**Backend status**")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.ok:
            data = resp.json()
            if data.get("model_loaded"):
                st.success("API connected · model loaded")
            else:
                st.warning("API connected · model NOT loaded")
        else:
            st.error(f"API returned status {resp.status_code}")
    except requests.exceptions.RequestException:
        st.error("Cannot reach API. Is `api.py` running?")

    st.caption(f"API endpoint: `{API_URL}`")
    st.divider()
    st.caption(
        "⚠️ Educational/portfolio project. Not a medical diagnostic tool."
    )


st.title("Brain Tumor MRI Classification — Explainable AI")
st.markdown(
    "Upload a brain MRI scan below. The model classifies it into one of "
    "four categories and highlights (via Grad-CAM) the regions that most "
    "influenced its decision."
)

col_upload, col_results = st.columns([1, 1.3])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload MRI scan (JPEG or PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded scan", use_container_width=True)
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
    else:
        analyze_clicked = False
        st.info("Upload an image to get started.")

with col_results:
    if uploaded_file is not None and analyze_clicked:
        with st.spinner("Running inference..."):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                start = time.perf_counter()
                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                round_trip_ms = (time.perf_counter() - start) * 1000
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API: {e}")
                response = None

        if response is not None:
            if response.status_code == 200:
                result = response.json()

                st.subheader(f"Prediction: `{result['predicted_class'].upper()}`")
                st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

                overlay_bytes = base64.b64decode(result["gradcam_overlay_base64"])
                overlay_img = Image.open(io.BytesIO(overlay_bytes))
                st.image(
                    overlay_img,
                    caption="Grad-CAM explanation (highlighted regions drove the prediction)",
                    use_container_width=True,
                )

                st.markdown("**Class probabilities:**")
                probs = result["probabilities"]
                for cls in CLASS_NAMES:
                    st.progress(probs.get(cls, 0.0), text=f"{cls}: {probs.get(cls, 0.0) * 100:.1f}%")

                st.caption(
                    f"Server inference: {result['inference_time_ms']:.1f} ms · "
                    f"Round trip: {round_trip_ms:.1f} ms"
                )
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(f"API error ({response.status_code}): {detail}")
    elif uploaded_file is None:
        st.empty()
