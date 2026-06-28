"""
FastAPI backend for the Brain Tumor XAI classifier.

Loads the trained CNN once at startup and exposes REST endpoints for
prediction (with Grad-CAM explanation). Designed to be the single backend
consumed by both the Streamlit dashboard and, optionally, the Gradio app.

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

For cloud deployment (Render, Railway, etc.), the model file is normally
NOT in the git repo (too large). Set the MODEL_URL environment variable to
a direct-download link (e.g. a Hugging Face Hub file URL) and this file
will download it to disk on first startup. If MODEL_URL is not set, it
falls back to loading from the local MODEL_PATH (for local/Colab use).

In Colab, see the notebook — it runs this in a background thread since
Colab cells block otherwise.
"""
import base64
import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from config import MODEL_PATH, MODELS_DIR, CLASS_NAMES, IMG_SIZE
from gradcam import explain_prediction

MODEL_URL = os.environ.get("MODEL_URL", "").strip()


def _ensure_model_on_disk():
    """
    If MODEL_URL is set and the model isn't already on disk, download it.
    Cloud hosts wipe disk on each redeploy, so this re-downloads each cold
    start — that's expected and fine for a model in the tens-of-MB range.
    """
    if os.path.exists(MODEL_PATH):
        print(f"Model already present at {MODEL_PATH}")
        return

    if not MODEL_URL:
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH} and MODEL_URL is not set. "
            f"Either place a trained model at that path, or set the MODEL_URL "
            f"environment variable to a direct-download link."
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"📥 Downloading model from MODEL_URL...")
    response = requests.get(MODEL_URL, stream=True, timeout=300)
    response.raise_for_status()
    total = 0
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)
    print(f"✅ Downloaded {total / (1024 * 1024):.1f} MB to {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Model is loaded once at startup and kept in memory (lifespan), not
# reloaded per-request — reloading a Keras model on every call is slow
# and is a common mistake in student API projects.
# ---------------------------------------------------------------------------
ml_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("📦 Preparing model...")
    _ensure_model_on_disk()
    ml_state["model"] = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded.")
    yield
    ml_state.clear()


app = FastAPI(
    title="BrainTumorXAI API",
    description="CNN brain tumor MRI classifier with Grad-CAM explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive CORS so the Streamlit/Colab frontend (running on a different
# port/origin) can call this without browser CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict
    gradcam_overlay_base64: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list


def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded="model" in ml_state,
        classes=CLASS_NAMES,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return health_check()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if "model" not in ml_state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload JPEG or PNG.",
        )

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    start = time.perf_counter()
    predicted_class, confidence, probs, overlay = explain_prediction(img, ml_state["model"])
    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probs,
        gradcam_overlay_base64=_pil_to_base64(overlay),
        inference_time_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
