import os
import tempfile

import gradio as gr
import requests
import tensorflow as tf
from PIL import Image

from gradcam import explain_prediction

MODEL_URL = "https://huggingface.co/akanksha31/brain-tumor-xai-model/resolve/main/Final%20Model.h5"

MODEL_PATH = os.path.join(tempfile.gettempdir(), "Final Model.h5")

model = None


def load_model():
    global model

    if model is not None:
        return model

    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL, stream=True)

        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    return model


def predict(image):

    model = load_model()

    predicted_class, confidence, probs, overlay = explain_prediction(
        image,
        model,
    )

    return (
        predicted_class,
        f"{confidence:.2%}",
        overlay,
    )


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
        gr.Image(label="Grad-CAM"),
    ],
    title="Brain Tumor MRI Classification",
    description="CNN + Grad-CAM Explainability",
)

demo.launch()