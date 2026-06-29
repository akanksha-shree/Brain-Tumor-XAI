"""
Grad-CAM implementation for Final Model.h5
Compatible with TensorFlow/Keras 3
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib

from PIL import Image

from config import CLASS_NAMES


# ------------------------------------------------------------------
# Model constants
# ------------------------------------------------------------------

IMG_SIZE = 128

# Last Conv2D layer of Final Model.h5
LAST_CONV_LAYER = "conv2d_7"


# ------------------------------------------------------------------
# Image preprocessing
# Matches the Kaggle notebook
# ------------------------------------------------------------------

def preprocess_image(pil_image):
    """
    Converts a PIL image into the exact format used during training.

    Returns
    -------
    img_array : np.ndarray
        Shape (1,128,128,3)
    display_image : np.ndarray
        RGB uint8 image for GradCAM overlay
    """

    img = np.array(pil_image.convert("RGB"))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = np.stack([img] * 3, axis=-1)

    display = img.copy()

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img, display


# ------------------------------------------------------------------
# Build GradCAM model
# Compatible with Sequential models in Keras 3
# ------------------------------------------------------------------

def create_gradcam_model(model, layer_name=LAST_CONV_LAYER):

    # Make sure model is built
    if not model.built:
        dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
        model(dummy)

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = inputs
    conv_output = None

    for layer in model.layers:

        x = layer(x)

        if layer.name == layer_name:
            conv_output = x

    if conv_output is None:
        raise ValueError(
            f"Layer '{layer_name}' not found.\n"
            f"Available layers:\n"
            f"{[layer.name for layer in model.layers]}"
        )

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[conv_output, x]
    )

    return grad_model
# ------------------------------------------------------------------
# Generate Grad-CAM heatmap
# ------------------------------------------------------------------

def make_gradcam_heatmap(img_array, model, layer_name=LAST_CONV_LAYER):
    """
    Returns
    -------
    heatmap : np.ndarray
    pred_index : int
    probabilities : np.ndarray
    """

    grad_model = create_gradcam_model(model, layer_name)

    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_tensor)

        pred_index = tf.argmax(predictions[0])

        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise RuntimeError(
            "Could not compute Grad-CAM gradients."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-10

    return (
        heatmap.numpy(),
        int(pred_index.numpy()),
        predictions[0].numpy()
    )


# ------------------------------------------------------------------
# Overlay heatmap on MRI
# ------------------------------------------------------------------

def overlay_heatmap(display_img, heatmap, alpha=0.40):

    heatmap = cv2.resize(
        heatmap,
        (display_img.shape[1], display_img.shape[0])
    )

    heatmap_uint8 = np.uint8(255 * heatmap)

    jet = matplotlib.colormaps["jet"]

    jet_colors = jet(np.arange(256))[:, :3]

    jet_heatmap = jet_colors[heatmap_uint8]

    jet_heatmap = np.uint8(jet_heatmap * 255)

    overlay = (
        alpha * jet_heatmap +
        (1 - alpha) * display_img
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(overlay)
# ------------------------------------------------------------------
# Main prediction function used by FastAPI
# ------------------------------------------------------------------

def explain_prediction(pil_image, model):
    """
    Parameters
    ----------
    pil_image : PIL.Image
    model : tf.keras.Model

    Returns
    -------
    predicted_class : str
    confidence : float
    probabilities : dict
    overlay : PIL.Image
    """

    # Preprocess image exactly like training
    img_array, display_img = preprocess_image(pil_image)

    # Ensure model is built (Keras 3 compatibility)
    if not model.built:
        model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)))

    # Prediction
    predictions = model.predict(img_array, verbose=0)[0]

    pred_index = int(np.argmax(predictions))

    confidence = float(predictions[pred_index])

    predicted_class = CLASS_NAMES[pred_index]

    probabilities = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    # Try Grad-CAM
    try:
        heatmap, _, _ = make_gradcam_heatmap(
            img_array,
            model,
            LAST_CONV_LAYER
        )

        overlay = overlay_heatmap(display_img, heatmap)

    except Exception as e:

        print("Grad-CAM failed:", e)

        # Return original image instead of crashing API
        overlay = Image.fromarray(display_img)

    return (
        predicted_class,
        confidence,
        probabilities,
        overlay,
    )