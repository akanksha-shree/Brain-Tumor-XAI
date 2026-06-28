"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for explainability.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (2017).

Produces a heatmap over the input image showing which regions most
influenced the model's predicted class — used here to highlight the
tumor region (or lack thereof) that drove the CNN's decision.
"""
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from PIL import Image

from config import LAST_CONV_LAYER_NAME, IMG_SIZE, CLASS_NAMES


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER_NAME, pred_index=None):
    """
    img_array: preprocessed batch of shape (1, H, W, 3), values in [0, 255]
               (the model's own Rescaling layer handles normalization).
    Returns: heatmap as a (h, w) numpy array, values in [0, 1].
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), predictions[0].numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """
    original_img: PIL Image or numpy array (H, W, 3), uint8, NOT normalized.
    heatmap: (h, w) array from make_gradcam_heatmap, values in [0, 1].
    Returns: PIL Image with heatmap overlaid.
    """
    if isinstance(original_img, Image.Image):
        original_img = np.array(original_img.convert("RGB"))

    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap_resized).resize(
        (original_img.shape[1], original_img.shape[0])
    )
    heatmap_resized = np.array(heatmap_img)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    superimposed = jet_heatmap * alpha + original_img * (1 - alpha)
    superimposed = np.uint8(np.clip(superimposed, 0, 255))
    return Image.fromarray(superimposed)


def explain_prediction(pil_image, model, alpha=0.4):
    """
    End-to-end: takes a raw PIL image, returns (predicted_class_name,
    confidence, dict of all class probabilities, Grad-CAM overlay PIL image).
    """
    img_resized = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img_resized).astype("float32"), axis=0)

    heatmap, pred_index, probs = make_gradcam_heatmap(img_array, model)
    overlay = overlay_heatmap(img_resized, heatmap, alpha=alpha)

    predicted_class = CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index])
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return predicted_class, confidence, prob_dict, overlay
