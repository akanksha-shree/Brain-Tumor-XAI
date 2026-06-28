import gradio as gr
import tensorflow as tf
from config import MODEL_PATH, CLASS_NAMES
from gradcam import explain_prediction

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")


def predict_and_explain(image):
    if image is None:
        return None, "Please upload an MRI image.", {}

    predicted_class, confidence, probs, overlay = explain_prediction(image, model)

    label = f"{predicted_class.upper()} ({confidence * 100:.1f}% confidence)"
    return overlay, label, probs


with gr.Blocks(title="Brain Tumor Classifier — Explainable AI") as demo:
    gr.Markdown(
        """
        # 🧠 Brain Tumor MRI Classifier with Explainable AI (Grad-CAM)

        Upload a brain MRI scan to classify it as **glioma**, **meningioma**,
        **pituitary tumor**, or **no tumor**. The highlighted (red/yellow)
        regions in the Grad-CAM overlay show which parts of the scan most
        influenced the model's decision.

        ⚠️ **Disclaimer**: This is a student/portfolio project for educational
        purposes only. It is **not** a medical diagnostic tool and must not
        be used for real clinical decisions.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload MRI Scan")
            submit_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Grad-CAM Explanation")
            output_label = gr.Textbox(label="Prediction")
            output_probs = gr.Label(label="Class Probabilities", num_top_classes=4)

    submit_btn.click(
        fn=predict_and_explain,
        inputs=input_image,
        outputs=[output_image, output_label, output_probs]
    )

    gr.Markdown("### Class reference: " + ", ".join(CLASS_NAMES))

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
