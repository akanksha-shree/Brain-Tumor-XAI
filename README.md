# 🧠 Brain Tumor MRI Classifier with Explainable AI (Grad-CAM)

An Explainable AI (XAI) based Brain Tumor MRI Classification system built using a custom Convolutional Neural Network (CNN). The application classifies brain MRI scans into four categories and provides visual explanations using **Grad-CAM**, allowing users to understand which regions of the MRI influenced the model's prediction.

> **Educational Project:** This project is intended for learning, research, and portfolio purposes only. It is **not** a medical diagnostic tool.

---

# 🚀 Live Demo

**Try the application online:**

👉 **https://huggingface.co/spaces/akanksha31/brain-tumor-xai**

---

# ✨ Features

* Brain MRI classification using a custom CNN
* Explainable AI with Grad-CAM heatmaps
* Interactive Gradio web application
* Confidence score and class probabilities
* Automatic model download from Hugging Face Hub
* TensorFlow/Keras implementation
* Modular and well-structured codebase
* Ready for local use and cloud deployment

---

# 🧠 Predicted Classes

The model classifies MRI scans into one of the following categories:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

---

# 📂 Project Structure

```text
BrainTumorXAI/
│
├── api.py
├── app.py
├── config.py
├── gradcam.py
├── model.py
├── train.py
├── evaluate.py
├── streamlit_app.py
├── requirements.txt
├── README.md
│
├── models/
├── outputs/
├── data/
└── images/
```

---

# 🏗 Model

* **Architecture:** Custom CNN
* **Framework:** TensorFlow / Keras
* **Input Size:** 128 × 128 × 3
* **Output Classes:** 4

---

# 🔍 Explainable AI

The project uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to highlight the image regions that most influenced the model's prediction.

Each prediction includes:

* Predicted class
* Confidence score
* Probability distribution
* Grad-CAM heatmap
* Heatmap overlay on the original MRI

---

# 💻 Tech Stack

### Machine Learning

* TensorFlow
* Keras
* NumPy
* OpenCV
* Pillow
* Matplotlib

### Backend

* FastAPI

### Frontend

* Gradio
* Streamlit (local interface)

### Explainability

* Grad-CAM

---

# 🚀 Running Locally

Clone the repository:

```bash
git clone https://github.com/akanksha-shree/Brain-Tumor-XAI.git
cd Brain-Tumor-XAI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
python api.py
```

Run the frontend:

```bash
streamlit run streamlit_app.py
```

---

# 🌐 API Endpoints

## Health Check

```
GET /
```

or

```
GET /health
```

## Prediction

```
POST /predict
```

Supported image formats:

* JPG
* JPEG
* PNG

Returns:

* Predicted class
* Confidence score
* Class probabilities
* Grad-CAM visualization

---

# 📊 Workflow

1. Upload a brain MRI image.
2. The CNN predicts the tumor category.
3. Confidence scores are computed.
4. Grad-CAM generates an attention heatmap.
5. The heatmap is overlaid on the MRI.
6. Results are displayed to the user.

---

# 🔮 Future Improvements

* SHAP explanations
* LIME explanations
* Docker containerization
* User authentication
* Prediction history
* Model versioning
* Additional CNN architectures

---

# 📦 Requirements

* Python 3.10+
* TensorFlow
* FastAPI
* Gradio
* Streamlit
* OpenCV
* Pillow
* NumPy
* Matplotlib

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

# ⚠️ Disclaimer

This software is intended solely for educational, research, and demonstration purposes.

It must **not** be used as a substitute for professional medical advice, diagnosis, or treatment.

---

#  Author

**Akanksha Shree**

---

---

⭐ If you found this project useful, consider giving the repository a star!
