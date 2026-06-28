# 🧠 BrainTumorXAI

An Explainable AI (XAI) based Brain Tumor MRI Classification system built using a custom CNN, FastAPI backend, and Streamlit frontend.

The application classifies brain MRI scans into four tumor categories and provides visual explanations using **Grad-CAM**, allowing users to understand which regions of the MRI influenced the model's prediction.

> **Educational Project:** This project is intended for research, learning, and portfolio purposes only. It is **not** a medical diagnostic tool.

---

# Features

- Custom CNN trained for Brain MRI classification
- FastAPI backend for inference
- Streamlit interactive web interface
- Grad-CAM visual explanations
- Confidence score for predictions
- REST API
- Modular project structure
- Ready for local deployment and cloud deployment

---

# Classes

The model predicts one of the following classes:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

# Project Structure

```
BrainTumorXAI/
│
├── api.py
├── app.py
├── config.py
├── gradcam.py
├── requirements.txt
├── README.md
│
├── models/
│   └── Final Model.h5
├── images/
│   └── Results
│
├── outputs/
│
├── data/
│   ├── Training/
│   └── Testing/
│
└── assets/
```

---

# Model

- Architecture: Custom CNN
- Framework: TensorFlow / Keras
- Input Size:

```
128 × 128 × 3
```

Classes:

```
Glioma
Meningioma
Pituitary
No Tumor
```

---

# Explainability

The project uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the regions of the MRI image that most influenced the model's prediction.

The output includes:

- Predicted class
- Confidence score
- Probability distribution
- Grad-CAM heatmap
- Heatmap overlay on the MRI image

---

# Tech Stack

### Backend

- FastAPI
- TensorFlow
- Keras
- NumPy
- Pillow
- OpenCV

### Frontend

- Streamlit

### Explainability

- Grad-CAM

---

# Installation

Clone the repository

```bash
git clone https://github.com/akanksha-shree/BrainTumorXAI.git

cd BrainTumorXAI
```

Create virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Download Model

Place the trained model inside

```
models/
```

The file should be

```
models/
    Final Model.h5
```

---

# Run Backend

```bash
python api.py
```

Server starts at

```
http://127.0.0.1:8000
```

---

# Run Frontend

In another terminal

```bash
streamlit run app.py
```

Open

```
http://localhost:8501
```

---

# API Endpoints

### Health Check

```
GET /
```

or

```
GET /health
```

Returns

```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": [
    "glioma",
    "meningioma",
    "pituitary",
    "no_tumor"
  ]
}
```

---

### Predict

```
POST /predict
```

Upload

- JPG
- JPEG
- PNG

Returns

```json
{
    "predicted_class":"glioma",
    "confidence":0.97,
    "probabilities":{},
    "gradcam_overlay_base64":"..."
}
```

---

# Dataset

Brain MRI dataset containing four classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

Dataset is used only for educational and research purposes.

---

# Example Workflow

1. Upload MRI image

↓

2. CNN predicts tumor class

↓

3. Confidence score generated

↓

4. Grad-CAM computes activation map

↓

5. Heatmap overlaid on MRI

↓

6. Results displayed in Streamlit

---

# Future Improvements

- SHAP explanations
- LIME explanations
- Multi-model comparison
- Docker support
- Cloud deployment
- User authentication
- Prediction history
- Model versioning

---

# Requirements

- Python 3.10+
- TensorFlow
- FastAPI
- Streamlit
- OpenCV
- Pillow
- NumPy
- Matplotlib

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

# Disclaimer

This software is intended solely for educational, research, and demonstration purposes.

It must **not** be used as a substitute for professional medical advice, diagnosis, or treatment.

---

# Author

**Akanksha Shree**

---

# License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to deal in the Software without restriction.