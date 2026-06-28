# 🧠 BrainTumorXAI — Brain Tumor MRI Classification with Explainable AI

A CNN-based brain tumor classifier trained from scratch on MRI scans, with
**Grad-CAM explainability** to visualize *why* the model made each prediction,
and an interactive **Gradio** demo for real-time inference.

> ⚠️ **Disclaimer**: Built as an academic/portfolio project. Not validated
> for clinical use and must not be used for real medical diagnosis.

---

## 🎯 Highlights

- **Custom CNN from scratch** (no transfer learning) — 4-block convolutional
  architecture with BatchNorm, Dropout, and L2 regularization
- **Class-imbalance handling** via computed class weights
- **Explainable AI (Grad-CAM)** — heatmaps showing which MRI regions drove
  each prediction
- **FastAPI backend** — REST API (`/predict`, `/health`) serving the model,
  with the trained model loaded once at startup
- **Streamlit dashboard** — interactive frontend that calls the FastAPI
  backend over HTTP (decoupled frontend/backend, not a monolith)
- **Gradio demo** — a second, fully standalone interactive demo option
- **Free cloud deployment** — FastAPI on Render + Streamlit on Streamlit
  Community Cloud, model hosted on Hugging Face; no local machine required
- **Full evaluation suite** — confusion matrix, per-class precision/recall,
  training curves

## 📊 Results

| Metric | Score |
|---|---|
| Validation Accuracy | _fill in after training_ |
| Test Accuracy | _fill in after training_ |

See `outputs/confusion_matrix.png`, `outputs/training_curves.png`, and
`outputs/gradcam_samples/` for visual results.

## 🗂️ Dataset

This project combines **two Kaggle sources** into one balanced dataset of
**14,176 images** across 4 classes (`glioma`, `meningioma`, `notumor`, `pituitary`):

1. [Brain Tumor MRI Dataset (Masoud Nickparvar)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 7,023 images
2. [Brain Tumor MRI Dataset (deeppythonist)](https://www.kaggle.com/datasets/deeppythonist/brain-tumor-mri-dataset) — 7,153 images, predefined train/test split

`merge_datasets.py` combines both into the unified `data/Training/<class>/`
and `data/Testing/<class>/` structure the rest of the project expects. It
auto-detects folder naming differences between the two sources (e.g.
`glioma` vs `glioma_tumor`, `Training` vs `train`) and prevents filename
collisions by prefixing the second source's files.

## 🏗️ Project Structure

```
BrainTumorXAI/
├── config.py            # Paths, hyperparameters, class names
├── data_pipeline.py      # Dataset loading, folder normalization, augmentation
├── model.py              # CNN architecture
├── train.py               # Training loop with class weighting + callbacks
├── gradcam.py             # Grad-CAM explainability implementation
├── evaluate.py            # Test set metrics, confusion matrix, sample explanations
├── merge_datasets.py       # Combines the two Kaggle source datasets into data/
├── api.py                  # FastAPI backend (serves the model over REST)
├── streamlit_app.py        # Streamlit dashboard (calls api.py)
├── app.py                  # Gradio standalone demo (loads model directly)
├── requirements.txt          # Full deps (for local use / the FastAPI host)
├── requirements-streamlit.txt # Lightweight deps for Streamlit Cloud only
├── render.yaml              # Render deployment config for api.py
├── models/                 # Saved model + training history (generated)
└── outputs/                # Evaluation artifacts (generated)
```

## 🚀 Running it yourself

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the dataset (two sources, then merge)
Download both Kaggle datasets and extract them anywhere on disk, then run:
```bash
python merge_datasets.py --source1 /path/to/nickparvar_extracted --source2 /path/to/deeppythonist_extracted
```
This creates the combined structure under `data/`:
```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```
(`data_pipeline.py` also auto-normalizes any remaining `_tumor`-suffixed
folder names as a safety net, in case you skip the merge script and point
it at a single source directly.)

### 3. Train
```bash
python train.py
```

### 4. Evaluate
```bash
python evaluate.py
```

### 5. Launch the app — pick one

**Option A: FastAPI + Streamlit (decoupled backend/frontend)**
```bash
# Terminal 1 — start the API
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2 — start the dashboard (talks to the API above)
streamlit run streamlit_app.py
```
Visit `http://localhost:8501` for the dashboard, or `http://localhost:8000/docs`
for interactive API docs (Swagger UI).

**Option B: Gradio (standalone, single process)**
```bash
python app.py
```

## ☁️ Deploying to the cloud (no local running needed)

GitHub only stores code — it doesn't run anything. To get a live, publicly
accessible dashboard with nothing running on your own machine, you need
**two free hosts**: one for the FastAPI backend, one for the Streamlit
frontend. The model file itself isn't in this repo (too large for git), so
it's hosted separately on Hugging Face and downloaded automatically by the
API on startup.

### Step 1 — Upload your trained model to Hugging Face

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Click your profile → **New Model** (or **New Dataset**, either works) →
   give it any name, e.g. `brain-tumor-cnn`
3. On the model page, click **Files** → **Add file** → **Upload file** →
   select your `models/brain_tumor_cnn.keras`
4. Once uploaded, click on the file and copy its **download URL** — it
   looks like:
   `https://huggingface.co/<your-username>/brain-tumor-cnn/resolve/main/brain_tumor_cnn.keras`

Keep that URL — you'll paste it into Render in the next step.

### Step 2 — Deploy the FastAPI backend on Render

1. Push this repo to GitHub first (see the steps earlier in this conversation)
2. Create a free account at [render.com](https://render.com) and sign in with GitHub
3. Click **New** → **Web Service** → select your `BrainTumorXAI` repo
4. Render should auto-detect `render.yaml` and pre-fill the settings. If not, set manually:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Under **Environment Variables**, add:
   - Key: `MODEL_URL`
   - Value: the Hugging Face URL you copied in Step 1
6. Click **Create Web Service** and wait for the build to finish (a few minutes)
7. Once live, Render gives you a URL like `https://braintumorxai-api.onrender.com`
   — test it by visiting `https://braintumorxai-api.onrender.com/health`,
   you should see `{"status":"ok","model_loaded":true,...}`

> ⚠️ Render's free tier "spins down" after 15 minutes of inactivity and
> takes ~30-60 seconds to wake back up on the next request. This is normal
> for free hosting, not a bug — the first prediction after idle time will
> just be slow.

### Step 3 — Deploy the Streamlit dashboard on Streamlit Community Cloud

1. Create a free account at [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app** → select your `BrainTumorXAI` repo, branch `main`
3. Set **Main file path** to `streamlit_app.py`
4. Click **Advanced settings** → set **Requirements file** to `requirements-streamlit.txt`
   (this skips installing TensorFlow, which the dashboard never needs —
   keeps the build fast and avoids hitting free-tier memory limits)
5. Under **Secrets**, add:
   ```
   BRAIN_TUMOR_API_URL = "https://braintumorxai-api.onrender.com"
   ```
   (use your actual Render URL from Step 2)
6. Click **Deploy**

You'll get a public URL like `https://your-app.streamlit.app` — that's
your live dashboard, shareable with anyone, with nothing running on your PC.

### Updating after changes

Both Render and Streamlit Cloud auto-redeploy whenever you push new commits
to the connected GitHub branch — `git push` is all you need afterward.

## 🧪 Methodology

The CNN uses 4 convolutional blocks (32→64→128→256 filters) with
BatchNormalization and increasing Dropout, followed by a dense classifier
head with L2 regularization. Data augmentation (flip, rotation, zoom,
contrast, brightness) and computed class weights address the dataset's
class imbalance. Grad-CAM is computed on the final convolutional layer to
produce localization heatmaps explaining each prediction.

## 📄 License

MIT
