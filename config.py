import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "brain_tumor_cnn.keras")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
GRADCAM_DIR = os.path.join(OUTPUTS_DIR, "gradcam_samples")
EVAL_REPORT_PATH = os.path.join(OUTPUTS_DIR, "evaluation_report.json")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
TRAINING_CURVES_PATH = os.path.join(OUTPUTS_DIR, "training_curves.png")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42

# Name of the last Conv2D layer in model.py — used by Grad-CAM.
# If you change the architecture, update this to match the new last conv layer's name.
LAST_CONV_LAYER_NAME = "last_conv"
