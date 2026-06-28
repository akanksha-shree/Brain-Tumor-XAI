import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

from config import (
    MODEL_PATH, CLASS_NAMES, OUTPUTS_DIR, GRADCAM_DIR,
    EVAL_REPORT_PATH, CONFUSION_MATRIX_PATH, TRAINING_CURVES_PATH,
    HISTORY_PATH, TEST_DIR
)
from data_pipeline import get_datasets
from gradcam import explain_prediction

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

print("📦 Loading model and test set...")
model = tf.keras.models.load_model(MODEL_PATH)
_, _, test_ds = get_datasets()

# ---- Predictions over full test set ----
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(tf.argmax(labels, axis=1).numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

test_accuracy = float(np.mean(y_true == y_pred))
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")

report = classification_report(
    y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

with open(EVAL_REPORT_PATH, "w") as f:
    json.dump({"test_accuracy": test_accuracy, "classification_report": report}, f, indent=2)
print(f"Saved evaluation report → {EVAL_REPORT_PATH}")

# ---- Confusion matrix plot ----
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(CLASS_NAMES)))
ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (Test Acc: {test_accuracy*100:.2f}%)")
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im)
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
plt.close()
print(f"Saved confusion matrix → {CONFUSION_MATRIX_PATH}")

# ---- Training curves plot ----
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH) as f:
        hist = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["accuracy"], label="train")
    axes[0].plot(hist["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(hist["loss"], label="train")
    axes[1].plot(hist["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(TRAINING_CURVES_PATH, dpi=150)
    plt.close()
    print(f"Saved training curves → {TRAINING_CURVES_PATH}")

# ---- Sample Grad-CAM visualizations (one per class, from the test folder) ----
print("\n🔍 Generating sample Grad-CAM explanations...")
for cls in CLASS_NAMES:
    class_dir = os.path.join(TEST_DIR, cls)
    if not os.path.isdir(class_dir):
        continue
    files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        continue
    sample_path = os.path.join(class_dir, files[0])
    img = Image.open(sample_path)
    pred_class, confidence, probs, overlay = explain_prediction(img, model)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(img.convert("RGB"))
    axes[0].set_title(f"True: {cls}")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Pred: {pred_class} ({confidence*100:.1f}%)")
    axes[1].axis("off")
    plt.tight_layout()
    out_path = os.path.join(GRADCAM_DIR, f"{cls}_gradcam.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  {cls}: predicted {pred_class} ({confidence*100:.1f}%) → {out_path}")

print(f"\n✅ Evaluation complete. Test accuracy: {test_accuracy*100:.2f}%")
if test_accuracy < 0.94:
    print("⚠️  Below 94% target on test set — see suggestions in train.py output.")
