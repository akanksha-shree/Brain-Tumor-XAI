import tensorflow as tf
import os
import json
import numpy as np
from config import (
    MODELS_DIR, MODEL_PATH, HISTORY_PATH, EPOCHS, LEARNING_RATE,
    RANDOM_SEED, NUM_CLASSES
)
from data_pipeline import get_datasets
from model import build_cnn

try:
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:
    raise SystemExit("scikit-learn is required: pip install scikit-learn")

os.makedirs(MODELS_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---- GPU check (if this prints an empty list, training is on CPU and
# will be ~10-20x slower — go to Runtime > Change runtime type > GPU,
# then Runtime > Restart session, then re-run all cells from the top) ----
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU devices available: {gpus if gpus else 'NONE — training will run on CPU!'}")

# ---- Mixed precision: T4/A100/etc. GPUs have dedicated fp16 tensor cores.
# This can meaningfully speed up training with negligible accuracy impact.
# Safe no-op on CPU-only runtimes.
if gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision (float16) enabled.")

print("📦 Loading datasets...")
train_ds, val_ds, test_ds = get_datasets()

# ---- Class weights (handles Sartaj's known class imbalance) ----
print("⚖️  Computing class weights...")
labels = []
for _, lbls in train_ds:
    labels.extend(tf.argmax(lbls, axis=1).numpy())
labels = np.array(labels)
class_weights = compute_class_weight(
    'balanced', classes=np.arange(NUM_CLASSES), y=labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ---- Model ----
model = build_cnn()
model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1
    ),
]

print("🚀 Training with class weights...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# EarlyStopping with restore_best_weights already gives us the best model
# in memory, but ModelCheckpoint already wrote the best epoch to disk too.
# This final save is now redundant with the best checkpoint UNLESS training
# completed all epochs without early stopping, in which case it's a no-op
# since restore_best_weights already restored the best weights.
model.save(MODEL_PATH)

history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(HISTORY_PATH, "w") as f:
    json.dump(history_dict, f, indent=2)

best_val_acc = max(history_dict.get('val_accuracy', [0]))
print("\n✅ TRAINING FINISHED!")
print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")

if best_val_acc < 0.94:
    print(
        "⚠️  Val accuracy is below the 94% target. Consider: more epochs, "
        "checking for label/folder errors, or switching to transfer learning."
    )
