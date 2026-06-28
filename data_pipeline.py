import os
import tensorflow as tf
from config import (
    TRAIN_DIR, TEST_DIR, CLASS_NAMES, IMG_SIZE, BATCH_SIZE,
    VALIDATION_SPLIT, RANDOM_SEED
)


def fix_folder_names():
    """Some versions of the Sartaj dataset use '_tumor' suffixed folder
    names. Normalize them to match CLASS_NAMES exactly."""
    mappings = {
        "glioma_tumor": "glioma",
        "meningioma_tumor": "meningioma",
        "no_tumor": "notumor",
        "pituitary_tumor": "pituitary",
    }
    for base in [TRAIN_DIR, TEST_DIR]:
        if not os.path.exists(base):
            continue
        for old, new in mappings.items():
            old_path = os.path.join(base, old)
            new_path = os.path.join(base, new)
            if os.path.exists(old_path) and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"Renamed {old} → {new}")


def _verify_classes(base_dir):
    """Sanity check: confirm every expected class folder exists and is
    non-empty before we hand things off to Keras."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    found = sorted(os.listdir(base_dir))
    missing = [c for c in CLASS_NAMES if c not in found]
    if missing:
        raise ValueError(
            f"Missing expected class folders in {base_dir}: {missing}. "
            f"Found instead: {found}"
        )
    for c in CLASS_NAMES:
        n = len(os.listdir(os.path.join(base_dir, c)))
        print(f"  {c}: {n} images")
        if n == 0:
            raise ValueError(f"Class folder '{c}' in {base_dir} is empty.")


def get_datasets():
    fix_folder_names()

    print("Training directory class counts:")
    _verify_classes(TRAIN_DIR)
    print("Testing directory class counts:")
    _verify_classes(TEST_DIR)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, class_names=CLASS_NAMES, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, seed=RANDOM_SEED,
        validation_split=VALIDATION_SPLIT, subset="training", label_mode="categorical"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, class_names=CLASS_NAMES, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, seed=RANDOM_SEED,
        validation_split=VALIDATION_SPLIT, subset="validation", label_mode="categorical"
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, class_names=CLASS_NAMES, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False, label_mode="categorical"
    )

    # Cache decoded images in memory after the first epoch's read+decode,
    # so epochs 2+ don't re-read/re-decode every JPEG from disk. This is
    # usually the single biggest speedup on Colab, where disk/Drive I/O
    # is much slower than local SSD.
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()
    test_ds = test_ds.cache()

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ])

    # Augmentation runs once per BATCH (not per image) and is cheap relative
    # to disk I/O, but still CPU-bound. AUTOTUNE lets TF pipeline this
    # alongside GPU compute instead of blocking on it.
    train_ds_aug = train_ds.map(
        lambda x, y: (aug(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return (
        train_ds_aug.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
        test_ds.prefetch(tf.data.AUTOTUNE),
    )
