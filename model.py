import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from config import IMG_SIZE, NUM_CLASSES, LAST_CONV_LAYER_NAME


def build_cnn():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1. / 255)(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 4 — last conv block. Named so Grad-CAM can locate it reliably
    # even if the architecture above changes.
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name=LAST_CONV_LAYER_NAME)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Flatten (not GAP) retains spatial layout info — helps distinguish
    # tumor location/shape, which GAP tends to wash out on this dataset.
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    # dtype='float32' here keeps the softmax output (and the loss computed
    # from it) numerically stable even when a mixed_float16 policy is set
    # globally — standard practice for mixed precision training.
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

    return models.Model(inputs, outputs)
