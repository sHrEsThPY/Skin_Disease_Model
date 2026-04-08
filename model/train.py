"""
SkinAI — HAM10000 Training Pipeline
EfficientNet-B0 fine-tuned on 7 HAM10000 skin lesion classes
CPU-safe, no mixed precision
"""

import os, sys, csv, shutil, random, json, glob
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, TensorBoard)
import numpy as np

# ── GPU / Precision guard ──────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"Using GPU: {gpus}  |  mixed_float16 ON")
else:
    tf.keras.mixed_precision.set_global_policy('float32')
    print("No GPU found. Running on CPU with float32.")

# ── Config ─────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent          # skinai/
RAW     = BASE / 'data' / 'raw'
META    = RAW  / 'HAM10000_metadata.csv'
IMG_DIR = BASE / 'data' / 'processed_ham'      # organised by class
SAVE    = BASE / 'model' / 'saved_model'

IMG_SIZE   = 224
BATCH      = 16
EPOCHS_P1  = 15
EPOCHS_P2  = 30
SEED       = 42

# HAM10000 label codes → human-readable names
LABEL_MAP = {
    'akiec': 'Actinic Keratosis',
    'bcc'  : 'Basal Cell Carcinoma',
    'bkl'  : 'Benign Keratosis',
    'df'   : 'Dermatofibroma',
    'mel'  : 'Melanoma',
    'nv'   : 'Melanocytic Nevi',
    'vasc' : 'Vascular Lesion',
}
CLASSES = sorted(LABEL_MAP.values())   # alphabetical, deterministic

# ── Step 1: Organise images by class from metadata CSV ───────────────────
def organise_dataset():
    print("\n=== Organising HAM10000 images by class ===")
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # read metadata
    mapping = {}          # image_id -> class_name
    with open(META, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['image_id']] = LABEL_MAP[row['dx']]

    # collect all .jpg paths from both parts
    all_images = list((RAW/'HAM10000_images_part_1').glob('*.jpg'))
    all_images += list((RAW/'HAM10000_images_part_2').glob('*.jpg'))
    print(f"Found {len(all_images)} raw images.")

    for cls in CLASSES:
        (IMG_DIR / cls).mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in all_images:
        stem = src.stem               # e.g. ISIC_0027419
        cls  = mapping.get(stem)
        if cls:
            dst = IMG_DIR / cls / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1

    # Print counts
    print("\nClass distribution:")
    dist = {}
    for cls in CLASSES:
        n = len(list((IMG_DIR/cls).glob('*.jpg')))
        dist[cls] = n
        print(f"  {cls:30s}: {n}")
    with open(BASE/'data'/'class_distribution.json','w') as f:
        json.dump(dist, f, indent=2)
    print(f"\nTotal organised: {sum(dist.values())} images")
    return dist


# ── Step 2: Build tf.data datasets ──────────────────────────────────────
def make_datasets():
    """ImageDataGenerator-style split via image_dataset_from_directory."""
    train_ds = keras.utils.image_dataset_from_directory(
        IMG_DIR,
        validation_split=0.15,
        subset='training',
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        label_mode='categorical',
    )
    val_ds = keras.utils.image_dataset_from_directory(
        IMG_DIR,
        validation_split=0.15,
        subset='validation',
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        label_mode='categorical',
    )

    print(f"\nClasses detected: {train_ds.class_names}")
    assert train_ds.class_names == CLASSES, \
        f"Class mismatch!\nExpected: {CLASSES}\nGot: {train_ds.class_names}"

    AUTOTUNE = tf.data.AUTOTUNE

    # Augmentation for training
    augment = keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name='augmentation')

    def preprocess_train(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = augment(x, training=True)
        # EfficientNet expects pixels in [0,255] — so scale back
        x = x * 255.0
        return x, y

    def preprocess_val(x, y):
        return tf.cast(x, tf.float32), y

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE) \
                       .cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(preprocess_val, num_parallel_calls=AUTOTUNE) \
                     .cache().prefetch(AUTOTUNE)
    return train_ds, val_ds


# ── Step 3: Compute class weights ────────────────────────────────────────
def compute_weights(dist):
    counts  = np.array([dist[c] for c in CLASSES], dtype=np.float32)
    total   = counts.sum()
    n_class = len(CLASSES)
    weights = total / (n_class * counts)
    weights = weights / weights.sum() * n_class
    return {i: float(w) for i, w in enumerate(weights)}


# ── Step 4: Build model ──────────────────────────────────────────────────
def build_model(n_classes):
    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)

    model = keras.Model(inputs, outputs)
    return model, base


# ── Step 5: Train ────────────────────────────────────────────────────────
def train():
    # Organise data
    dist = organise_dataset()
    train_ds, val_ds = make_datasets()
    class_weights    = compute_weights(dist)
    n_classes        = len(CLASSES)

    print(f"\nClass weights: {class_weights}")

    SAVE.mkdir(parents=True, exist_ok=True)

    # Save class names so inference can load them
    with open(SAVE / 'class_names.json', 'w') as f:
        json.dump(CLASSES, f)

    model, base = build_model(n_classes)

    # ── Phase 1: top layers only ──────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    cb_p1 = [
        ModelCheckpoint(str(SAVE/'best_model.h5'), save_best_only=True, monitor='val_accuracy', verbose=1),
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=str(BASE/'logs'/'tensorboard'), histogram_freq=0),
    ]

    print("\n=== Phase 1: Training top layers ===")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_P1,
        class_weight=class_weights,
        callbacks=cb_p1,
    )
    best_p1 = max(h1.history['val_accuracy'])

    # ── Phase 2: fine-tune top 40 layers of backbone ─────────────────
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    cb_p2 = [
        ModelCheckpoint(str(SAVE/'best_model.h5'), save_best_only=True, monitor='val_accuracy', verbose=1),
        EarlyStopping(patience=7, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=str(BASE/'logs'/'tensorboard'), histogram_freq=0),
    ]

    print("\n=== Phase 2: Fine-tuning backbone ===")
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_P1 + EPOCHS_P2,
        initial_epoch=len(h1.history['accuracy']),
        class_weight=class_weights,
        callbacks=cb_p2,
    )
    best_p2 = max(h2.history['val_accuracy'])

    print(f"\n{'='*50}")
    print(f"✅ Training complete.")
    print(f"Phase 1 best val accuracy : {best_p1:.4f}")
    print(f"Phase 2 best val accuracy : {best_p2:.4f}")
    print(f"Model saved to: {SAVE/'best_model.h5'}")
    print(f"{'='*50}")


if __name__ == '__main__':
    train()
