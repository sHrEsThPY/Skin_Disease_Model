import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

layers = tf.keras.layers
models = tf.keras.models
EfficientNetB3 = tf.keras.applications.EfficientNetB3
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
TensorBoard = tf.keras.callbacks.TensorBoard
CSVLogger = tf.keras.callbacks.CSVLogger

# Random seeds
tf.random.set_seed(42)
np.random.seed(42)

# GPU memory growth & Mixed precision
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        # Enable mixed precision only if GPU is present
        mixed_precision = tf.keras.mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("GPU found: Enabled memory growth and mixed_float16 precision.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU with standard float32 precision.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'model', 'saved_model')

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.join(LOGS_DIR, 'tensorboard'), exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 40

def build_model(num_classes):
    backbone = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    backbone.trainable = False

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(factor=0.083),
        layers.RandomZoom(height_factor=0.2),
        layers.RandomBrightness(factor=0.2),
        layers.RandomContrast(factor=0.2),
    ], name="data_augmentation")

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs, outputs)
    return model, backbone

def main():
    if not os.path.exists(os.path.join(PROCESSED_DIR, 'train')):
        print("Dataset not found. Please run download_dataset.py first.")
        return

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DIR, 'train'),
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DIR, 'val'),
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    # Preprocessing
    def preprocess_image(image, label):
        image = image / 255.0
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - mean) / std
        return image, label

    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Weights
    print("Calculate weights...")
    y_train = np.concatenate([y for x, y in train_ds.as_numpy_iterator()], axis=0)
    class_counts = np.sum(y_train, axis=0)
    total_samples = np.sum(class_counts)
    class_weights = {i: total_samples / (num_classes * count) for i, count in enumerate(class_counts) if count > 0}

    model, backbone = build_model(num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(os.path.join(MODEL_SAVE_DIR, 'best_model.h5'), save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
        TensorBoard(log_dir=os.path.join(LOGS_DIR, 'tensorboard')),
        CSVLogger(os.path.join(LOGS_DIR, 'training_log.csv'))
    ]

    print("Phase 1: Training top layers...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks,
        class_weight=class_weights
    )

    print("Phase 2: Fine-tuning...")
    fine_tune_at = int(len(backbone.layers) * 0.7)
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in backbone.layers[fine_tune_at:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        initial_epoch=history1.epoch[-1] if history1.epoch else EPOCHS_PHASE1,
        callbacks=callbacks,
        class_weight=class_weights
    )

    model.save(os.path.join(MODEL_SAVE_DIR, 'final_model.h5'))

    with open(os.path.join(LOGS_DIR, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    acc = history1.history.get('accuracy', []) + history2.history.get('accuracy', [])
    val_acc = history1.history.get('val_accuracy', []) + history2.history.get('val_accuracy', [])
    loss = history1.history.get('loss', []) + history2.history.get('loss', [])
    val_loss = history1.history.get('val_loss', []) + history2.history.get('val_loss', [])

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history1.epoch), color='r', linestyle='--', label='Fine-tune start')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axvline(x=len(history1.epoch), color='r', linestyle='--', label='Fine-tune start')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(LOGS_DIR, 'training_curves.png'))

    print("✅ Training complete.")
    print(f"Phase 1 best validation accuracy: {max(history1.history.get('val_accuracy', [0])):.2f}")
    print(f"Phase 2 best validation accuracy: {max(history2.history.get('val_accuracy', [0])):.2f}")
    print(f"Model saved to: {os.path.join(MODEL_SAVE_DIR, 'best_model.h5')}")
    print(f"TensorBoard logs: {os.path.join(LOGS_DIR, 'tensorboard')}")

if __name__ == '__main__':
    main()
