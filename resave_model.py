"""
resave_model.py
---------------
Loads the existing Keras model from the .h5 file and re-saves it in the
native .keras format (Keras 3 / TF 2.16+).

Run once from the project root:
    python resave_model.py

The output file will be:
    model/saved_model/model.keras

Do NOT retrain; this is a format-only conversion.
"""

import os
import keras

SRC  = os.path.join("model", "saved_model", "best_model.h5")
DEST = os.path.join("model", "saved_model", "model.keras")


def main():
    if not os.path.exists(SRC):
        raise FileNotFoundError(
            f"Source model not found: {SRC}\n"
            "Make sure you run this script from the project root (skinai/)."
        )

    print(f"[resave_model] Loading model from: {SRC}")

    # PatchedDense — silently drops 'quantization_config' from legacy .h5 weights
    class PatchedDense(keras.layers.Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    # FixedBatchNormalization — drops renorm kwargs written by older TF versions
    class FixedBatchNormalization(keras.layers.BatchNormalization):
        def __init__(self, **kwargs):
            kwargs.pop("renorm", None)
            kwargs.pop("renorm_clipping", None)
            kwargs.pop("renorm_momentum", None)
            super().__init__(**kwargs)

    model = keras.models.load_model(
        SRC,
        custom_objects={
            "FixedBatchNormalization": FixedBatchNormalization,
            "PatchedDense": PatchedDense,
        },
        compile=False,
    )

    print(f"[resave_model] Model loaded. Summary:")
    model.summary(line_length=90)

    print(f"\n[resave_model] Saving to: {DEST}")
    model.save(DEST)
    print(f"[resave_model] [OK] Done - model saved to {DEST}")


if __name__ == "__main__":
    main()
