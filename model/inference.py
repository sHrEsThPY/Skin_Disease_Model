"""
SkinAI Inference Engine — HAM10000 7-class EfficientNetB0
"""
import os, io, json
os.environ['TF_USE_LEGACY_KERAS'] = '1'   # Fix for TF 2.16+ keras split

import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ── Disease info ─────────────────────────────────────────────────────────────
DISEASE_INFO = {
    'Actinic Keratosis': {
        'alias':       'AK / Solar Keratosis',
        'severity':    '⚠️ Precancerous',
        'description': 'Rough, scaly patches caused by years of sun exposure. Actinic keratosis is a precancerous skin condition; if left untreated, it can develop into squamous cell carcinoma.',
        'symptoms':    'Rough, dry, scaly patch, flat-to-bumpy surface, itching or burning, skin discolouration',
        'treatment':   'Cryotherapy (liquid nitrogen), topical fluorouracil (5-FU), imiquimod cream, photodynamic therapy',
        'prevention':  'Daily SPF 50+ sunscreen, protective clothing, regular dermatologist checks',
    },
    'Basal Cell Carcinoma': {
        'alias':       'BCC',
        'severity':    '🔴 Malignant (slow growing)',
        'description': 'The most common form of skin cancer. BCC arises from basal cells in the lowest layer of the epidermis, typically on sun-exposed areas. Rarely metastasises but can cause significant local tissue damage.',
        'symptoms':    'Pearly or waxy bump, flat flesh-coloured lesion, bleeding sore that heals and returns, pink growth with raised edges',
        'treatment':   'Surgical excision, Mohs surgery, cryosurgery, radiation, topical imiquimod',
        'prevention':  'Avoid UV exposure, use broad-spectrum sunscreen, annual skin checks',
    },
    'Benign Keratosis': {
        'alias':       'Seborrheic Keratosis / BKL',
        'severity':    '✅ Benign',
        'description': 'Common, non-cancerous skin growth that appears as waxy, wart-like, tan to brown plaques. They appear to be "stuck on" the skin surface. No treatment is required unless they become irritated.',
        'symptoms':    'Waxy, stuck-on appearance, tan/brown/black colour, round or oval, variable size',
        'treatment':   'Usually no treatment needed. Cryotherapy or curettage if irritated or cosmetically bothersome',
        'prevention':  'No specific prevention; genetic predisposition plays a role',
    },
    'Dermatofibroma': {
        'alias':       'DF / Fibrous Histiocytoma',
        'severity':    '✅ Benign',
        'description': 'A common, harmless skin growth that results from an accumulation of fibroblasts (soft tissue cells). Often feels like a hard lump under the skin and may dimple inward when pinched.',
        'symptoms':    'Firm, raised skin bump, brownish-red colour, dimples inward when pinched, slow growing',
        'treatment':   'No treatment required unless symptomatic. Surgical excision if desired cosmetically',
        'prevention':  'No known specific prevention',
    },
    'Melanoma': {
        'alias':       'Malignant Melanoma',
        'severity':    '🔴 Malignant (high-risk)',
        'description': 'The most dangerous form of skin cancer. Melanoma develops from melanocytes and can metastasise to organs. Early detection is critical — survival rates drop significantly in later stages.',
        'symptoms':    'Asymmetric mole, irregular/ragged border, multiple colours, diameter >6mm, evolving size or shape',
        'treatment':   'Surgical excision, immunotherapy (pembrolizumab), targeted therapy (BRAF inhibitors), radiation',
        'prevention':  'Avoid tanning beds, use SPF 30+ daily, regular self-checks using the ABCDE rule',
    },
    'Melanocytic Nevi': {
        'alias':       'Common Mole / NV',
        'severity':    '✅ Benign',
        'description': 'Common benign skin lesions composed of clusters of melanocytes. Most people have between 10 and 40 moles. Although generally harmless, they should be monitored for changes that could indicate melanoma.',
        'symptoms':    'Round, uniform brown spots, smooth surface, well-defined border, stable over time',
        'treatment':   'No treatment needed. Surgical removal if desired or if suspicious change is noted',
        'prevention':  'Limit sun exposure, use sunscreen, regular monitoring for changes',
    },
    'Vascular Lesion': {
        'alias':       'Vascular Lesion / VASC',
        'severity':    '✅ Benign',
        'description': 'Skin lesions that arise from blood vessels, including cherry angiomas, angiokeratomas, and pyogenic granulomas. These are generally benign and often a cosmetic concern.',
        'symptoms':    'Bright red to purple discolouration, may bleed easily, smooth surface, various sizes',
        'treatment':   'Laser therapy, electrosurgery, cryotherapy if removal is desired',
        'prevention':  'Generally not preventable; some may resolve spontaneously',
    },
}

class SkinAIPredictor:
    def __init__(self):
        self.model = None
        self.class_names = []
        self._load_model()

    def _load_model(self):
        base = os.path.dirname(__file__)
        save_dir = os.path.join(base, 'saved_model')

        # Load class names saved during training
        names_path = os.path.join(save_dir, 'class_names.json')
        if os.path.exists(names_path):
            with open(names_path) as f:
                self.class_names = json.load(f)
        else:
            # Fallback — alphabetical order used in training
            self.class_names = sorted(DISEASE_INFO.keys())

        model_path = os.path.join(save_dir, 'best_model.h5')
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}. Predictions will be unavailable.")
            return

        try:
            # TF 2.16+ ships keras standalone; try multiple paths
            try:
                import tf_keras
                self.model = tf_keras.models.load_model(model_path)
            except (ImportError, Exception):
                try:
                    import keras
                    self.model = keras.models.load_model(model_path)
                except (ImportError, Exception):
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(model_path)
            logger.info("✅ Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def predict(self, image_bytes: bytes) -> dict:
        if self.model is None:
            return self._fallback()

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = img.resize((224, 224))
            arr = np.array(img, dtype=np.float32)   # [0,255]
            arr = np.expand_dims(arr, 0)             # (1,224,224,3)

            preds = self.model.predict(arr, verbose=0)[0]  # (7,)

            top3_idx = preds.argsort()[::-1][:3]
            results  = []
            for i in top3_idx:
                name = self.class_names[i]
                info = DISEASE_INFO.get(name, {})
                results.append({
                    'disease':     name,
                    'alias':       info.get('alias', ''),
                    'confidence':  round(float(preds[i]) * 100, 1),
                    'severity':    info.get('severity', ''),
                    'description': info.get('description', ''),
                    'symptoms':    info.get('symptoms', ''),
                    'treatment':   info.get('treatment', ''),
                    'prevention':  info.get('prevention', ''),
                })
            return {'results': results, 'fallback': False}

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._fallback()

    def _fallback(self):
        return {
            'results': [{
                'disease': 'Melanocytic Nevi',
                'alias': 'Common Mole',
                'confidence': 0.0,
                'severity': '✅ Benign',
                'description': 'Model not loaded. Please train the model first.',
                'symptoms': '—',
                'treatment': '—',
                'prevention': '—',
            }],
            'fallback': True,
        }
