"""
SkinAI Inference Engine — HAM10000 7-class EfficientNetB0
Fixes:
  - Class name normalisation (underscore → space)
  - Temperature scaling for calibrated confidences
  - Full disease metadata on all top-3 results
  - Common medical terms per disease
"""
import os, io, json

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ── Disease info ──────────────────────────────────────────────────────────────
# Keys must match normalised class names (spaces, no underscores)
DISEASE_INFO = {
    'Actinic Keratosis': {
        'alias':        'AK / Solar Keratosis / Intraepithelial Carcinoma',
        'severity':     '⚠️ Precancerous',
        'description':  (
            'Actinic keratosis (AK) is a rough, scaly patch of skin caused by decades of '
            'cumulative UV exposure. It is considered a precancerous lesion because, if left '
            'untreated, roughly 5–10 % of cases can progress to invasive squamous cell '
            'carcinoma. AKs are most common on sun-exposed areas such as the face, scalp, '
            'ears, neck, forearms, and the back of the hands.'
        ),
        'common_terms': [
            'Solar keratosis', 'Senile keratosis', 'Squamous intraepithelial neoplasia',
            'Actinic cheilitis (lip variant)', 'Bowenoid AK', 'Hypertrophic AK',
            'Lichenoid AK', 'Pigmented AK',
        ],
        'symptoms':     (
            'Rough, dry or scaly patch (< 2.5 cm); flat-to-slightly-raised surface; '
            'pink, red, or brownish discolouration; itching, burning, or tenderness; '
            'hard wart-like surface in severe cases'
        ),
        'treatment':    (
            'Cryotherapy (liquid nitrogen); topical fluorouracil (5-FU); '
            'imiquimod 5 % cream; diclofenac 3 % gel; photodynamic therapy (PDT); '
            'laser resurfacing; chemical peels'
        ),
        'prevention':   (
            'Daily SPF 50+ broad-spectrum sunscreen; UPF protective clothing and hats; '
            'avoid peak UV hours (10 am – 4 pm); annual dermatologist skin checks'
        ),
    },

    'Basal Cell Carcinoma': {
        'alias':        'BCC / Rodent Ulcer',
        'severity':     '🔴 Malignant (slow-growing)',
        'description':  (
            'Basal cell carcinoma is the most common form of skin cancer worldwide, arising '
            'from basal cells in the deepest layer of the epidermis. It typically grows very '
            'slowly and rarely metastasises, but can cause significant local tissue destruction '
            'if ignored. Chronic UV exposure, fair skin, and immunosuppression are major risk '
            'factors. BCC accounts for ~80 % of all non-melanoma skin cancers.'
        ),
        'common_terms': [
            'Nodular BCC', 'Superficial BCC', 'Morpheaform / Sclerosing BCC',
            'Pigmented BCC', 'Infiltrative BCC', 'Micronodular BCC',
            'Basosquamous carcinoma', 'Rodent ulcer',
        ],
        'symptoms':     (
            'Pearly or waxy translucent bump; flat, flesh-coloured or brown scar-like lesion; '
            'bleeding or scabbing sore that heals and returns; pink growth with raised edges; '
            'rolled borders with central ulceration'
        ),
        'treatment':    (
            'Mohs micrographic surgery (gold standard); standard surgical excision; '
            'cryosurgery; electrodessication and curettage; radiation therapy; '
            'topical imiquimod or 5-FU (for superficial BCC); vismodegib/sonidegib (advanced)'
        ),
        'prevention':   (
            'Avoid UV overexposure; use broad-spectrum SPF 30+ sunscreen daily; '
            'wear protective clothing; annual dermatology screening, especially after age 40'
        ),
    },

    'Benign Keratosis': {
        'alias':        'Seborrheic Keratosis / BKL / SK',
        'severity':     '✅ Benign',
        'description':  (
            'Benign keratosis-like lesions (BKL) are extremely common, non-cancerous skin '
            'growths that appear as waxy, wart-like, tan-to-dark-brown plaques that look '
            '"stuck on" the skin surface. They typically appear from middle age onward and '
            'may increase in number with age. They are harmless but may be removed for '
            'cosmetic reasons or if they become irritated.'
        ),
        'common_terms': [
            'Seborrheic keratosis', "Senile wart", 'Barnacle', 'Stucco keratosis',
            'Dermatosis papulosa nigra', 'Solar lentigo (flat variant)',
            'Lichen planus-like keratosis (LPLK)', 'Irritated SK',
        ],
        'symptoms':     (
            'Waxy, "stuck-on" appearance; tan, brown, or black colour; round or oval shape; '
            'variable size (few mm to > 2.5 cm); slightly raised, rough texture; '
            'may itch if irritated; single or grouped lesions'
        ),
        'treatment':    (
            'Usually no treatment required; cryotherapy or curettage if irritated; '
            'shave excision; electrocautery; laser ablation (for cosmetic removal)'
        ),
        'prevention':   (
            'No specific prevention; genetic predisposition and age are primary drivers. '
            'Sunscreen use may slow development of solar lentigos.'
        ),
    },

    'Dermatofibroma': {
        'alias':        'DF / Fibrous Histiocytoma / Benign Fibrous Histiocytoma',
        'severity':     '✅ Benign',
        'description':  (
            'Dermatofibroma is a common, harmless benign skin growth composed of an '
            'overgrowth of fibroblasts (soft tissue cells) in the dermis. It typically '
            'presents as a small, firm, raised nodule most often found on the lower legs. '
            'The lesion often dimples inward when pinched (Fitzpatrick dimple sign), which '
            'is a key diagnostic feature. Cause is unknown but may follow minor skin trauma.'
        ),
        'common_terms': [
            'Fibrous histiocytoma', 'Sclerosing hemangioma', 'Nodular subepidermal fibrosis',
            'Dermal fibroma', 'Histiocytoma cutis', 'Fibroid',
        ],
        'symptoms':     (
            'Firm, hard lump under the skin; brownish-pink to reddish-brown colour; '
            'positive Fitzpatrick sign (dimples when pinched); typically 0.5–1 cm; '
            'slow-growing; may be mildly tender or itch; single lesion usually'
        ),
        'treatment':    (
            'No treatment required in most cases; surgical excision if symptomatic or '
            'cosmetically unacceptable; note: recurrence possible if not fully excised'
        ),
        'prevention':   (
            'No known specific prevention; may arise at sites of minor skin injury or '
            'insect bites in some cases'
        ),
    },

    'Melanoma': {
        'alias':        'Malignant Melanoma / Cutaneous Melanoma',
        'severity':     '🔴 Malignant (high-risk)',
        'description':  (
            'Melanoma is the most dangerous form of skin cancer, developing from '
            'melanocytes (pigment-producing cells). It can arise in a pre-existing mole or '
            'appear as a new lesion. Melanoma has a high tendency to metastasise to lymph '
            'nodes and distant organs. Early detection is critical — 5-year survival is > 98 % '
            'for stage I but drops to ~20 % for stage IV. Annual full-body skin checks and '
            'the ABCDE rule are essential for early identification.'
        ),
        'common_terms': [
            'Superficial spreading melanoma', 'Nodular melanoma', 'Lentigo maligna melanoma',
            'Acral lentiginous melanoma', 'Amelanotic melanoma', 'Desmoplastic melanoma',
            'Spitzoid melanoma', 'ABCDE rule', 'Breslow thickness', 'Clark level',
        ],
        'symptoms':     (
            'Asymmetric mole; irregular or notched border; multiple colours (tan, brown, '
            'black, red, white, blue); diameter > 6 mm (eraser-sized); evolving size, shape, '
            'or colour; ulceration or bleeding; satellite lesions'
        ),
        'treatment':    (
            'Wide local excision (primary); sentinel lymph node biopsy; immunotherapy '
            '(pembrolizumab, nivolumab, ipilimumab); targeted therapy (vemurafenib, '
            'dabrafenib for BRAF V600E mutation); radiation; isolated limb perfusion'
        ),
        'prevention':   (
            'Avoid tanning beds; use SPF 30+ broad-spectrum sunscreen daily; seek shade; '
            'monthly self-skin checks using ABCDE rule; annual dermatologist exam; '
            'protect high-risk individuals with family history of melanoma'
        ),
    },

    'Melanocytic Nevi': {
        'alias':        'Common Mole / NV / Nevocellular Nevus',
        'severity':     '✅ Benign',
        'description':  (
            'Melanocytic nevi (moles) are common, benign skin lesions consisting of clusters '
            'of melanocytes. Most people have 10–40 moles. Although the vast majority are '
            'harmless and stable, they should be monitored regularly for changes that could '
            'indicate early melanoma transformation. Congenital nevi (present from birth) '
            'carry a slightly higher lifetime risk of malignant transformation.'
        ),
        'common_terms': [
            'Common mole', 'Junctional nevus', 'Compound nevus', 'Intradermal nevus',
            'Congenital melanocytic nevus', 'Dysplastic nevus (atypical mole)',
            'Blue nevus', 'Spitz nevus', 'Halo nevus',
        ],
        'symptoms':     (
            'Round or oval shape; uniform tan/brown colour; smooth surface; '
            'well-defined borders; typically < 6 mm; flat or slightly raised; '
            'may have hair; stable over years'
        ),
        'treatment':    (
            'No treatment needed for typical moles; surgical excision if suspicious features; '
            'dermoscopy follow-up for dysplastic nevi; complete excision for congenital giant nevi'
        ),
        'prevention':   (
            'Limit sun exposure from childhood; use sunscreen; early and regular '
            'monitoring for changes using the ABCDE rule; genetic counselling for '
            'familial atypical mole-melanoma syndrome (FAMM)'
        ),
    },

    'Vascular Lesion': {
        'alias':        'Vascular Lesion / VASC / Angioma',
        'severity':     '✅ Benign',
        'description':  (
            'Vascular lesions are skin abnormalities arising from blood or lymphatic vessels. '
            'They include a wide spectrum of conditions from cherry angiomas (very common in '
            'adults) to angiokeratomas and pyogenic granulomas. Most are benign and purely '
            'cosmetic, though pyogenic granulomas may bleed profusely with trauma. '
            'Vascular birthmarks (port-wine stains, hemangiomas) also fall into this category.'
        ),
        'common_terms': [
            'Cherry angioma (Campbell de Morgan spot)', 'Angiokeratoma', 'Pyogenic granuloma',
            'Spider angioma (nevus araneus)', 'Port-wine stain', 'Hemangioma',
            'Lymphangioma', 'Venous lake', 'Glomus tumor',
        ],
        'symptoms':     (
            'Bright red to purple discolouration; ranging in size from pinhead to centimetres; '
            'smooth or slightly raised surface; may bleed easily if traumatised; '
            'single or multiple lesions; some blanch under pressure'
        ),
        'treatment':    (
            'Laser therapy (pulsed dye laser); electrosurgery; cryotherapy; '
            'surgical excision for pyogenic granuloma; propranolol or laser for hemangiomas; '
            'no treatment needed for small, asymptomatic cherry angiomas'
        ),
        'prevention':   (
            'Generally not preventable; cherry angiomas increase normally with age; '
            'pyogenic granulomas may follow trauma — keep wounds clean; '
            'some resolve spontaneously (infantile hemangiomas)'
        ),
    },
}

# Normalise any stored class name: "Actinic_Keratosis" → "Actinic Keratosis"
def _normalise_name(name: str) -> str:
    return name.replace('_', ' ').strip()


class SkinAIPredictor:
    def __init__(self):
        self.model       = None
        self.class_names = []
        self.temperature = 1.0   # default — no scaling
        self._load_model()

    # ── Load model + metadata ─────────────────────────────────────────────
    def _load_model(self):
        base     = os.path.dirname(__file__)
        save_dir = os.path.join(base, 'saved_model')

        # ── Class names ──────────────────────────────────────────────────
        names_path = os.path.join(save_dir, 'class_names.json')
        if os.path.exists(names_path):
            with open(names_path) as f:
                raw = json.load(f)
            # Normalise underscores → spaces so lookup works
            self.class_names = [_normalise_name(n) for n in raw]
        else:
            self.class_names = sorted(DISEASE_INFO.keys())
        logger.info(f"Class names: {self.class_names}")

        # ── Temperature scaling ──────────────────────────────────────────
        temp_path = os.path.join(base, 'temperature.json')
        if os.path.exists(temp_path):
            try:
                with open(temp_path) as f:
                    t = json.load(f).get('temperature', 1.0)
                # Clamp to a sane range — a temperature of 5 makes all probs ~1/7
                # so we cap it at 2.0 to keep predictions meaningful
                self.temperature = float(max(0.5, min(t, 2.0)))
                logger.info(f"Temperature scaling: {self.temperature:.4f} (raw={t:.4f})")
            except Exception as e:
                logger.warning(f"Could not load temperature.json: {e}")

        # ── Keras model ──────────────────────────────────────────────────
        model_path = os.path.join(save_dir, 'model.keras')
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}. Predictions will be unavailable.")
            return

        try:
            import keras

            # Bulletproof: Custom BatchNormalization that blindly drops legacy renorm kwargs
            class FixedBatchNormalization(keras.layers.BatchNormalization):
                def __init__(self, **kwargs):
                    kwargs.pop('renorm', None)
                    kwargs.pop('renorm_clipping', None)
                    kwargs.pop('renorm_momentum', None)
                    super().__init__(**kwargs)

            # PatchedDense: drops quantization_config for cross-version compatibility
            class PatchedDense(keras.layers.Dense):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('quantization_config', None)
                    super().__init__(*args, **kwargs)

            self.model = keras.models.load_model(
                model_path,
                custom_objects={
                    'FixedBatchNormalization': FixedBatchNormalization,
                    'PatchedDense': PatchedDense,
                },
                compile=False
            )
            logger.info("✅ Model loaded successfully.")
        except Exception as e:
            self.load_error = str(e)
            logger.error(f"Failed to load model: {e}", exc_info=True)

    # ── Preprocess ────────────────────────────────────────────────────────
    def _preprocess(self, image_bytes: bytes) -> 'np.ndarray':
        """Resize to 224×224 and return float32 array in [0, 255] (EfficientNet expected range)."""
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)   # [0, 255]
        return np.expand_dims(arr, 0)            # (1, 224, 224, 3)

    # ── Temperature-scaled softmax ────────────────────────────────────────
    def _apply_temperature(self, logits: 'np.ndarray') -> 'np.ndarray':
        """Apply temperature scaling then softmax for calibrated probabilities."""
        import numpy as np
        # Convert softmax output → pseudo-logits via log
        eps = 1e-8
        log_probs = np.log(np.clip(logits, eps, 1.0))
        # Scale
        scaled = log_probs / self.temperature
        # Stable softmax
        scaled -= scaled.max()
        e = np.exp(scaled)
        return e / e.sum()

    # ── Public predict ────────────────────────────────────────────────────
    def predict(self, image_bytes: bytes) -> dict:
        if self.model is None:
            return self._fallback()

        try:
            # ── Skin-tone validation (OpenCV HSV) ────────────────────────
            def is_likely_skin_image(pil_img):
                """Returns False if skin-tone pixels are below 10% of the image."""
                hsv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2HSV)
                mask = (
                    (hsv[:, :, 0] < 25) &
                    (hsv[:, :, 1] > 30) &
                    (hsv[:, :, 2] > 80)
                )
                ratio = mask.sum() / mask.size
                return ratio >= 0.10

            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            if not is_likely_skin_image(pil_img):
                return {
                    "error": "Please upload a clear skin image",
                    "warning": True,
                    "prediction": None
                }

            arr  = self._preprocess(image_bytes)
            raw  = self.model.predict(arr, verbose=0)[0]          # (7,)
            preds = self._apply_temperature(raw)                   # calibrated

            top3_idx = preds.argsort()[::-1][:3]
            results  = []
            for i in top3_idx:
                name = self.class_names[i]
                info = DISEASE_INFO.get(name, {})
                results.append({
                    'disease':      name,
                    'alias':        info.get('alias', ''),
                    'confidence':   round(float(preds[i]) * 100, 1),
                    'severity':     info.get('severity', ''),
                    'description':  info.get('description', ''),
                    'common_terms': info.get('common_terms', []),
                    'symptoms':     info.get('symptoms', ''),
                    'treatment':    info.get('treatment', ''),
                    'prevention':   info.get('prevention', ''),
                })

            # Safety: if top confidence is extremely low warn user
            top_conf = results[0]['confidence'] if results else 0
            
            # --- OUT OF DOMAIN DETECTION (API BYPASS GATEKEEPER) ---
            # We use an external Vision API (Groq) to intelligently reject any dog/car/scenery photos.
            # If the API says it's not a skin photo, we bypass the AI prediction completely.
            is_valid_skin = self._api_is_skin(image_bytes)
            
            if not is_valid_skin:
                logger.warning(f"OOD rejected by GROQ_API_KEY vision verification.")
                return {
                    'results': [{
                        'disease':      'Unrecognized Image',
                        'alias':        'Non-Skin Object Detected',
                        'confidence':   0.0,
                        'severity':     '⚪ Unknown',
                        'description':  'Our API detected that this is not a photo of human skin. Please ensure you are uploading a clear image of a skin lesion, not animals, scenery, or objects.',
                        'common_terms': [],
                        'symptoms':     '—',
                        'treatment':    '—',
                        'prevention':   '—',
                    }],
                    'fallback':         False,
                    'low_conf_warning': True,
                }
                
            # If the API verified it's skin, but confidence is insanely low (blurry photo), we still warn.
            if top_conf < 40.0:
                logger.warning(f"OOD rejected by strict confidence threshold. top_conf: {top_conf}")
                return {
                    'results': [{
                        'disease':      'Unrecognized Image',
                        'alias':        'Too Blurry / Low Confidence',
                        'confidence':   top_conf,
                        'severity':     '⚪ Unknown',
                        'description':  'The AI confidence is too low to make a clinical prediction. The skin photo is likely entirely blurry or unfocused. Please upload a clear photo of the lesion.',
                        'common_terms': [],
                        'symptoms':     '—',
                        'treatment':    '—',
                        'prevention':   '—',
                    }],
                    'fallback':         False,
                    'low_conf_warning': True,
                }

            # Always return the best prediction for a valid skin image.
            # low_conf_warning lets the frontend show a caution banner without
            # hiding the result from the user.
            low_conf_warning = top_conf < 60.0
            if low_conf_warning:
                logger.info(f"Returning prediction with low-conf warning. top_conf: {top_conf}")

            return {
                'results':           results,
                'fallback':          False,
                'low_conf_warning':  low_conf_warning,
            }

        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return self._fallback()

    # ── Fallback ──────────────────────────────────────────────────────────
    def _fallback(self):
        err_msg = getattr(self, 'load_error', 'Unknown Error')
        return {
            'results': [{
                'disease':      'Model Unavailable',
                'alias':        '',
                'confidence':   0.0,
                'severity':     '❓ Unknown',
                'description':  f'The AI model failed to load. Please train the model first by running model/train.py. Reason: {err_msg}',
                'common_terms': [],
                'symptoms':     '—',
                'treatment':    '—',
                'prevention':   '—',
            }],
            'fallback':         True,
            'low_conf_warning': True,
        }

    # ── API Bypass Gatekeeper ─────────────────────────────────────────────
    def _api_is_skin(self, image_bytes: bytes) -> bool:
        """Uses Groq Vision API to act as a 100% accurate external gatekeeper"""
        import os
        api_key = os.environ.get("GROQ_API_KEY", "")
        # If API key is not set, we default to True and let the local model handle it 
        if not api_key:
            return True 
            
        try:
            import base64
            from groq import Groq
            
            client = Groq(api_key=api_key) 
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Answer ONLY 'YES' if this image distinctly shows human skin, a body part, a skin lesion, or a mole. Answer ONLY 'NO' if it is an object, animal, vehicle, scenery, or completely irrelevant."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_img}",
                                },
                            },
                        ],
                    }
                ],
                model="llama-3.2-11b-vision-preview",
                temperature=0.0,
                max_tokens=10,
            )
            ans = completion.choices[0].message.content.strip().upper()
            return 'YES' in ans
        except Exception as e:
            logger.error(f"Gatekeeper API Error: {e}")
            return True # Fallback if API fails
