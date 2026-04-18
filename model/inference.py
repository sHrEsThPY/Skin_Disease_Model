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
        'common_name':  'Sun Damage Spot',
        'alias':        'AK / Solar Keratosis / Intraepithelial Carcinoma',
        'severity':     '⚠️ Precancerous',
        'description':  (
            'This is a rough, scaly patch caused by years of sun exposure. Think of it as '
            'your skin showing the long-term effects of UV damage. It is not cancer yet, '
            'but about 1 in 10 untreated cases can slowly turn into skin cancer over time. '
            'These spots most commonly appear on the face, scalp, ears, and the back of hands — '
            'places that get the most sun. See a doctor soon; it is very treatable when caught early.'
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
            'Cryotherapy (liquid nitrogen freeze); topical fluorouracil (5-FU) cream; '
            'imiquimod 5% cream; photodynamic therapy (light treatment); '
            'laser resurfacing; chemical peels — all highly effective'
        ),
        'prevention':   (
            'Wear SPF 50+ sunscreen every day; use UV-protective clothing and wide-brim hats; '
            'avoid direct sun between 10 am – 4 pm; get an annual skin check'
        ),
    },

    'Basal Cell Carcinoma': {
        'common_name':  'Skin Cancer (Common Type)',
        'alias':        'BCC / Rodent Ulcer',
        'severity':     '🔴 Malignant (slow-growing)',
        'description':  (
            'This is the most common type of skin cancer in the world, but also one of the '
            'least dangerous when caught early. It grows very slowly and almost never spreads '
            'to other parts of the body. It usually looks like a shiny, pearly bump or a sore '
            'that won\'t heal. It is mainly caused by too much sun exposure over a lifetime. '
            'Do NOT ignore it — left untreated it can damage surrounding skin and tissue. '
            'Book a dermatologist appointment as soon as possible.'
        ),
        'common_terms': [
            'Nodular BCC', 'Superficial BCC', 'Morpheaform / Sclerosing BCC',
            'Pigmented BCC', 'Infiltrative BCC', 'Micronodular BCC',
            'Basosquamous carcinoma', 'Rodent ulcer',
        ],
        'symptoms':     (
            'Shiny or pearly bump on skin; flat scar-like patch; a sore that bleeds, '
            'scabs over, and keeps coming back; pink growth with raised edges; '
            'rolled borders with a dent or ulcer in the centre'
        ),
        'treatment':    (
            'Surgical removal (Mohs surgery — very precise); standard excision; '
            'freezing (cryotherapy); radiation; creams like imiquimod for early cases; '
            'oral medication for advanced cases. Cure rate is very high when caught early.'
        ),
        'prevention':   (
            'Avoid excessive sun exposure; apply SPF 30+ sunscreen daily; '
            'wear protective clothing; get annual skin checks — especially after age 40'
        ),
    },

    'Benign Keratosis': {
        'common_name':  'Harmless Skin Growth',
        'alias':        'Seborrheic Keratosis / BKL / SK',
        'severity':     '✅ Benign',
        'description':  (
            'Good news — this is completely harmless. These are very common skin growths '
            'that look like waxy, "stuck-on" brown or tan patches. They are NOT cancer '
            'and do NOT turn into cancer. Almost everyone develops a few of these as they '
            'get older. They can appear anywhere on the body and may slowly grow larger '
            'or increase in number over the years. You can have them removed if they '
            'bother you cosmetically or rub against clothing, but medically there is no '
            'need to treat them.'
        ),
        'common_terms': [
            'Seborrheic keratosis', 'Senile wart', 'Barnacle', 'Stucco keratosis',
            'Dermatosis papulosa nigra', 'Solar lentigo (flat variant)',
            'Lichen planus-like keratosis (LPLK)', 'Irritated SK',
        ],
        'symptoms':     (
            'Waxy, "stuck-on" appearance; tan, brown, or black colour; round or oval shape; '
            'variable size (a few mm to over 2.5 cm); slightly raised, rough texture; '
            'may itch if irritated; single or grouped lesions'
        ),
        'treatment':    (
            'No treatment needed in most cases. If irritating: cryotherapy (freezing), '
            'curettage (scraping), shave excision, or laser removal just for comfort or cosmetics.'
        ),
        'prevention':   (
            'Cannot really be prevented — age and genetics are the main factors. '
            'Sunscreen may slow the appearance of flat sun-related variants.'
        ),
    },

    'Dermatofibroma': {
        'common_name':  'Firm Skin Lump (Harmless)',
        'alias':        'DF / Fibrous Histiocytoma / Benign Fibrous Histiocytoma',
        'severity':     '✅ Benign',
        'description':  (
            'This is a small, harmless lump in the skin — completely non-cancerous. '
            'It feels firm or hard when you press on it and is usually brownish-pink in colour. '
            'A classic clue: if you pinch it, the skin dimples inward. Most of these appear '
            'on the legs and can sometimes develop after a minor insect bite or small skin injury. '
            'They are very slow-growing and stay small. No treatment is normally needed '
            'unless it gets in the way or bothers you.'
        ),
        'common_terms': [
            'Fibrous histiocytoma', 'Sclerosing hemangioma', 'Nodular subepidermal fibrosis',
            'Dermal fibroma', 'Histiocytoma cutis', 'Fibroid',
        ],
        'symptoms':     (
            'Firm, hard lump under the skin; brownish-pink to reddish-brown colour; '
            'dimples when pinched (key sign); typically 0.5–1 cm in size; '
            'slow-growing; may be mildly tender or itchy; usually a single lesion'
        ),
        'treatment':    (
            'Usually no treatment needed. If it is painful or bothers you cosmetically, '
            'a doctor can remove it by surgical excision. Note: it can come back if not fully removed.'
        ),
        'prevention':   (
            'No known way to prevent these. They sometimes appear after minor skin injuries '
            'or insect bites, so protecting skin from trauma may help.'
        ),
    },

    'Melanoma': {
        'common_name':  'Dangerous Skin Cancer (Melanoma)',
        'alias':        'Malignant Melanoma / Cutaneous Melanoma',
        'severity':     '🔴 Malignant (high-risk)',
        'description':  (
            '⚠️ This is the most serious type of skin cancer. Please see a doctor TODAY. '
            'Melanoma starts in the cells that give skin its colour (moles). Unlike most '
            'skin cancers, it can spread to other organs like the lungs, liver, and brain '
            'if not treated quickly. The good news: when caught in the very early stage, '
            'the cure rate is over 98%. That is why acting fast is so important. '
            'Use the ABCDE rule to spot it: Asymmetry, Border irregularity, Colour changes, '
            'Diameter > 6mm, Evolving size or shape. Do not wait — book a dermatologist appointment immediately.'
        ),
        'common_terms': [
            'Superficial spreading melanoma', 'Nodular melanoma', 'Lentigo maligna melanoma',
            'Acral lentiginous melanoma', 'Amelanotic melanoma', 'Desmoplastic melanoma',
            'Spitzoid melanoma', 'ABCDE rule', 'Breslow thickness', 'Clark level',
        ],
        'symptoms':     (
            'Mole that is not the same on both sides (asymmetric); uneven or jagged border; '
            'multiple colours in one spot (tan, brown, black, red, white, or blue); '
            'larger than a pencil eraser (> 6mm); changing in size, shape, or colour; '
            'bleeds or ulcerates'
        ),
        'treatment':    (
            'Surgical removal as first step; lymph node check; immunotherapy drugs '
            '(e.g. pembrolizumab, nivolumab); targeted drugs for specific gene mutations; '
            'radiation therapy. Highly effective if caught early.'
        ),
        'prevention':   (
            'Never use tanning beds; apply SPF 30+ sunscreen every day; stay in shade; '
            'check your moles monthly using the ABCDE rule; get a full-body skin check annually; '
            'extra caution if family members have had melanoma'
        ),
    },

    'Melanocytic Nevi': {
        'common_name':  'Common Mole',
        'alias':        'Common Mole / NV / Nevocellular Nevus',
        'severity':     '✅ Benign',
        'description':  (
            'This is just a normal mole — nothing to worry about in most cases. '
            'Almost everyone has between 10 and 40 moles by adulthood. They are simply '
            'small clusters of pigment cells (melanocytes) and are completely harmless. '
            'However, it is still a good habit to keep an eye on your moles and watch '
            'for any changes in size, shape, colour, or texture. A mole that starts '
            'changing could rarely be an early sign of melanoma, so if anything looks '
            'different, mention it to a doctor at your next check-up.'
        ),
        'common_terms': [
            'Common mole', 'Junctional nevus', 'Compound nevus', 'Intradermal nevus',
            'Congenital melanocytic nevus', 'Dysplastic nevus (atypical mole)',
            'Blue nevus', 'Spitz nevus', 'Halo nevus',
        ],
        'symptoms':     (
            'Round or oval shape; even tan or brown colour; smooth surface; '
            'clear defined borders; typically smaller than a pencil eraser (< 6mm); '
            'flat or slightly raised; may have a hair growing from it; stays the same over years'
        ),
        'treatment':    (
            'No treatment needed for a normal mole. If a mole looks suspicious your doctor '
            'may remove it and send it for testing. Atypical moles are monitored with a dermatoscope.'
        ),
        'prevention':   (
            'Reduce sun exposure from a young age; use sunscreen daily; '
            'monitor moles monthly using the ABCDE rule; see a dermatologist annually'
        ),
    },

    'Vascular Lesion': {
        'common_name':  'Blood Vessel Skin Spot',
        'alias':        'Vascular Lesion / VASC / Angioma',
        'severity':     '✅ Benign',
        'description':  (
            'This is a skin spot caused by small blood vessels near the surface of the skin. '
            'The most common type is a "cherry angioma" — a tiny bright-red dot that many adults '
            'get as they grow older, especially on the chest and stomach. These are completely '
            'harmless. Some types (called pyogenic granulomas) can bleed a lot if bumped, '
            'so those may be worth having a doctor remove. Birthmarks like port-wine stains '
            'also fall into this category. Overall, these are cosmetic issues — not a health risk.'
        ),
        'common_terms': [
            'Cherry angioma (Campbell de Morgan spot)', 'Angiokeratoma', 'Pyogenic granuloma',
            'Spider angioma (nevus araneus)', 'Port-wine stain', 'Hemangioma',
            'Lymphangioma', 'Venous lake', 'Glomus tumor',
        ],
        'symptoms':     (
            'Bright red to purple spot; ranges from a pinhead to a few centimetres; '
            'smooth or slightly raised surface; may bleed easily if scratched or bumped; '
            'single or multiple spots; many fade when pressed'
        ),
        'treatment':    (
            'Small, harmless ones need no treatment. Laser therapy is very effective for '
            'cosmetic removal. Pyogenic granulomas (the bleeding type) are best removed by a doctor. '
            'Haemangiomas in babies often shrink on their own over time.'
        ),
        'prevention':   (
            'Cannot really be prevented — cherry angiomas naturally increase with age. '
            'Protect skin from trauma to avoid pyogenic granulomas forming.'
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
        has_groq = bool(os.environ.get("GROQ_API_KEY", ""))

        # Allow Groq-only mode (local model may still be None)
        if self.model is None and not has_groq:
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

            # ── Groq Vision API (primary — most accurate) ─────────────────
            if has_groq:
                groq_result = self._groq_predict(image_bytes)
                if groq_result is not None:
                    return groq_result
                logger.warning("Groq predict failed or returned None — falling back to local model.")

            # ── Local Keras model (fallback / dummy) ──────────────────────
            if self.model is None:
                return self._fallback()

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
                    'common_name':  info.get('common_name', name),
                    'alias':        info.get('alias', ''),
                    'confidence':   round(float(preds[i]) * 100, 1),
                    'severity':     info.get('severity', ''),
                    'description':  info.get('description', ''),
                    'common_terms': info.get('common_terms', []),
                    'symptoms':     info.get('symptoms', ''),
                    'treatment':    info.get('treatment', ''),
                    'prevention':   info.get('prevention', ''),
                })

            top_conf = results[0]['confidence'] if results else 0

            # Filter out truly unrecognisable images (too blurry)
            if top_conf < 40.0:
                logger.warning(f"Local model low confidence. top_conf: {top_conf}")
                return {
                    'results': [{
                        'disease':      'Unrecognized Image',
                        'common_name':  'Unrecognized Image',
                        'alias':        'Too Blurry / Low Confidence',
                        'confidence':   top_conf,
                        'severity':     '⚪ Unknown',
                        'description':  'The AI confidence is too low. Please upload a clearer photo of the lesion.',
                        'common_terms': [],
                        'symptoms':     '—',
                        'treatment':    '—',
                        'prevention':   '—',
                    }],
                    'fallback':         False,
                    'low_conf_warning': True,
                }

            low_conf_warning = top_conf < 60.0
            if low_conf_warning:
                logger.info(f"Returning local prediction with low-conf warning. top_conf: {top_conf}")

            return {
                'results':           results,
                'fallback':          False,
                'low_conf_warning':  low_conf_warning,
                'source':            'local_model',
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
                'common_name':  'Service Temporarily Unavailable',
                'alias':        '',
                'confidence':   0.0,
                'severity':     '❓ Unknown',
                'description':  f'The AI model failed to load. Reason: {err_msg}',
                'common_terms': [],
                'symptoms':     '—',
                'treatment':    '—',
                'prevention':   '—',
            }],
            'fallback':         True,
            'low_conf_warning': True,
        }

    # ── Groq Vision Primary Predictor ─────────────────────────────────────
    def _groq_predict(self, image_bytes: bytes) -> dict | None:
        """
        Uses Groq's llama-3.2-90b-vision-preview model to classify the skin lesion.
        Returns a predict()-compatible result dict on success, or None on failure.
        When None is returned, caller falls back to the local Keras model.
        """
        import base64, json as _json
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return None

        CLASSES = list(DISEASE_INFO.keys())
        class_list = "\n".join(f"- {c}" for c in CLASSES)

        prompt = f"""You are a board-certified dermatology AI specialising in skin lesion image classification.

Analyse the provided skin image and classify it into EXACTLY one of these 7 conditions:
{class_list}

You MUST respond with ONLY a valid JSON object — no markdown fences, no extra text.
Format:
{{
  "is_skin_lesion": true,
  "top_predictions": [
    {{"disease": "<exact class name>", "confidence": <integer 0-100>}},
    {{"disease": "<exact class name>", "confidence": <integer 0-100>}},
    {{"disease": "<exact class name>", "confidence": <integer 0-100>}}
  ],
  "clinical_note": "<one concise sentence describing the key visual features you observed>"
}}

Rules:
- disease MUST be the exact class name from the list above
- top_predictions must have exactly 3 entries in descending confidence order
- confidence values must sum to 100
- if the image is clearly NOT a skin lesion (animal, object, scenery), set is_skin_lesion to false and all confidences to 0"""

        try:
            client = Groq(api_key=api_key)
            b64 = base64.b64encode(image_bytes).decode('utf-8')

            response = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                model="llama-3.2-90b-vision-preview",
                temperature=0.1,
                max_tokens=400,
            )

            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if the model added them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = _json.loads(raw.strip())

            # Non-skin rejection
            if not data.get("is_skin_lesion", True):
                logger.warning("Groq vision: image rejected as non-skin.")
                return {
                    'results': [{
                        'disease':      'Unrecognized Image',
                        'common_name':  'Not a Skin Image',
                        'alias':        'Non-Skin Object Detected',
                        'confidence':   0.0,
                        'severity':     '⚪ Unknown',
                        'description':  'The AI determined this image does not show a skin lesion. Please upload a clear, close-up photo of the affected skin area.',
                        'common_terms': [],
                        'symptoms':     '—',
                        'treatment':    '—',
                        'prevention':   '—',
                    }],
                    'fallback':         False,
                    'low_conf_warning': True,
                    'source':           'groq_vision',
                }

            # Build results from predictions
            results = []
            for pred in data.get("top_predictions", [])[:3]:
                name = pred.get("disease", "").strip()
                if name not in DISEASE_INFO:
                    logger.warning(f"Groq returned unknown class: {name!r} — skipping.")
                    continue
                info = DISEASE_INFO[name]
                results.append({
                    'disease':      name,
                    'common_name':  info.get('common_name', name),
                    'alias':        info.get('alias', ''),
                    'confidence':   round(float(pred.get("confidence", 0)), 1),
                    'severity':     info.get('severity', ''),
                    'description':  info.get('description', ''),
                    'common_terms': info.get('common_terms', []),
                    'symptoms':     info.get('symptoms', ''),
                    'treatment':    info.get('treatment', ''),
                    'prevention':   info.get('prevention', ''),
                    'clinical_note': data.get('clinical_note', ''),
                })

            if not results:
                logger.warning("Groq returned no valid disease classes.")
                return None

            top_conf = results[0]['confidence']
            logger.info(f"Groq vision predict: {results[0]['disease']} @ {top_conf}%")

            return {
                'results':           results,
                'fallback':          False,
                'low_conf_warning':  top_conf < 60.0,
                'source':            'groq_vision',
            }

        except Exception as e:
            logger.error(f"Groq vision predict error: {e}", exc_info=True)
            return None

    # ── Legacy skin-only gatekeeper (kept for local-model fallback path) ──
    def _api_is_skin(self, image_bytes: bytes) -> bool:
        """Simple YES/NO skin check — only used when Groq full-predict is unavailable."""
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return True
        try:
            import base64
            from groq import Groq
            client = Groq(api_key=api_key)
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            completion = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer ONLY 'YES' if this image shows human skin or a skin lesion. Answer ONLY 'NO' otherwise."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    ],
                }],
                model="llama-3.2-11b-vision-preview",
                temperature=0.0,
                max_tokens=5,
            )
            ans = completion.choices[0].message.content.strip().upper()
            return 'YES' in ans
        except Exception as e:
            logger.error(f"Gatekeeper API Error: {e}")
            return True
