import os
import io
import time
import json
import numpy as np
from PIL import Image
try:
    import tensorflow as tf
except ImportError:
    tf = None

class SkinAIPredictor:
    CLASS_NAMES = ['Acne Vulgaris', 'Eczema', 'Psoriasis', 'Fungal Infection']
    CONFIDENCE_THRESHOLDS = {'high': 0.80, 'medium': 0.60}
    
    CONDITION_INFO = {
        'Acne Vulgaris': {
            'description': 'Acne vulgaris is a common skin condition that happens when hair follicles under the skin become clogged. Sebum—oil that helps keep skin from drying out—and dead skin cells plug the pores, which leads to outbreaks of lesions, commonly called pimples or zits.',
            'recommendation': 'Mild cases can typically be managed with over-the-counter products containing benzoyl peroxide or salicylic acid. For persistent or severe acne, consider consulting a dermatologist for prescription medications to prevent scarring.'
        },
        'Eczema': {
            'description': 'Atopic dermatitis, known as eczema, is a condition that makes your skin red and itchy. It is common in children but can occur at any age and tends to flare periodically.',
            'recommendation': 'Moisturize your skin regularly and identify/avoid flare triggers. If standard moisturizing routines are not effectively controlling symptoms, see a healthcare provider or dermatologist for specialized treatments.'
        },
        'Psoriasis': {
            'description': 'Psoriasis is a skin disease that causes a rash with itchy, scaly patches, most commonly on the knees, elbows, trunk and scalp. It is a common, long-term (chronic) disease with no cure, and it tends to go through cycles, flaring for a few weeks or months, then subsiding for a while.',
            'recommendation': 'Treatment focuses on removing scales and stopping skin cells from growing so quickly. Topical ointments, light therapy, and medications can offer relief, so consulting with a dermatologist is highly recommended.'
        },
        'Fungal Infection': {
            'description': 'Also known as tinea, a fungal skin infection is caused by a fungus. Common types include athlete\'s foot, jock itch, and ringworm. Fungi thrive in warm, moist environments and can be highly contagious.',
            'recommendation': 'Keep the affected area clean and dry, and consider using over-the-counter antifungal creams. If the infection does not improve after a couple of weeks, or if it spreads, seek medical advice.'
        }
    }

    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'model', 'saved_model', 'best_model.h5')
        
        self.model = None
        self.temperature = 1.0

        if tf is not None and os.path.exists(model_path):
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.model = tf.keras.models.load_model(model_path)
                
                temp_path = os.path.join(base_dir, 'model', 'temperature.json')
                if os.path.exists(temp_path):
                    with open(temp_path, 'r') as f:
                        t_data = json.load(f)
                        self.temperature = t_data.get('temperature', 1.0)
                        
                # Warm up
                dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
                self.model.predict(dummy, verbose=0)
            except Exception as e:
                print(f"Failed to load model: {e}")

    def preprocess(self, image_bytes):
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception:
            raise ValueError("Corrupted image or unsupported format")

        width, height = img.size
        if width < 50 or height < 50:
            raise ValueError("Image too small (< 50x50 px)")

        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized).astype(np.float32)

        # Normalize to [0,1]
        img_array = img_array / 255.0

        # Subtract ImageNet per-channel mean and divide by std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch, width, height

    def predict(self, image_bytes):
        start_time = time.time()

        img_batch, width, height = self.preprocess(image_bytes)

        if self.model is None:
            # For testing without a model
            probs = np.array([[0.1, 0.8, 0.05, 0.05]])
        else:
            if self.temperature != 1.0:
                # Apply temperature scaling to logits
                logits = self.model.predict(img_batch, verbose=0)
                scaled_logits = logits / self.temperature
                probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
            else:
                probs = self.model.predict(img_batch, verbose=0)
                
        probs = probs[0]
        sorted_indices = np.argsort(probs)[::-1]
        
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1]
        
        top1_prob = float(probs[top1_idx])
        top1_class = self.CLASS_NAMES[top1_idx]
        top2_prob = float(probs[top2_idx])
        top2_class = self.CLASS_NAMES[top2_idx]

        conf_level = self._get_confidence_level(top1_prob)
        fallback = top1_prob < self.CONFIDENCE_THRESHOLDS['medium']

        all_probs = {self.CLASS_NAMES[i]: float(probs[i]) for i in range(len(self.CLASS_NAMES))}

        color_map = {'high': 'green', 'medium': 'yellow', 'low': 'red'}

        inference_time_ms = int((time.time() - start_time) * 1000)

        result = {
            "top1": {
                "class": top1_class,
                "confidence": top1_prob,
                "confidence_pct": f"{top1_prob * 100:.1f}%",
                "confidence_level": conf_level,
                "color_code": color_map[conf_level]
            },
            "top2": {
                "class": top2_class,
                "confidence": top2_prob,
                "confidence_pct": f"{top2_prob * 100:.1f}%"
            },
            "all_probabilities": all_probs,
            "fallback_warning": fallback,
            "condition_description": self.CONDITION_INFO[top1_class]['description'],
            "recommendation": self.CONDITION_INFO[top1_class]['recommendation'],
            "width": width,
            "height": height,
            "inference_time_ms": inference_time_ms
        }
        return result

    def _get_confidence_level(self, confidence):
        if confidence >= self.CONFIDENCE_THRESHOLDS['high']:
            return 'high'
        elif confidence >= self.CONFIDENCE_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'low'
