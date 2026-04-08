from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from model.inference import SkinAIPredictor
from database.logs import PredictionLogger
import os

app = Flask(__name__)
# Limit upload size globally to 5MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["30 per minute"]
)

# Initialize Predictor and Logger
predictor = SkinAIPredictor()
logger = PredictionLogger()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
@limiter.limit("30 per minute")
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image field found", "code": "400"}), 400

    if file.filename:
        if not allowed_file(file.filename) and request.content_type not in ['image/png', 'image/jpeg']:
            return jsonify({"error": "File type not supported", "code": "400"}), 400

    image_bytes = file.read()
    if len(image_bytes) == 0:
        return jsonify({"error": "Empty file", "code": "400"}), 400
        
    try:
        result = predictor.predict(image_bytes)
        
        try:
            logger.log_prediction(
                top1_class=result['top1']['class'],
                top1_conf=result['top1']['confidence'],
                top2_class=result['top2']['class'],
                top2_conf=result['top2']['confidence'],
                img_width=result.get('width', 0),
                img_height=result.get('height', 0),
                inference_ms=result.get('inference_time_ms', 0),
                fallback=result['fallback_warning']
            )
        except Exception as e:
            print("Log Error:", e)

        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve), "code": "400"}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "code": "500"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok", 
        "model_loaded": predictor.model is not None if hasattr(predictor, 'model') else False, 
        "classes": SkinAIPredictor.CLASS_NAMES, 
        "version": "1.0.0"
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    recent = logger.get_recent(100)
    return jsonify(recent), 200

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
