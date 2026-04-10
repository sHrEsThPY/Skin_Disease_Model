"""
SkinAI — Flask Application
HAM10000 EfficientNetB0 skin lesion classifier
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import io, base64, logging
from flask import Flask, request, jsonify, render_template
from model.inference import SkinAIPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB

predictor = SkinAIPredictor()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': predictor.model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    img_bytes = None

    # Accept multipart file upload
    if 'file' in request.files and request.files['file'].filename != '':
        img_bytes = request.files['file'].read()

    # Accept base64 JSON (from camera capture)
    elif request.is_json:
        data  = request.get_json(silent=True) or {}
        b64   = data.get('image', '').split(',')[-1]
        try:
            img_bytes = base64.b64decode(b64)
        except Exception as e:
            return jsonify({'error': f'Invalid base64 data: {e}'}), 400

    if not img_bytes:
        return jsonify({'error': 'No image provided'}), 400

    result = predictor.predict(img_bytes)
    return jsonify(result)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
