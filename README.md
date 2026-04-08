# SkinAI — Skin Disease Detection Using Transfer Learning

**Team 21 | 213CSE4301 | Pattern & Anomaly Detection**

An AI-powered web application that classifies skin images into four common conditions: Acne Vulgaris, Eczema, Psoriasis, and Fungal Infection (Tinea).

Built using Transfer Learning with EfficientNet-B3.

## 1. Project Overview
- **Core Stack:** Python 3.10+, TensorFlow 2.x, Flask, Bootstrap 5.
- **Model:** EfficientNet-B3 backbone pretrained on ImageNet with a custom classification head.
- **Features:** File upload & real-time camera capture, immediate inference, comprehensive performance evaluation, SQLite logging.

## 2. Quick Start

Run these three commands to get the pipeline up and running (assuming you have Python 3.10+):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end tests to verify pipelines
python scripts/test_pipeline.py

# 3. Start the application locally
python app.py
```
Open `http://localhost:5000` in your web browser.

## 3. Dataset Setup (HAM10000)
The dataset script will download the image archives and metadata from the ISIC Archive. Run the following command:

```bash
python data/download_dataset.py
```
*Note: HAM10000 dataset is licensed under CC BY-NC 4.0. Some mappings have been performed since HAM10000 does not have exact matches for some proxy targets.*

## 4. Training Instructions
After downloading and preparing the dataset, execute the two-phase training process:
```bash
python model/train.py
```
Training logs and visualization curves will be saved in `logs/`. Model checkpoints are in `model/saved_model/`.

## 5. Evaluation
Evaluate the trained model via:
```bash
python model/evaluate.py
```
This script will produce a confusion matrix, ROC curves, PR curves, and compute Expected Calibration Error (with optional Temperature Scaling applied to improve reliability). Outputs will be placed in `logs/evaluation/`.

## 6. API Reference
- `GET /` : Loads the main interface.
- `GET /health` : Returns system status and loaded model details.
- `GET /metrics` : Returns recent predictions from SQLite logs.
- `POST /predict` : Validates incoming image stream or file upload -> Returns JSON with top 2 class predictions, confidence percentages, fallback warnings if $< 60\%$ threshold, and descriptions payload.

## 7. Model Architecture
```text
Inputs (224x224x3)
     │
[Data Augmentation Layers]
     │
[EfficientNet-B3 Backbone] (Frozen / Fine-tuned)
     │
GlobalAveragePooling2D
     │
Dense (512, ReLU)
     │
Dropout (0.4)
     │
Dense (128, ReLU)
     │
Dropout (0.3)
     │
Dense (4, Softmax) — Output Layer
```

## 8. Deployment

### Local Dev
```bash
python app.py
```

### Docker Compose
```bash
docker-compose up --build
```
Access at `http://localhost:8000`.

### Render
1. Create a "Web Service" in Render.
2. Select standard Docker environment.
3. Use the generated `render.yaml` and `Dockerfile`.
4. Deploy dynamically online. (Ensure sufficient memory for Keras inference).

### Heroku
```bash
heroku create skinai
heroku stack:set container
git push heroku main
```

## 9. Disclaimer
⚠ **SkinAI is intended for educational and preliminary screening purposes only. It does not constitute medical advice. Always consult a licensed dermatologist for diagnosis and treatment.**
