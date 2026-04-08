import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Central paths
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'saved_model', 'best_model.h5')
DB_PATH = os.path.join(BASE_DIR, 'database', 'skinai_logs.db')

CLASS_NAMES = ['Acne Vulgaris', 'Eczema', 'Psoriasis', 'Fungal Infection']
CONFIDENCE_THRESHOLDS = {'high': 0.80, 'medium': 0.60}
