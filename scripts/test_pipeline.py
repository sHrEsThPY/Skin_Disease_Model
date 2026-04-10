import os
import sys
import numpy as np
import io
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.inference import SkinAIPredictor
from database.logs import PredictionLogger

def create_dummy_image():
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def run_tests():
    total_tests = 4
    passed = 0
    print("Starting pipeline tests...")

    # 1. Preprocessing Test
    try:
        predictor = SkinAIPredictor()
        img_bytes = create_dummy_image()
        img_batch = predictor._preprocess(img_bytes)
        assert img_batch.shape == (1, 224, 224, 3)
        assert np.max(img_batch) <= 255.0 and np.min(img_batch) >= 0.0 # EfficientNet range

        print("✅ Preprocessing OK")
        passed += 1
    except AssertionError as e:
        print(f"❌ Preprocessing Test Failed: {e}")

    # 2. Inference Engine Test
    try:
        class MockModel:
            def predict(self, x, verbose=0):
                return np.array([[0.1, 0.8, 0.05, 0.05]])
        predictor.model = MockModel()
        res = predictor.predict(img_bytes)
        req_keys = ['top1', 'top2', 'all_probabilities', 'fallback_warning', 'condition_description', 'recommendation']
        for k in req_keys:
            assert k in res, f"Missing key {k}"
        sum_probs = sum(res['all_probabilities'].values())
        assert abs(sum_probs - 1.0) < 1e-4
        print("✅ Inference engine OK")
        passed += 1
    except AssertionError as e:
        print(f"❌ Inference engine Test Failed: {e}")

    # 3. Flask API Test
    try:
        from app import app
        app.config['TESTING'] = True
        client = app.test_client()
        
        # Health check
        resp = client.get('/health')
        assert resp.status_code == 200
        
        resp = client.post('/predict', data={'image': (io.BytesIO(img_bytes), 'test.png')})
        assert resp.status_code == 200
        assert 'top1' in resp.get_json()
        
        resp = client.post('/predict', data={})
        assert resp.status_code == 400
        
        resp = client.post('/predict', data={'image': (io.BytesIO(b'dummy'), 'test.txt')})
        assert resp.status_code == 400

        print("✅ API endpoints OK")
        passed += 1
    except AssertionError as e:
        print(f"❌ API points Test Failed: {e}")

    # 4. Database Logging Test
    try:
        test_db = 'database/test_skinai_logs.db'
        if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), test_db)):
            os.remove(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), test_db))
            
        logger = PredictionLogger(test_db)
        for _ in range(3):
            logger.log_prediction('Acne', 0.9, 'Eczema', 0.05, 224, 224, 100, False)
        recent = logger.get_recent(10)
        assert len(recent) == 3
        stats = logger.get_stats()
        assert stats['total'] == 3
        print("✅ Database logging OK")
        passed += 1
    except AssertionError as e:
        print(f"❌ Database logging Test Failed: {e}")

    print("══════════════════════════════════")
    if passed == total_tests:
        print("ALL TESTS PASSED ✅")
    else:
        print(f"{passed}/{total_tests} TESTS PASSED ❌")
    print("SkinAI pipeline is ready.")
    print("Run: python app.py")
    print("══════════════════════════════════")

if __name__ == '__main__':
    run_tests()
