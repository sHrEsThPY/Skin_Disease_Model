import sqlite3
import os
import datetime

class PredictionLogger:
    def __init__(self, db_path='database/skinai_logs.db'):
        if db_path == ':memory:':
            self.db_path = db_path
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(base_dir, db_path)
            # Ensure dir exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                top1_class TEXT,
                top1_confidence REAL,
                top2_class TEXT,
                top2_confidence REAL,
                image_width INTEGER,
                image_height INTEGER,
                inference_time_ms INTEGER,
                fallback_triggered BOOLEAN
            )
        ''')
        conn.commit()
        conn.close()

    def log_prediction(self, top1_class, top1_conf, top2_class, top2_conf, 
                       img_width, img_height, inference_ms, fallback):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.datetime.utcnow().isoformat()
        c.execute('''
            INSERT INTO predictions 
            (timestamp, top1_class, top1_confidence, top2_class, top2_confidence, 
             image_width, image_height, inference_time_ms, fallback_triggered)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, top1_class, float(top1_conf), top2_class, float(top2_conf), 
              img_width, img_height, inference_ms, fallback))
        conn.commit()
        conn.close()

    def get_recent(self, n=100):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (n,))
            rows = c.fetchall()
            conn.close()
            return [dict(ix) for ix in rows]
        except:
            return []

    def get_stats(self):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM predictions')
            total = c.fetchone()[0]
            
            c.execute('SELECT AVG(top1_confidence) FROM predictions')
            avg_conf = c.fetchone()[0] or 0.0
            
            c.execute('SELECT top1_class, COUNT(*) FROM predictions GROUP BY top1_class')
            dist = {row[0]: row[1] for row in c.fetchall()}
            
            conn.close()
            return {
                "total": total,
                "avg_confidence": avg_conf,
                "class_distribution": dist
            }
        except:
            return {"total": 0, "avg_confidence": 0.0, "class_distribution": {}}
