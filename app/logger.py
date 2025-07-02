import sqlite3
import os
from datetime import datetime
import pytz  # Make sure to install this: pip install pytz

LOG_DB = "logs/predict.db"

def init_db():
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect(LOG_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            timestamp TEXT,
            user_id TEXT,
            model_used TEXT,
            prediction INTEGER,
            random_value REAL,
            true_label TEXT,
            correct TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(user_id, random_value, model_used, prediction, true_label):
    conn = sqlite3.connect(LOG_DB)
    cursor = conn.cursor()

    correct = None
    if true_label != "-":
        correct = str(int(prediction == int(true_label)))

    # Get current time in IST
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO predictions (timestamp, user_id, model_used, prediction, random_value, true_label, correct)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (now, user_id, model_used, prediction, random_value, true_label, correct))

    conn.commit()
    conn.close()
