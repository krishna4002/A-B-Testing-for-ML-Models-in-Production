from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uuid
import numpy as np
import random
import sqlite3
from app.logger import log_prediction, init_db
from app.utils import predict, validate_features

app = FastAPI()
init_db()

class InputData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float
    true_label: Optional[int] = None

@app.post("/predict")
def predict_entry(data: InputData):
    features = [
        data.alcohol, data.malic_acid, data.ash, data.alcalinity_of_ash,
        data.magnesium, data.total_phenols, data.flavanoids,
        data.nonflavanoid_phenols, data.proanthocyanins,
        data.color_intensity, data.hue,
        data.od280_od315_of_diluted_wines, data.proline
    ]

    if not validate_features(data):
        raise HTTPException(status_code=400, detail="Invalid input range.")

    user_id = str(uuid.uuid4())[:8]
    r = round(random.random(), 4)
    model_choice = "champion" if r > 0.5 else "challenger"
    prediction = predict(model_choice, features)

    true_label = data.true_label if data.true_label is not None else "-"
    log_prediction(user_id, r, model_choice, prediction, true_label)

    correct = prediction == data.true_label if data.true_label is not None else None

    return {
        "user_id": user_id,
        "random_value": r,
        "model_used": model_choice,
        "prediction": prediction,
        "true_label": data.true_label,
        "correct": correct
    }

@app.get("/predictions")
def get_predictions():
    conn = sqlite3.connect("logs/predict.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 100")
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "timestamp": row[0],
            "user_id": row[1],
            "random_value": row[2],
            "model_used": row[3],
            "prediction": row[4],
            "true_label": row[5],
            "correct": row[6]
        } for row in rows
    ]

@app.get("/stats")
def get_statistics():
    conn = sqlite3.connect("logs/predict.db")
    cursor = conn.cursor()

    # Total predictions (all)
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    # Model usage (all)
    cursor.execute("SELECT model_used, COUNT(*) FROM predictions GROUP BY model_used")
    model_usage = {row[0]: row[1] for row in cursor.fetchall()}

    # Accuracy only for predictions with known true_label (not '-', '', or NULL)
    cursor.execute("""
        SELECT model_used, correct
        FROM predictions
        WHERE true_label IS NOT NULL AND true_label != ''
              AND true_label != '-'
    """)
    accuracy = {"champion": [0, 0], "challenger": [0, 0]}  # [correct, total]
    model_correctness = {
        "champion": {"correct": 0, "incorrect": 0},
        "challenger": {"correct": 0, "incorrect": 0}
    }

    for model_used, correct in cursor.fetchall():
        model = model_used.lower()
        if model in accuracy:
            accuracy[model][1] += 1
            if str(correct) == "1":
                accuracy[model][0] += 1
                model_correctness[model]["correct"] += 1
            elif str(correct) == "0":
                model_correctness[model]["incorrect"] += 1

    # Accuracy percentages
    accuracy_percent = {
        m: round((c / t) * 100, 2) if t > 0 else 0.0
        for m, (c, t) in accuracy.items()
    }

    # Class distribution
    cursor.execute("""
        SELECT prediction, model_used, COUNT(*)
        FROM predictions
        GROUP BY prediction, model_used
    """)
    class_distribution_by_model = {}
    for pred, model_used, count in cursor.fetchall():
        pred = str(pred)
        if pred not in class_distribution_by_model:
            class_distribution_by_model[pred] = {"champion": 0, "challenger": 0}
        class_distribution_by_model[pred][model_used] = count

    conn.close()

    return {
        "total_predictions": total_predictions,
        "model_usage": model_usage,
        "accuracy_percent": accuracy_percent,
        "class_distribution_by_model": class_distribution_by_model,
        "model_correctness": model_correctness
    }



app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def get_dashboard():
    return FileResponse("dashboard/index.html")
