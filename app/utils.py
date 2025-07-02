# -----------------------------
# File: app/utils.py
# -----------------------------

import mlflow
import mlflow.sklearn
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:8080")

def load_model(alias):
    model_uri = f"models:/production model@{alias}"
    return mlflow.sklearn.load_model(model_uri)

champion_model = load_model("champion")
challenger_model = load_model("challenger")

def predict(model_type: str, features: list):
    model = champion_model if model_type == "champion" else challenger_model
    arr = np.array(features).reshape(1, -1)
    return int(model.predict(arr)[0])

feature_ranges = {
    "alcohol": (11, 15),
    "malic_acid": (0.7, 6),
    "ash": (1.5, 3.5),
    "alcalinity_of_ash": (10, 31),
    "magnesium": (70, 165),
    "total_phenols": (0.9, 4),
    "flavanoids": (0.3, 5.2),
    "nonflavanoid_phenols": (0.1, 1),
    "proanthocyanins": (0.3, 4),
    "color_intensity": (1, 15),
    "hue": (0.3, 1.8),
    "od280_od315_of_diluted_wines": (1, 5),
    "proline": (250, 1700),
}

def validate_features(data) -> bool:
    for field, (min_val, max_val) in feature_ranges.items():
        val = getattr(data, field)
        if not (min_val <= val <= max_val):
            return False
    return True