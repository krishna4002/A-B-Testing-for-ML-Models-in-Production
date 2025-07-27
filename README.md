# A/B Testing for ML Models in Production

This project demonstrates a complete A/B Testing system for evaluating and comparing machine learning models (Champion vs Challenger) in a real-time production environment. It includes model registration via MLflow, versioning, prediction API, user input validation, logging to SQLite, and a live dashboard for monitoring performance.

## Key Features

- Automatically selects between Champion and Challenger models for prediction using a random value.
- Accepts user input through a FastAPI POST endpoint with strict range validation.
- Supports optional true label input for accuracy tracking.
- Logs prediction, random value, model used, and correctness into an SQLite database.
- Interactive HTML + JS dashboard auto-updated with prediction statistics and logs.
- Uses MLflow for managing model versions and loading models with aliases.
- Designed for local experimentation and real-time monitoring.

## Why A/B Testing is Required in Production

- **Real-world validation:** Lab metrics (accuracy, F1-score) may not reflect actual user behavior. A/B testing helps validate models in real environments.
- **Safe experimentation:** New models can be evaluated without replacing the current model for all users.
- **Incremental rollout:** You can gradually test a model on a portion of traffic before full deployment.
- **Performance monitoring:** Live feedback from real inputs helps compare models beyond training/test data.
- **Decision-making:** A/B results inform product and engineering teams whether a model upgrade is justified.

## Project Structure

```
ab_testing_ml_production/
├── app/
│   ├── main.py             # FastAPI API logic
│   ├── utils.py            # Model loading and prediction logic
│   └── logger.py           # SQLite logging logic
├── dashboard/
│   ├── index.html          # Dashboard UI
│   ├── dashboard.js        # JS logic to update dashboard
│   └── styles.css          # Dashboard styles
├── logs/
│   └── predict.db          # SQLite DB file
├── train_models.py         # Model training and MLflow registration
├── requirements.txt        # Required Python packages
└── README.md
```

## Technologies Used

- **Python**, **FastAPI**, **MLflow**, **Scikit-learn**, **SQLite3**
- **Pandas**, **NumPy**, **Pydantic**
- **HTML, **CSS**, **JavaScript** 
- **Uvicorn** for running the backend server
- **Postman** for API testing (optional)

## How It Works

1. Train multiple ML models with different hyperparameters using `train_models.py`.
2. Log all models into MLflow and manually tag best two as `champion` and `challenger`.
3. When a user hits `/predict`, one of the two models is randomly selected.
4. The prediction is made and logged along with metadata into SQLite.
5. If `true_label` is provided, it evaluates correctness and logs it.
6. A real-time dashboard (HTML + JS) fetches and displays recent predictions, model accuracy, and usage stats.

## API Usage

**Endpoint:** `POST /predict`  
**Content-Type:** `application/json`  
**Sample Input:**

```json
{
  "alcohol": 13.5,
  "malic_acid": 1.2,
  "ash": 2.3,
  "alcalinity_of_ash": 15.0,
  "magnesium": 100,
  "total_phenols": 2.5,
  "flavanoids": 2.1,
  "nonflavanoid_phenols": 0.3,
  "proanthocyanins": 1.4,
  "color_intensity": 4.8,
  "hue": 1.0,
  "od280_od315_of_diluted_wines": 3.0,
  "proline": 950,
  "true_label": 1
}
```

**Sample Response:**

```json
{
  "user_id": "a1b2c3d4",
  "model_used": "champion",
  "random_value": 0.6842,
  "prediction": 1,
  "true_label": 1,
  "correct": true
}
```

## Dashboard Access

- Run the FastAPI app and open: `http://localhost:8000/`
- Displays:
  - Total Predictions
  - Accuracy for Champion and Challenger
  - Recent 100 Predictions Log
- Auto-refreshes to reflect live activity.

## Run This Project

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train and register models
python train_models.py

# Step 3: Start the FastAPI server
uvicorn app.main:app --reload

# Step 4: Open dashboard in browser
http://127.0.0.1:8000/
```

Make sure MLflow is running locally at `http://127.0.0.1:8080` and that champion and challenger aliases are set on two versions of the model registered as "production model".

## 📂 SQLite Log Fields

Each prediction is stored in logs/predict.db with fields:

- `timestamp`
- `user_id`
- `model_used`
- `prediction`
- `random_value`
- `true_label`
- `correct`
