import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# Set MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("A/B Testing for model")

# Load dataset
data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Registered model name
REGISTERED_MODEL_NAME = "production model"

# Updated: Define model configurations with multiple versions
model_configs = [
    {
        "model_name": "LogisticRegression",
        "model_class": LogisticRegression,
        "versions": [
            {"C": 0.1, "solver": "liblinear", "max_iter": 200},
            {"C": 1.0, "solver": "lbfgs", "max_iter": 300},
            {"C": 10.0, "solver": "saga", "max_iter": 500, "penalty": "elasticnet", "l1_ratio": 0.5}
        ]
    },
    {
        "model_name": "SVC",
        "model_class": SVC,
        "versions": [
            {"C": 0.5, "kernel": "linear", "probability": True},
            {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True},
            {"C": 2.0, "kernel": "poly", "degree": 3, "probability": True}
        ]
    },
    
]

# Training and logging loop
for config in model_configs:
    model_name = config["model_name"]
    model_class = config["model_class"]
    versions = config["versions"]

    for idx, params in enumerate(versions, start=1):
        with mlflow.start_run(run_name=f"{model_name}_V{idx}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("version", f"v{idx}")
            for key, val in params.items():
                mlflow.log_param(key, val)

            model = model_class(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="macro")
            rec = recall_score(y_test, preds, average="macro")
            f1 = f1_score(y_test, preds, average="macro")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME
            )

            print(f"✅ {model_name} (v{idx}) logged to MLflow with F1 = {f1:.4f}")