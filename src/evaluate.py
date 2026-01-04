import pandas as pd
import yaml
import joblib
import json
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

dataset_name = params["data"]["name"]
target_col = params["data"]["target"]
features_cfg = params["features"]["columns"]
model_type = params["model"]["type"]
model_params = params["model"]
test_size = params["data"].get("test_size", 0.2)

def get_exp_id():
    return os.getenv("DVC_EXP_NAME", "noexp")

exp_id = get_exp_id()

model_filename = f"{model_type}_{model_params['n_estimators']}_{model_params['max_depth']}_{model_params['random_state']}--{dataset_name}.pkl"
model_path = os.path.join("models", model_filename)

model = joblib.load(model_path)

# Load data
df = pd.read_csv(f"data/{dataset_name}.csv")

y = df[target_col]

if features_cfg == "all":
    X = df.drop(columns=[target_col])
else:
    X = df[features_cfg]

X = pd.get_dummies(X)

X_train = pd.read_csv("data/train_features.csv")
y_train = pd.read_csv("data/train_target.csv").values.ravel()
X_test = pd.read_csv("data/test_features.csv")
y_test = pd.read_csv("data/test_target.csv").values.ravel()

y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

y_train_pred = model.predict(X_train)
y_train_proba = model.predict_proba(X_train)[:, 1]

def calculate_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "support_positive": int(y_true.sum()),
        "support_total": int(len(y_true)),
    }

metrics = {
    "train": calculate_metrics(y_train, y_train_pred, y_train_proba),
    "test": calculate_metrics(y_test, y_test_pred, y_test_proba)
}

# Save metrics
os.makedirs("metrics", exist_ok=True)
metrics_path = f"metrics/evaluate--{model_type}_{model_params['n_estimators']}_{model_params['max_depth']}_{model_params['random_state']}--{dataset_name}.json"

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)


# ---------- Pretty print ----------
#print("Evaluation metrics:")
#print(json.dumps(metrics, indent=2))

#print("\nClassification report:")
#print(classification_report(y_test, y_pred, zero_division=0))

print(f"\nSaved metrics: {metrics_path}")
