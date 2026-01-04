import pandas as pd
import joblib
import yaml
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split   

# ---------- Load params ----------
with open("params.yaml") as f:
    params = yaml.safe_load(f)

def get_exp_id():
    return os.getenv("DVC_EXP_NAME", "noexp")

dataset_name = params["data"]["name"]
features_cfg = params["features"]["columns"]
model_params = params["model"]
target_col = params["data"]["target"]

exp_id = get_exp_id()

# Load dataset 
df = pd.read_csv(f"data/{dataset_name}.csv")  # импортированный .dvc развернёт CSV

y = df[target_col]
if features_cfg == "all":
    X = df.drop(columns=[target_col])
else:
    X = df[features_cfg]

X = pd.get_dummies(X)

X_train = pd.read_csv("data/train_features.csv")
y_train = pd.read_csv("data/train_target.csv").values.ravel() # ravel для корректной формы

# Train Model
model = RandomForestClassifier(
    n_estimators=model_params["n_estimators"],
    max_depth=model_params["max_depth"],
    random_state=model_params["random_state"]
    class_weight="balanced"
)

model.fit(X_train, y_train)

X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# Save Model
model_filename = f"{model_params['type']}--{dataset_name}.pkl"
save_path = os.path.join("models", model_filename)
os.makedirs("models", exist_ok=True)

joblib.dump(model, save_path)

print(f"Saved model: {save_path}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
