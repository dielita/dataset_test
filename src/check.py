import joblib
import pandas as pd
import yaml
import os

with open("params.yaml") as f:
    params = yaml.safe_load(f)

model_type = params["model"]["type"]
model_params = params["model"]
dataset_name = params["data"]["name"]

model_filename = f"{model_type}_{model_params['n_estimators']}_{model_params['max_depth']}_{model_params['random_state']}--{dataset_name}.pkl"

# Загружаем модель и данные
model_path = os.path.join("models", model_filename)

model = joblib.load(model_path)

X_train = pd.read_csv("data/train_features.csv")

# Извлекаем важность признаков
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("Топ-10 признаков, на которых учится модель:")
print(feature_importance_df.head(10))