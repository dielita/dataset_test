import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

# ---------- Загрузка параметров ----------
with open("params.yaml") as f:
    params = yaml.safe_load(f)

dataset_name = params["data"]["name"]
target_col = params["data"]["target"]
features_cfg = params["features"]["columns"]
test_size = params["data"].get("test_size", 0.2)
random_state = params["model"]["random_state"]

# ---------- Загрузка данных ----------
df = pd.read_csv(f"data/{dataset_name}.csv")

# Выделение признаков и таргета
y = df[target_col]
if features_cfg == "all":
    X = df.drop(columns=[target_col])
else:
    X = df[features_cfg]

# One-Hot Encoding (делаем его ДО сплита, чтобы колонки в train/test совпали)
X = pd.get_dummies(X)

# ---------- Тот самый Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y  
)

# ---------- Сохранение результатов ----------
os.makedirs("data", exist_ok=True)

# Сохраняем признаки и таргет отдельно для удобства
X_train.to_csv("data/train_features.csv", index=False)
X_test.to_csv("data/test_features.csv", index=False)
y_train.to_csv("data/train_target.csv", index=False)
y_test.to_csv("data/test_target.csv", index=False)

print(f"Data split finished!")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")