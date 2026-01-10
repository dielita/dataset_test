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

# ---------- ПРЕДОБРАБОТКА ----------

# 1. Извлекаем месяц из погодной даты (для учета сезонности)
df['weather_date'] = pd.to_datetime(df['weather_date'])
df['month'] = df['weather_date'].dt.month

# 2. Удаляем "читерские" колонки и неинформативные тексты
# Удаляем: weather_date (уже взяли месяц), Дата схода, Время (утечка), Лавинный очаг (слишком много категорий)
cols_to_drop = [target_col, 'weather_date', 'Дата схода', 'Время', 'Лавинный очаг (место схода)']
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 3. Заполняем пропуски в числовых колонках (например, Площадь)
# Модели не умеют работать с NaN
X['Площадь'] = X['Площадь'].fillna(0)

# 4. Кодируем только реальные категории (Экспозиция, Условия схода)
# get_dummies теперь не создаст тысячи колонок с датами!
X = pd.get_dummies(X)

y = df[target_col]

# ---------- Train/Test Split ----------
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