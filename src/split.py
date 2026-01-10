import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

def main():
    # ---------- 1. Загрузка параметров ----------
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    dataset_name = params["data"]["name"]
    target_col = params["data"]["target"]
    test_size = params["data"].get("test_size", 0.2)
    random_state = params["model"]["random_state"]

    # ---------- 2. Загрузка данных ----------
    input_path = f"data/{dataset_name}.csv"
    if not os.path.exists(input_path):
        print(f"Ошибка: Файл {input_path} не найден.")
        return

    df = pd.read_csv(input_path)
    print(f"Загружен датасет: {dataset_name} ({len(df)} строк)")

    # ---------- 3. Предобработка и очистка от утечек ----------
    
    # Преобразуем дату в формат datetime
    df['weather_date'] = pd.to_datetime(df['weather_date'])
    
    # Извлекаем безопасные признаки времени (те, что известны заранее)
    df['month'] = df['weather_date'].dt.month
    
    # Список колонок, которые являются "читерскими" (утечки данных)
    # Это данные, которые заполняются ПОСЛЕ или ВО ВРЕМЯ схода лавины
    cols_to_drop = [
        target_col, 
        'weather_date', 
        'Дата схода', 
        'Время', 
        'Лавинный очаг (место схода)', 
        'Площадь', 
        'Объем лавины'
    ]

    # Автоматически находим и удаляем колонки с результатами (Экспозиция, Условия схода)
    # Если их оставить, модель будет просто "угадывать" по ним факт лавины
    leakage_prefixes = ['Условия схода', 'Экспозиция']
    dynamic_drops = [c for c in df.columns if any(pre in c for pre in leakage_prefixes)]
    
    final_drop_list = list(set(cols_to_drop + dynamic_drops))
    
    # Логика для часового датасета
    if 'relative_humidity_2m' in df.columns:
        print("Режим: Часовой датасет. Добавляю признак 'hour'.")
        df['hour'] = df['weather_date'].dt.hour
    else:
        print("Режим: Суточный датасет.")

    # Формируем матрицу признаков X и целевой вектор y
    X = df.drop(columns=[c for c in final_drop_list if c in df.columns])
    y = df[target_col]

    # Заполняем пустые значения (NaN) нулями, чтобы обучение не упало
    X = X.fillna(0)

    # Преобразуем оставшиеся текстовые категории (если есть) в числа (0/1)
    X = pd.get_dummies(X)

    print(f"Удалено колонок-утечек: {len(dynamic_drops) + (len(cols_to_drop)-1)}")
    print(f"Итого признаков для обучения: {X.shape[1]}")

    # ---------- 4. Разделение на Train/Test ----------
    # Используем stratify=y, так как лавины — редкое событие
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  
    )

    # ---------- 5. Сохранение ----------
    os.makedirs("data", exist_ok=True)
    
    X_train.to_csv("data/train_features.csv", index=False)
    X_test.to_csv("data/test_features.csv", index=False)
    y_train.to_csv("data/train_target.csv", index=False)
    y_test.to_csv("data/test_target.csv", index=False)

    print(f"Файлы сохранены в папку data/")

if __name__ == "__main__":
    main()