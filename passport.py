import pandas as pd
import joblib
import yaml
import json
import os
import subprocess

import os
import subprocess

def get_git_metadata():
    commit = (
        os.getenv("CI_COMMIT_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    )

    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        tag = None

    repo_url = (
        os.getenv("CI_REPOSITORY_URL")
        or subprocess.check_output(
            ["git", "remote", "get-url", "origin"]
        ).decode().strip()
    )

    if repo_url.startswith("git@"):
        repo_url = repo_url.replace("git@", "https://").replace(":", "/")
    repo_url = repo_url.removesuffix(".git")

    return repo_url, commit, tag

repo_url, commit, tag = get_git_metadata()

commit_url = f"{repo_url}/commit/{commit}"
tag_url = (f"{repo_url}/releases/tag/{tag}" if tag else None)

def main():
    # ---------- Загрузка настроек ----------
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    dataset_name = params["data"]["name"]
    model_params = params["model"]
    model_type = params["model"]["type"]
    model_ver = f"{model_params['n_estimators']}_{model_params['max_depth']}_{model_params['random_state']}"
    df = pd.read_csv(f"data/{dataset_name}.csv")
    X_train = pd.read_csv("data/train_features.csv")
    X_test = pd.read_csv("data/test_features.csv")
    # Пути
    model_filename = f"{model_type}_{model_params['n_estimators']}_{model_params['max_depth']}_{model_params['random_state']}--{dataset_name}.pkl"
    model_path = os.path.join("models", model_filename)
    metrics_path = f"metrics/evaluate--{model_type}_{model_params['n_estimators']}_{model_params['max_depth']}_{model_params['random_state']}--{dataset_name}.json"

    # ---------- Сбор данных ----------
    # 1. Свойства модели
    model = joblib.load(model_path)
    
    # 2. Метрики (берем из уже созданного файла)
    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)
    test_metrics = all_metrics.get("test", {})

    # 3. Признаки (читаем одну строку из train, чтобы узнать колонки)
    X_sample = pd.read_csv("data/train_features.csv", nrows=1)
    
    # ---------- Сборка JSON ----------
    passport = {
        "sector_id": params["data"].get("sector_id", "unknown"),
        "model_name": model_filename,
        "model_version": model_ver,
        "model_type": model_type,
        "gitlab_commit_hash": commit_url,
        "gitlab_download_url": tag_url,
        "hyperparameters": model.get_params(),
        "features": {
            "features": X_sample.columns.tolist()
        },
        "training_data_info": {
            "dataset_size": int(len(df)), # Весь файл
            "train_size": int(X_train.shape[0]), # 80% (например)
            "test_size": int(X_test.shape[0])   # 20%
        },
        "metrics": {
            "accuracy": test_metrics.get("accuracy"),
            "precision": test_metrics.get("precision"),
            "recall": test_metrics.get("recall"),
            "f1_score": test_metrics.get("f1"),
            "auc": test_metrics.get("roc_auc")
        }
    }

    # ---------- Сохранение ----------
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/model_passport.json", "w", encoding='utf-8') as f:
        json.dump(passport, f, indent=4, ensure_ascii=False)

    print(f"Passport created at metrics/model_passport.json")

if __name__ == "__main__":
    main()