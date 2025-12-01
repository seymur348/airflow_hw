import os
import dill
import pandas as pd
import json

path = os.environ.get("PROJECT_PATH", ".")

MODEL_DIR = f"{path}/data/models"
TEST_DIR = f"{path}/data/test"
PRED_DIR = f"{path}/data/predictions"


def load_model():
    """Находим последний сохранённый pkl-файл и загружаем."""
    files = sorted(os.listdir(MODEL_DIR))
    pkl_files = [f for f in files if f.endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError("Нет сохранённых моделей в /data/models/")

    latest_model = pkl_files[-1]
    model_path = os.path.join(MODEL_DIR, latest_model)

    with open(model_path, "rb") as f:
        model = dill.load(f)

    return model


def load_test_files():
    """Читаем все JSON из test/ и превращаем в DataFrame."""
    dataframes = []

    for file in os.listdir(TEST_DIR):
        if file.endswith(".json"):
            with open(os.path.join(TEST_DIR, file), "r") as f:
                j = json.load(f)
            df = pd.DataFrame([j])  # одна строка
            dataframes.append(df)

    if not dataframes:
        raise ValueError("В папке test/ нет JSON-файлов.")

    return pd.concat(dataframes, ignore_index=True)


def predict():
    os.makedirs(PRED_DIR, exist_ok=True)

    print("Загрузка модели...")
    model = load_model()

    print("Загрузка тестовых данных...")
    df = load_test_files()

    # Перед предсказанием модель ожидает такие же признаки, как и при обучении
    print("Выполняем предсказание...")
    preds = model.predict(df)

    result = pd.DataFrame({
        "car_id": df["id"],
        "pred": preds
    })

    out_file = os.path.join(PRED_DIR, "prediction.csv")
    result.to_csv(out_file, index=False)

    print(f"Готово! Файл сохранён: {out_file}")


if __name__ == "__main__":
    predict()
