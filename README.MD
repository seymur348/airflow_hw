# airflow_hw — планируемый конвейер данных на Apache Airflow

Проект переобучает модель классификации и считает прогнозы по расписанию.
По данным объявления об автомобиле модель предсказывает его ценовую категорию (`price_category`).

## Как это работает

DAG `car_price_prediction` запускается каждый день в 00:15 (`15 0 * * *`) и состоит из двух задач:

```
pipeline  >>  predict
```

| Задача     | Файл                  | Что делает                                                                 |
|------------|-----------------------|----------------------------------------------------------------------------|
| `pipeline` | `modules/pipeline.py` | читает `data/train/homework.csv`, обучает модели, сохраняет лучшую в pkl    |
| `predict`  | `modules/predict.py`  | берёт последнюю модель, читает `data/test/*.json`, пишет `prediction.csv`   |

`predict` стартует только после успешного завершения `pipeline`.
При сбое Airflow делает одну повторную попытку через 1 минуту (`retries=1`, `retry_delay=1 min`).
`catchup=False`: пропущенные даты с 2022 года не догоняются.

### Обучение (`pipeline`)

1. Удаляются ненужные колонки: `id`, `url`, `region`, `region_url`, `price`, `manufacturer`, `image_url`, `description`, `posting_date`, `lat`, `long`.
2. Выбросы в `year` обрезаются по границам Q1 − 1,5·IQR и Q3 + 1,5·IQR.
3. Добавляются признаки `short_model` (первое слово модели) и `age_category` (`new` при `year > 2013`, `old` при `year < 2006`, иначе `average`).
4. Числовые признаки: заполнение медианой и `StandardScaler`. Категориальные: самое частое значение и `OneHotEncoder`.
5. Сравниваются `LogisticRegression`, `RandomForestClassifier` и `SVC` по accuracy на 4-кратной кросс-валидации.
6. Лучший пайплайн обучается на всех данных и сохраняется через `dill` в `data/models/cars_pipe_<ГГГГММДДччмм>.pkl`.

### Прогноз (`predict`)

Загружается последний по имени `.pkl` из `data/models`, все `.json` из `data/test` собираются в таблицу (один файл — одна строка).
Результат пишется в `data/predictions/prediction.csv` с колонками `car_id` и `pred`.

## Структура

```
airflow_hw/
├── dags/hw_dag.py          # описание DAG
├── modules/pipeline.py     # обучение
├── modules/predict.py      # прогноз
└── data/                   # создаётся локально, в git не хранится (кроме примера prediction.csv)
    ├── train/homework.csv  # обучающая таблица с колонкой price_category
    ├── test/*.json         # объявления для прогноза
    ├── models/             # сохранённые модели
    └── predictions/        # результат
```

Папка `data` в `.gitignore`: обучающие данные, тестовые файлы и модели нужно положить локально.

## Запуск

```bash
pip install apache-airflow pandas scikit-learn dill
export PROJECT_PATH=~/airflow_hw
export AIRFLOW__CORE__DAGS_FOLDER=$PROJECT_PATH/dags
mkdir -p $PROJECT_PATH/data/models

airflow standalone                          # веб-интерфейс и планировщик
airflow dags trigger car_price_prediction   # ручной запуск
```

DAG использует `airflow.providers.standard.operators.python` (Airflow 3).

Без Airflow обе задачи можно проверить из корня проекта:

```bash
python modules/pipeline.py
python modules/predict.py
```

## Пример результата

```csv
car_id,pred
7315173150,low
7316152972,medium
```

## Что можно улучшить

- задача проверки качества данных перед обучением;
- уведомление о сбое (почта или Telegram);
- запуск Airflow в Docker;
- не переобучать модель, если данные не изменились.
