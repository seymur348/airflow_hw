# airflow_hw — планируемый конвейер данных на Apache Airflow

Проект переобучает модель классификации и считает прогнозы по расписанию.
По данным объявления об автомобиле модель предсказывает его ценовую категорию (`price_category`).

## Как это работает

DAG `car_price_prediction` запускается каждый день в 00:15 (`15 0 * * *`) и состоит из трёх задач:

```
validate_data  >>  pipeline  >>  predict
```

| Задача          | Файл                  | Что делает                                                                 |
|-----------------|-----------------------|----------------------------------------------------------------------------|
| `validate_data` | `modules/validate.py` | проверяет обучающую таблицу: колонки, число строк, целевая колонка, пропуски |
| `pipeline` | `modules/pipeline.py` | читает `data/train/homework.csv`, обучает модели, сохраняет лучшую в pkl    |
| `predict`  | `modules/predict.py`  | берёт последнюю модель, читает `data/test/*.json`, пишет `prediction.csv`   |

Каждая следующая задача стартует только после успеха предыдущей: если данные не прошли проверку, модель не переобучается.
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
├── modules/validate.py     # проверка качества данных
├── modules/pipeline.py     # обучение
├── modules/predict.py      # прогноз
├── scripts/make_sample_data.py  # генератор синтетических данных
├── tests/test_modules.py   # pytest-тесты
├── requirements.txt
├── Dockerfile, docker-compose.yml
└── data/
    ├── train/homework.csv  # обучающая таблица с колонкой price_category
    ├── test/*.json         # объявления для прогноза
    ├── models/             # сохранённые модели (в git не хранятся)
    └── predictions/        # результат
```

**Важно: `data/train/homework.csv` и `data/test/*.json` в репозитории синтетические.** Их создаёт
`scripts/make_sample_data.py`, они нужны только для проверки, что конвейер работает.
Метрики на них ничего не говорят о реальной модели. Для настоящих результатов положите свой `homework.csv`
с теми же колонками и тестовые JSON-файлы.

## Запуск

```bash
pip install -r requirements.txt
export PROJECT_PATH=~/airflow_hw
export AIRFLOW__CORE__DAGS_FOLDER=$PROJECT_PATH/dags
mkdir -p $PROJECT_PATH/data/models

airflow standalone                          # веб-интерфейс и планировщик
airflow dags trigger car_price_prediction   # ручной запуск
```

DAG использует `airflow.providers.standard.operators.python` (Airflow 3).
Проверено запуском `airflow dags test car_price_prediction` на Airflow 3.3.2 и Python 3.13: все три задачи в статусе success.

Без Airflow задачи можно проверить из корня проекта:

```bash
python scripts/make_sample_data.py   # только если нет своих данных
python modules/validate.py
python modules/pipeline.py
python modules/predict.py
pytest tests
```

### Docker

```bash
docker compose up --build
```

Интерфейс Airflow будет на http://localhost:8080 (логин `admin`, пароль печатается в логе контейнера).
Docker-файлы не запускались в среде разработки, где проект проверялся, их нужно проверить у себя.

## Пример результата

Файл `data/predictions/prediction.csv` в репозитории — результат прежнего запуска на реальных данных:

```csv
car_id,pred
7315173150,low
7316152972,medium
```

Запуск на синтетических данных из репозитория дал такие accuracy (4-кратная кросс-валидация):
LogisticRegression 0,821; SVC 0,789; RandomForest 0,766. Лучшая модель — LogisticRegression.
Цифры показывают только, что конвейер работает, а не качество на реальных данных.

## Изменения совместимости

`LogisticRegression(solver='liblinear')` в новых версиях scikit-learn не поддерживает три класса и падает.
В `pipeline.py` заменено на `solver='lbfgs', max_iter=1000`.

## Что можно улучшить

- уведомление о сбое (почта или Telegram);
- запуск Airflow в Docker;
- не переобучать модель, если данные не изменились.
