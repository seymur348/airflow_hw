import datetime as dt
import os
import sys

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# путь к проекту
path = os.path.expanduser('~/airflow_hw')

os.environ['PROJECT_PATH'] = path
sys.path.insert(0, path)

from modules.pipeline import pipeline
from modules.predict import predict

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
}

with DAG(
        dag_id='car_price_prediction',
        schedule="15 0 * * *",   # замена schedule_interval
        default_args=args,
        catchup=False            # чтобы не запускался за прошлые даты
) as dag:

    pipeline_task = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
    )

    predict_task = PythonOperator(
        task_id='predict',
        python_callable=predict,
    )

    pipeline_task >> predict_task

