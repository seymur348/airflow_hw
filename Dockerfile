FROM apache/airflow:3.0.6

USER airflow
RUN pip install --no-cache-dir pandas scikit-learn dill

ENV PROJECT_PATH=/opt/airflow/airflow_hw \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/airflow_hw/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False

WORKDIR /opt/airflow/airflow_hw
COPY --chown=airflow:root . /opt/airflow/airflow_hw
RUN mkdir -p data/models data/predictions

CMD ["airflow", "standalone"]
