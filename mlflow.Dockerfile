# версия должна совпадать с mlflow в uv.lock
FROM ghcr.io/mlflow/mlflow:v3.17.1
RUN pip install --no-cache-dir psycopg[binary]