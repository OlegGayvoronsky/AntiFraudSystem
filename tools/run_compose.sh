```bash
#!/usr/bin/env bash
set -Eeuo pipefail

read -rp "PostgreSQL user [postgres]: " POSTGRES_USER
POSTGRES_USER="${POSTGRES_USER:-postgres}"

read -rsp "PostgreSQL password: " POSTGRES_PASSWORD
echo

read -rp "PostgreSQL database [antifraud]: " POSTGRES_DB
POSTGRES_DB="${POSTGRES_DB:-antifraud}"

read -rp "MLflow PostgreSQL user [mlflow]: " MLFLOW_DB_USER
MLFLOW_DB_USER="${MLFLOW_DB_USER:-mlflow}"

read -rsp "MLflow PostgreSQL password: " MLFLOW_DB_PASSWORD
echo

read -rp "MLflow database [mlflow]: " MLFLOW_DB_NAME
MLFLOW_DB_NAME="${MLFLOW_DB_NAME:-mlflow}"

if [[ -z "$POSTGRES_PASSWORD" || -z "$MLFLOW_DB_PASSWORD" ]]; then
  echo "Error: passwords must not be empty." >&2
  exit 1
fi

export POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB
export MLFLOW_DB_USER MLFLOW_DB_PASSWORD MLFLOW_DB_NAME

export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}"

export MLFLOW_BACKEND_STORE_URI="postgresql+psycopg://${MLFLOW_DB_USER}:${MLFLOW_DB_PASSWORD}@mlflow-db:5432/${MLFLOW_DB_NAME}"

docker compose "$@"
```