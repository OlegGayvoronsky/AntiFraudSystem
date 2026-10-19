```bash
#!/usr/bin/env bash

set -Eeuo pipefail

echo "=== Инициализация AntiFraudSystem ==="

echo "[1/4] Запускаем PostgreSQL..."
docker compose up -d db

echo "[2/4] Собираем актуальные образы..."
docker compose --profile init build warm-start api

echo "[3/4] Выполняем первоначальный прогрев..."
docker compose --profile init run --rm warm-start

echo "[4/4] Запускаем API..."
docker compose up -d api

echo
echo "=== Инициализация завершена ==="
docker compose ps

echo
echo "Логи API: docker compose logs -f api"
```