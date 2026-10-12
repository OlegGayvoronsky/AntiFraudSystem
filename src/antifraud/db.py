from contextlib import contextmanager

import psycopg
from psycopg.types.json import Json

from antifraud.config import settings

TRANSACTIONS_DDL = """
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id uuid PRIMARY KEY,
        step int NOT NULL,
        customer text NOT NULL,
        merchant text NOT NULL,
        age text NOT NULL,
        gender text NOT NULL,
        category text NOT NULL,
        amount double precision NOT NULL,
        received_at timestamptz NOT NULL DEFAULT now()
    )
"""

TRANSACTIONS_INDEXES_DDL = """
    CREATE INDEX IF NOT EXISTS idx_transactions_customer_step
    ON transactions(customer_id, step);

    CREATE INDEX IF NOT EXISTS idx_transactions_merchant_step
    ON transactions(merchant_id, step);
"""

FRAUD_LABELS_DDL = """
    CREATE TABLE IF NOT EXISTS fraud_labels (
        transaction_id uuid PRIMARY KEY,
        fraud boolean NOT NULL,
        label_received_at timestamptz NOT NULL DEFAULT now(),

        FOREIGN KEY (transaction_id)
            REFERENCES transactions(transaction_id) ON DELETE CASCADE
    )
"""

PREDICTIONS_DDL = """
    CREATE TABLE IF NOT EXISTS predictions (
        transaction_id uuid PRIMARY KEY,
        scored_at timestamptz NOT NULL DEFAULT now(),
        model_version text NOT NULL,
        features jsonb NOT NULL,
        score double precision,
        decision text,
        in_review_queue boolean,
        latency_ms double precision,
        response_code int NOT NULL

        FOREIGN KEY (transaction_id)
            REFERENCES transactions(transaction_id) ON DELETE CASCADE
    )
"""

def init() -> None:
    if settings.database_url is None:
        return
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(7001)")
        conn.execute(TRANSACTIONS_DDL)
        conn.execute(FRAUD_LABELS_DDL)
        conn.execute(PREDICTIONS_DDL)


def save_fraud_label(
    transaction_id: str,
    fraud: bool,
) -> None:
    if settings.database_url is None:
        return

    with psycopg.connect(settings.database_url) as conn:
        insert_fraud_label = """
            INSERT INTO fraud_labels (
                transaction_id,
                fraud
            )
            VALUES (%s, %s)
        """
        conn.execute(
            insert_fraud_label,
            (
                transaction_id,
                fraud,
            ),
        )

def save_prediction(
    transaction_id: str,
    model_version: str,
    features: dict,
    score: float | None,
    decision: str | None,
    in_review_queue: bool | None,
    latency_ms: float | None,
    response_code: int
) -> None:
    if settings.database_url is None:
        return
    with psycopg.connect(settings.database_url) as conn:
        insert_request = """
            INSERT INTO predictions (
                transaction_id,
                model_version,
                features,
                score,
                decision,
                in_review_queue,
                latency_ms,
                response_code
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        conn.execute(
            insert_request,
            (
                transaction_id,
                model_version,
                Json(features),
                score,
                decision,
                in_review_queue,
                latency_ms,
                response_code
            )
        )


@contextmanager
def transactions_session():
    """Одна транзакция для чтения истории и записи новых транзакций."""
    if settings.database_url is None:
        raise RuntimeError("DATABASE_URL не задан")

    with psycopg.connect(settings.database_url) as conn:
        yield conn


def fetch_entity_stats(conn, kind: str, ids: set, step: int) -> dict:
    """
    Возвращает статистику по транзакциям с step меньше заданного.

    Результат:
        {
            entity_id: (count, mean, std, last_step, unique_customers)
        }
    """
    if not ids:
        return {}

    query = f"""
        SELECT
            {kind},
            COUNT(*) AS transaction_count,
            AVG(amount) AS mean_amount,
            COALESCE(STDDEV_SAMP(amount), 0) AS std_amount,
            MAX(step) AS last_step,
            COUNT(DISTINCT customer_id) AS unique_customers
        FROM transactions
        WHERE {kind} = ANY(%s)
          AND step < %s
        GROUP BY {kind}
    """

    rows = conn.execute(query, (list(ids), step)).fetchall()

    return {
        entity_id: (
            int(count),
            float(mean_amount),
            float(std_amount),
            int(last_step),
            int(unique_customers),
        )
        for (
            entity_id,
            count,
            mean_amount,
            std_amount,
            last_step,
            unique_customers,
        ) in rows
    }


def insert_transactions(conn, rows: list[dict]) -> None:
    """Пакетная вставка транзакций без дубликатов."""
    if not rows:
        return

    query = """
        INSERT INTO transactions (
            transaction_id,
            step,
            customer,
            merchant,
            age,
            gender,
            category,
            amount
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (transaction_id) DO NOTHING
    """

    conn.cursor().executemany(
        query,
        [
            (
                row["transaction_id"],
                row["step"],
                row["customer"],
                row["merchant"],
                row["age"],
                row["gender"],
                row["category"],
                row["amount"],
            )
            for row in rows
        ],
    )