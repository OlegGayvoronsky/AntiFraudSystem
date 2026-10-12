from contextlib import contextmanager

import psycopg
from psycopg.types.json import Json

from antifraud.config import settings

DDL = """
    CREATE TABLE IF NOT EXISTS predictions (
        request_id uuid PRIMARY KEY,
        ts timestamptz NOT NULL DEFAULT now(),
        model_version text NOT NULL,
        features jsonb NOT NULL,
        score double precision,
        decision text,
        latency_ms double precision,
        response_code int NOT NULL
    )
"""

STATE_DDL = """
    CREATE TABLE IF NOT EXISTS entity_state (
        kind text NOT NULL,
        key text NOT NULL,
        state jsonb NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (kind, key)
    )
"""


def init() -> None:
    if settings.database_url is None:
        return
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(7001)")
        conn.execute(DDL)
        conn.execute(STATE_DDL)


def save_prediction(
    request_id: str,
    model_version: str,
    features: dict,
    score: float | None,
    decision: str | None,
    latency_ms: float | None,
    response_code: int
) -> None:
    if settings.database_url is None:
        return
    with psycopg.connect(settings.database_url) as conn:
        insert_request = """
            INSERT INTO predictions (
                request_id,
                model_version,
                features,
                score,
                decision,
                latency_ms,
                response_code
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        conn.execute(
            insert_request,
            (
                request_id,
                model_version,
                Json(features),
                score,
                decision,
                latency_ms,
                response_code
            )
        )


_LOCK_STATE_SQL = (
    "SELECT pg_advisory_xact_lock(hashtextextended(k, 0)) "
    "FROM (SELECT k FROM unnest(%s::text[]) AS k ORDER BY k) s"
)
_SELECT_STATE_SQL = (
    "SELECT kind, key, state FROM entity_state "
    "WHERE (kind, key) IN (SELECT * FROM unnest(%s::text[], %s::text[]))"
)
_UPSERT_STATE_SQL = (
    "INSERT INTO entity_state (kind, key, state) VALUES (%s, %s, %s) "
    "ON CONFLICT (kind, key) DO UPDATE "
    "SET state = EXCLUDED.state, updated_at = now()"
)


@contextmanager
def state_transaction(keys, commit: bool = True):
    """
    keys: {(kind, key), ...}. Одна транзакция: блокирует ключи (в отсортированном
    порядке -- без дедлоков), читает состояния и отдаёт dict {(kind, key): dict}
    (отсутствующие ключи в нём не будут). Вызывающий код может изменить этот dict;
    при commit=True и штатном выходе всё, что в нём лежит, пишется upsert-ом.
    При исключении транзакция откатывается.
    """
    if settings.database_url is None:
        raise RuntimeError("DATABASE_URL не задан: состояние в БД недоступно")

    keys = sorted(keys)
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(_LOCK_STATE_SQL, ([f"{k}:{v}" for k, v in keys],))
        rows = conn.execute(
            _SELECT_STATE_SQL,
            ([k for k, _ in keys], [v for _, v in keys]),
        ).fetchall()
        states = {(kind, key): state for kind, key, state in rows}

        yield states

        if commit and states:
            with conn.cursor() as cur:
                cur.executemany(
                    _UPSERT_STATE_SQL,
                    [(k, v, Json(s)) for (k, v), s in states.items()],
                )


def delete_entity_states(key_prefix: str) -> None:
    """Для тестов: удалить состояния, ключи которых начинаются с prefix."""
    if settings.database_url is None:
        return
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(
            "DELETE FROM entity_state WHERE starts_with(key, %s)",
            (key_prefix,),
        )
