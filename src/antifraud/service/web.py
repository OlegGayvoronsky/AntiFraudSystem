"""Web-интерфейс Anti-Fraud: рабочее окно аналитика.
Ожидается, что в app.state.features лежит FeatureBuilder.
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import psycopg
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from psycopg.rows import dict_row
from pydantic import BaseModel

from antifraud.db import db as _db

log = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

API_PREFIX = "/scoring/api"

TX_FIELDS = (
                "transaction_id", "step", "customer",
                "merchant", "age", "gender", "category", "amount"
            )

SHAP_TABLE = "shap_values"
SHAP_COLUMN = "shap"

MAX_IDS = 1000


def _clean(o: Any) -> Any:
    """NaN/inf -> None, numpy-скаляры -> python (иначе JSON-ответ упадёт)."""
    if hasattr(o, "item") and callable(o.item) and not isinstance(o, (dict, list, tuple)):
        try:
            o = o.item()
        except Exception:
            pass
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


@router.get("/scoring", response_class=HTMLResponse)
def scoring_page(request: Request):
    return templates.TemplateResponse(request, "scoring.html", {"api": API_PREFIX})


_SELECT = """
    SELECT t.transaction_id, t.step, t.customer, t.merchant, t.age, t.gender,
           t.category, t.amount, t.received_at,
           p.score, p.decision, p.in_review_queue, p.model_version, p.scored_at,
           p.features, l.fraud AS manual_fraud, l.label_received_at
"""
_FROM = """
    FROM predictions p
    JOIN transactions t ON t.transaction_id = p.transaction_id
    LEFT JOIN fraud_labels l ON l.transaction_id = p.transaction_id
"""

_MODE_WHERE = {
    "review": "p.score IS NOT NULL AND upper(p.decision) = 'REVIEW' AND p.in_review_queue IS TRUE",
    "decided": "p.score IS NOT NULL AND upper(p.decision) IN ('ALLOW', 'BLOCK')",
}


def _parse_ids(raw: str) -> list[UUID]:
    parts = [x for x in re.split(r"[\s,;]+", raw.strip()) if x]
    if len(parts) > MAX_IDS:
        raise HTTPException(422, f"Слишком много id (максимум {MAX_IDS})")
    try:
        return [UUID(x) for x in parts]
    except ValueError as e:
        raise HTTPException(422, f"Некорректный transaction_id: {e}") from e


@router.get(f"{API_PREFIX}/transactions")
def list_transactions(
    mode: Literal["review", "decided"] = "review",
    ids: str | None = Query(None, description="transaction_id через запятую/пробел/перенос строки"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    where = [_MODE_WHERE[mode]]
    params: dict[str, Any] = {"limit": limit, "offset": offset}

    if ids and ids.strip():
        params["ids"] = _parse_ids(ids)
        where.append("t.transaction_id = ANY(%(ids)s)")

    where_sql = " WHERE " + " AND ".join(where)
    order_sql = " ORDER BY p.score DESC, t.transaction_id"

    with _db(dict_row) as conn:
        total = conn.execute("SELECT count(*) AS n" + _FROM + where_sql, params).fetchone()["n"]
        rows = conn.execute(
            _SELECT + _FROM + where_sql + order_sql + " LIMIT %(limit)s OFFSET %(offset)s", params
        ).fetchall()

    for row in rows:
        feats = row.get("features") or {}
        row["features"] = {k: v for k, v in feats.items() if k not in TX_FIELDS}
        row["features_error"] = None

    return _clean({"total": total, "offset": offset, "limit": limit, "items": rows})


@router.get(API_PREFIX + "/transactions/{transaction_id}/shap")
def get_shap(transaction_id: UUID):
    try:
        with _db(dict_row) as conn:
            row = conn.execute(
                f"SELECT {SHAP_COLUMN} AS shap FROM {SHAP_TABLE} WHERE transaction_id = %s",
                (transaction_id,),
            ).fetchone()
    except psycopg.errors.UndefinedTable:
        raise HTTPException(501, f"Таблица {SHAP_TABLE} пока не создана") from None

    if row is None or not row["shap"]:
        raise HTTPException(404, "SHAP для этой транзакции не найден")

    items = [{"feature": k, "value": v} for k, v in row["shap"].items()]
    items.sort(key=lambda x: abs(x["value"] or 0), reverse=True)
    return _clean({"transaction_id": transaction_id, "items": items})


class DecisionIn(BaseModel):
    decision: Literal["ALLOW", "BLOCK"]


@router.post(API_PREFIX + "/transactions/{transaction_id}/decision")
def set_decision(transaction_id: UUID, body: DecisionIn):
    fraud = body.decision == "BLOCK"
    with _db(dict_row) as conn:
        exists = conn.execute(
            "SELECT 1 FROM transactions WHERE transaction_id = %s", (transaction_id,)
        ).fetchone()
        if exists is None:
            raise HTTPException(404, "Транзакция не найдена")

        cur = conn.execute(
            "UPDATE predictions SET in_review_queue = false "
            "WHERE transaction_id = %s AND in_review_queue IS TRUE",
            (transaction_id,),
        )
        left_queue = cur.rowcount > 0

        label = conn.execute(
            """
            INSERT INTO fraud_labels (transaction_id, fraud, fraud_type)
            VALUES (%s, %s, %s)
            ON CONFLICT (transaction_id)
            DO UPDATE SET fraud = EXCLUDED.fraud, label_received_at = now()
            RETURNING fraud, label_received_at
            """,
            (transaction_id, fraud, "ANALYST"),
        ).fetchone()

    return _clean(
        {
            "transaction_id": transaction_id,
            "decision": body.decision,
            "manual_fraud": label["fraud"],
            "label_received_at": label["label_received_at"],
            "left_review_queue": left_queue,
        }
    )