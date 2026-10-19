"""
Препроцессинг без состояния: исторические признаки считаются по таблице
transactions, по ВСЕМ транзакциям сущности (customer / merchant) со
step < step текущей транзакции.

Батч обрабатывается по шагам (step по возрастанию):
  1. берём все строки текущего step;
  2. для каждой считаем признаки по step < s (строки того же step друг друга не видят);
  3. сохраняем эти строки в transactions (в рамках той же БД-транзакции);
  4. переходим к следующему step -- он уже видит сохранённые строки.

Использование (commit в БД происходит при выходе из with, после предсказания;
при исключении всё откатывается):

    with builder.session() as fs:
        enriched = fs.build(rows)
        scores = model.predict_proba(frame_from(enriched))[:, 1]

Имена признаков сохранены (*_prev_step, ...), но теперь это агрегаты по всей
предыдущей истории, а не по одному предыдущему шагу -- модель и офлайн-признаки
должны быть обучены с тем же определением.
"""
from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from antifraud import db
from antifraud.config import settings

CUSTOMER, MERCHANT = "customer", "merchant"
NAMESPACE = uuid.UUID("6f1c2a52-4a0e-4c3b-9d57-0b7a3a1f2e11")


@dataclass(frozen=True)
class Stats:
    n: int
    mean: float
    std: float
    last_step: int
    unique: int



def entity_features(kind: str, st: Stats | None, step: int, amount: float) -> dict:
    avg = st.mean if st else 0.0
    f = {
        f"{kind}_transaction_count_prev_step": st.n if st else 0,
        f"{kind}_avg_amount_prev_step": avg,
        f"{kind}_std_amount_prev_step": st.std if st else 0.0,
        f"{kind}_time_since_last_transaction": (step - st.last_step) if st else 0,
        f"amount_to_{kind}_avg_amount": amount / avg if avg > 0 else 0.0,
        f"has_{kind}_history": int(st is not None),
    }
    if kind == MERCHANT:
        f["merchant_unique_customers_prev_step"] = st.unique if st else 0
    return f


class DbSession:
    def __init__(self, conn) -> None:
        self.conn = conn

    def entity_stats(self, kind: str, ids: set, step: int) -> dict:
        raw = db.fetch_entity_stats(self.conn, kind, ids, step)
        return {k: Stats(*v) for k, v in raw.items()}

    def insert(self, rows: list[dict]) -> None:
        db.insert_transactions(self.conn, rows)


class DbStore:
    @contextmanager
    def session(self, commit: bool = True) -> Iterator[DbSession]:
        with db.transactions_session() as conn:
            yield DbSession(conn)
            if not commit:
                conn.rollback()


class MemorySession:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self._ids = {r["transaction_id"] for r in rows}

    def entity_stats(self, kind: str, ids: set, step: int) -> dict:
        col = kind
        buckets: dict = defaultdict(list)
        for row in self.rows:
            if row["step"] < step and row[col] in ids:
                buckets[row[col]].append(row)
        out = {}
        for key, rows in buckets.items():
            amounts = np.array([row["amount"] for row in rows], dtype=float)
            out[key] = Stats(
                n=len(rows),
                mean=float(amounts.mean()),
                std=float(amounts.std(ddof=1)) if len(rows) > 1 else 0.0,
                last_step=max(row["step"] for row in rows),
                unique=len({row["customer"] for row in rows}),
            )
        return out

    def insert(self, rows: list[dict]) -> None:
        for row in rows:
            if row["transaction_id"] not in self._ids:
                self._ids.add(row["transaction_id"])
                self.rows.append(row)


class InMemoryStore:
    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._lock = threading.Lock()

    @contextmanager
    def session(self, commit: bool = True) -> Iterator[MemorySession]:
        with self._lock:
            sess = MemorySession(list(self._rows))
            yield sess
            if commit:
                self._rows = sess.rows


class TransactionStore(Protocol):
    def session(self, commit: bool = True): ...


def make_store() -> TransactionStore:
    return DbStore() if settings.database_url else InMemoryStore()


class FeatureSession:
    def __init__(self, store_session) -> None:
        self.store_session = store_session

    def build(self, rows: list[dict]) -> list[dict]:
        """
        rows: словари с transaction_id, step, customer, merchant, age, gender, category, amount
        Возвращает словари в исходном порядке: исходные поля + transaction_id
        + исторические признаки. Строки сохраняются в transactions шаг за шагом.
        """
        out: list[dict | None] = [None] * len(rows)
        by_step: dict[int, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            by_step[int(row["step"])].append(i)

        for step in sorted(by_step):
            idxs = by_step[step]
            cids = {rows[i]["customer"] for i in idxs}
            mids = {rows[i]["merchant"] for i in idxs}

            c_stats = self.store_session.entity_stats(CUSTOMER, cids, step)
            m_stats = self.store_session.entity_stats(MERCHANT, mids, step)

            to_insert = []
            for i in idxs:
                row = rows[i]
                amount = float(row["amount"])
                tid = str(row.get("transaction_id") or uuid.uuid4())
                cid = row["customer"]
                mid = row["merchant"]

                out[i] = {
                    **row,
                    "transaction_id": tid,
                    **entity_features(CUSTOMER, c_stats.get(cid), step, amount),
                    **entity_features(MERCHANT, m_stats.get(mid), step, amount),
                }
                to_insert.append({
                    "transaction_id": tid, "step": step,
                    "customer": cid, "merchant": mid,
                    "age": str(row["age"]), "gender": str(row["gender"]),
                    "category": str(row["category"]), "amount": amount,
                })

            self.store_session.insert(to_insert)
        return out


class FeatureBuilder:
    def __init__(self, store: TransactionStore | None = None) -> None:
        self.store = store or make_store()

    @contextmanager
    def session(self, commit: bool = True) -> Iterator[FeatureSession]:
        """commit=False: признаки считаются (в т.ч. с учётом более ранних step
        внутри батча), но в БД ничего не остаётся."""
        with self.store.session(commit=commit) as store:
            yield FeatureSession(store)

    def build(self, rows: list[dict], commit: bool = True) -> list[dict]:
        with self.session(commit=commit) as feature_session:
            return feature_session.build(rows)


def warm_start(builder: FeatureBuilder, history: pd.DataFrame, chunk: int = 20_000) -> None:
    """
    Загружает историю в transactions (без расчёта признаков). Идемпотентно:
    transaction_id берётся из колонки transaction_id, а если её нет --
    детерминированно строится из содержимого строки, поэтому повторный
    запуск на том же файле не создаёт дублей (ON CONFLICT DO NOTHING).
    """
    h = history.sort_values("step", kind="stable").reset_index(drop=True)
    for start in range(0, len(h), chunk):
        part = h.iloc[start:start + chunk]
        rows = []
        for pos, row in zip(
            range(start, start + len(part)),
            part.itertuples(index=False),
            strict=True
        ):
            tid = (
                row.transaction_id if "transaction_id" in part.columns
                else str(uuid.uuid5(
                    NAMESPACE,
                    f"{pos}|{row.step}|{row.customer}|{row.age}|"
                    f"{row.gender}|{row.merchant}|{row.category}|{row.amount}"
                ))
            )

            rows.append({
                "transaction_id": tid, "step": int(row.step),
                "customer": row.customer,
                "merchant": row.merchant,
                "age": row.age, "gender": str(row.gender),
                "category": str(row.category), "amount": float(row.amount),
            })
        with builder.store.session() as store:
            store.insert(rows)
