"""
Онлайн-препроцессинг: воспроизводит обучающие исторические признаки
(make_customer_agg_features / make_merchant_agg_features) по инкрементальному
состоянию на сущность (customer / merchant). Всё общение с БД -- через db.py.

Семантика (как в обучении):
  * *_prev_step -- агрегаты по ПРЕДЫДУЩЕМУ шагу, в котором у сущности были
    транзакции (не step-1). Транзакции того же шага не учитываются.
  * холодный старт -- нули и has_*_history = 0.
"""
from __future__ import annotations

import copy
import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from antifraud import db
from antifraud.config import settings

CUSTOMER, MERCHANT = "customer", "merchant"


class LateTransactionError(ValueError):
    """step транзакции меньше текущего шага сущности: состояние не откатить."""


@dataclass
class EntityState:
    cur_step: int | None = None
    cur_n: int = 0
    cur_mean: float = 0.0
    cur_m2: float = 0.0
    cur_partners: set = field(default_factory=set)
    prev: dict | None = None

    def to_dict(self) -> dict:
        return {
            "cur_step": self.cur_step,
            "cur_n": self.cur_n,
            "cur_mean": self.cur_mean,
            "cur_m2": self.cur_m2,
            "cur_partners": sorted(self.cur_partners),
            "prev": self.prev,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> EntityState:
        if not data:
            return cls()

        data = dict(data)
        data["cur_partners"] = set(data.get("cur_partners", []))
        
        return cls(**data)


def _close_step(st: EntityState) -> None:
    n = st.cur_n
    st.prev = {
        "step": st.cur_step,
        "n": n,
        "mean": st.cur_mean,
        "std": math.sqrt(st.cur_m2 / (n - 1)) if n > 1 else 0.0,
        "unique": len(st.cur_partners),
    }

    st.cur_n, st.cur_mean, st.cur_m2 = 0, 0.0, 0.0
    st.cur_partners = set()


def _advance(st: EntityState, step: int) -> None:
    if st.cur_step is None:
        st.cur_step = step

    elif step < st.cur_step:
        raise LateTransactionError(
            f"step={step} < current step {st.cur_step} of the entity"
        )

    elif step > st.cur_step:
        _close_step(st)
        st.cur_step = step


def _update(
    st: EntityState,
    amount: float,
    partner: str | None
) -> None:
    st.cur_n += 1
    
    delta = amount - st.cur_mean
    
    st.cur_mean += delta / st.cur_n
    st.cur_m2 += delta * (amount - st.cur_mean)
    
    if partner is not None:
        st.cur_partners.add(partner)


def _entity_features(
    kind: str,
    st: EntityState,
    step: int,
    amount: float
) -> dict:
    p = st.prev

    count = p["n"] if p else 0
    avg = p["mean"] if p else 0.0
    std = p["std"] if p else 0.0
    time_since = (step - p["step"]) if p else 0
    am_to_avg = amount / avg if avg > 0 else 0.0
    
    f = {
        f"{kind}_transaction_count_prev_step": count,
        f"{kind}_avg_amount_prev_step": avg,
        f"{kind}_std_amount_prev_step": std,
        f"{kind}_time_since_last_transaction": time_since,
        f"amount_to_{kind}_avg_amount": am_to_avg,
        f"has_{kind}_history": int(p is not None),
    }
    
    if kind == MERCHANT:
        unique_cust = p["unique"] if p else 0
        f["merchant_unique_customers_prev_step"] = unique_cust
    
    return f


class StateStore(Protocol):
    def transaction(self, keys: set[tuple[str, str]], commit: bool = True): ...


class DbStateStore:
    """Состояние в Postgres через db.state_transaction (блокировки и upsert там)."""

    @contextmanager
    def transaction(self, keys, commit: bool = True) -> Iterator[dict]:
        with db.state_transaction(keys, commit=commit) as raw:
            states = {k: EntityState.from_dict(raw.get(k)) for k in keys}
            
            yield states
            
            if commit:
                raw.clear()
                raw.update({k: st.to_dict() for k, st in states.items()})


class InMemoryStateStore:
    """Fallback без БД (DATABASE_URL не задан) и для юнит-тестов."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], EntityState] = {}
        self._lock = threading.Lock()

    @contextmanager
    def transaction(self, keys, commit: bool = True) -> Iterator[dict]:
        with self._lock:
            states = {
                k: copy.deepcopy(self._data[k]) if k in self._data else EntityState()
                for k in keys
            }

            yield states
            
            if commit:
                self._data.update(states)


def make_store() -> StateStore:
    return DbStateStore() if settings.database_url else InMemoryStateStore()



class FeatureBuilder:
    def __init__(self, store: StateStore | None = None) -> None:
        self.store = store or make_store()

    def build(self, rows: list[dict], commit: bool = True) -> list[dict]:
        """
        rows: dict-и с step, customer, merchant, amount (+ любые другие поля).
        Возвращает dict-и в ИСХОДНОМ порядке: исходные поля + исторические признаки.
        commit=False -- посчитать признаки, не меняя состояние.
        """
        return self._process(rows, want_features=True, commit=commit)

    def ingest(self, rows: list[dict]) -> None:
        """Только обновить состояние (прогрев историей)."""
        self._process(rows, want_features=False, commit=True)

    def _process(self, rows, want_features: bool, commit: bool):
        if not rows:
            return []

        keys = {(CUSTOMER, str(r["customer"])) for r in rows}
        keys |= {(MERCHANT, str(r["merchant"])) for r in rows}

        out: list[dict | None] = [None] * len(rows)
        order = sorted(range(len(rows)), key=lambda i: rows[i]["step"])

        with self.store.transaction(keys, commit=commit) as states:
            for i in order:
                r = rows[i]
                step, amount = int(r["step"]), float(r["amount"])

                c, m = str(r["customer"]), str(r["merchant"])
                cs, ms = states[(CUSTOMER, c)], states[(MERCHANT, m)]

                _advance(cs, step)
                _advance(ms, step)

                if want_features:
                    out[i] = {
                        **r,
                        **_entity_features(CUSTOMER, cs, step, amount),
                        **_entity_features(MERCHANT, ms, step, amount),
                    }

                _update(cs, amount, None)
                _update(ms, amount, c)
        return out


def warm_start(
    builder: FeatureBuilder,
    history: pd.DataFrame,
    chunk: int = 20_000
) -> None:
    """Прогрев состояния историей (step, customer, merchant, amount) по возрастанию step."""
    h = history.sort_values("step", kind="stable")
    cols = ["step", "customer", "merchant", "amount"]
    for start in range(0, len(h), chunk):
        builder.ingest(h.iloc[start:start + chunk][cols].to_dict("records"))