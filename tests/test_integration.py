import os
import uuid

import numpy as np
import pandas as pd
import psycopg
import pytest
from offline_preprocessing import make_features

from antifraud import db
from antifraud.service.preprocessing import DbStateStore, FeatureBuilder

DATABASE_URL = os.getenv("DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DATABASE_URL,
        reason="нужен Postgres: задайте DATABASE_URL"
    ),
]


def test_prediction_is_logged(client, good_row):
    body = client.post("/v1/predict", json=good_row).json()

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT model_version, score, features->>'category', response_code "
            "FROM predictions WHERE request_id = %s",
            (body["request_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == body["model_version"]
    assert row[1] == pytest.approx(body["score"])
    assert row[2] == good_row["category"]
    assert row[3] == 200


def test_bad_age_is_logged(client, good_row):
    bad_row = {**good_row, "age": "10"}
    response = client.post("/v1/predict", json=bad_row)

    assert response.status_code == 422
    request_id = response.headers["X-Request-ID"]

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT model_version, score, response_code "
            "FROM predictions WHERE request_id = %s",
            (request_id,),
        ).fetchone()

    assert row is not None
    assert row[2] == 422


def test_missing_field_is_logged(client, good_row):
    bad_row = dict(good_row)
    del bad_row["age"]
    response = client.post("/v1/predict", json=bad_row)
    
    assert response.status_code == 422
    request_id = response.headers["X-Request-ID"]

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT model_version, score, response_code "
            "FROM predictions WHERE request_id = %s",
            (request_id,),
        ).fetchone()

    assert row is not None
    assert row[2] == 422


def test_extra_field_is_422(client, good_row):
    bad_row = {**good_row, "extra_field": 1}
    response = client.post("/v1/predict", json=bad_row)

    assert response.status_code == 422
    request_id = response.headers["X-Request-ID"]

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT model_version, score, response_code "
            "FROM predictions WHERE request_id = %s",
            (request_id,),
        ).fetchone()

    assert row is not None
    assert row[2] == 422


def test_batch_prediction_is_logged(client, good_row):
    good_rows = [good_row, {**good_row, "category": "es_transportation"}]
    body = client.post("/v1/predict/batch", json={"rows": good_rows}).json()

    request_ids = [item["request_id"] for item in body]
    assert len(set(request_ids)) == len(request_ids)

    with psycopg.connect(DATABASE_URL) as conn:
        db_rows = conn.execute(
            "SELECT request_id, score, response_code "
            "FROM predictions WHERE request_id = ANY(%s) "
            "ORDER BY request_id",
            (request_ids,),
        ).fetchall()

    assert len(db_rows) == len(good_rows)
    assert all(r[2] == 200 for r in db_rows)

    body_sorted = sorted(body, key=lambda r: r["request_id"])

    for db_row, resp_row in zip(
        db_rows,
        body_sorted,
        strict=True
    ):
        assert str(db_row[0]) == resp_row["request_id"]
        assert db_row[1] == pytest.approx(resp_row["score"])


def test_trash_row_in_batch_is_logged_once(client, good_row):
    bad_rows = [good_row, {**good_row, "age": 10}]
    response = client.post("/v1/predict/batch", json={"rows": bad_rows})

    assert response.status_code == 422
    request_id = response.headers["X-Request-ID"]

    with psycopg.connect(DATABASE_URL) as conn:
        db_rows = conn.execute(
            "SELECT response_code FROM predictions WHERE request_id = %s",
            (request_id,),
        ).fetchall()

    assert len(db_rows) == 1
    assert db_rows[0][0] == 422


@pytest.fixture
def state_prefix():
    db.init()
    prefix = f"parity_{uuid.uuid4().hex[:8]}_"
    yield prefix
    db.delete_entity_states(prefix)
 
 
def test_online_features_match_offline(raw_transactions, state_prefix):
    raw = raw_transactions.copy()
    raw["customer"] = state_prefix + raw["customer"].astype(str)
    raw["merchant"] = state_prefix + raw["merchant"].astype(str)
    raw["_row_id"] = np.arange(len(raw))
 
    offline, _ = make_features(raw.copy())
    offline = offline.set_index("_row_id").sort_index()
 

    builder = FeatureBuilder(DbStateStore())
    ordered = raw.sort_values("step", kind="stable")
    online_rows = []

    for start in range(0, len(ordered), 500):
        online_rows += builder.build(
            ordered.iloc[start:start + 500].to_dict("records")
        )
    
    online = pd.DataFrame(online_rows).set_index("_row_id").sort_index()
 
    feature_cols = [c for c in online.columns if c not in raw.columns]
    assert feature_cols, "онлайн-модуль не вернул ни одного признака"
 
    missing = [c for c in feature_cols if c not in offline.columns]
    assert not missing, f"признаков нет в офлайн-версии: {missing}"
 
    for col in feature_cols:
        np.testing.assert_allclose(
            online[col].to_numpy(float),
            offline[col].to_numpy(float),
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"онлайн и офлайн признак расходятся: {col}",
        )
 
 
def test_state_is_persisted_between_requests(state_prefix):
    row = {
        "step": 1,
        "age": 4,
        "gender": "M",
        "customer": f"{state_prefix}c1",
        "merchant": f"{state_prefix}m1",
        "amount": 100.0,
    }
    builder = FeatureBuilder(DbStateStore())
 
    first = builder.build([row])[0]
    assert first["has_customer_history"] == 0
    assert first["has_merchant_history"] == 0
 
    second = FeatureBuilder(DbStateStore()).build(
        [{**row, "step": 3, "amount": 50.0}]
    )[0]
    assert second["has_customer_history"] == 1
    assert second["customer_transaction_count_prev_step"] == 1
    assert second["customer_avg_amount_prev_step"] == pytest.approx(100.0)
    assert second["customer_time_since_last_transaction"] == 2
    assert second["amount_to_customer_avg_amount"] == pytest.approx(0.5)
    assert second["merchant_unique_customers_prev_step"] == 1
 
 
def test_late_transaction_is_rejected_and_state_not_changed(state_prefix):
    from antifraud.service.preprocessing import LateTransactionError
 
    row = {
        "step": 5,
        "age": 4,
        "gender": "M",
        "customer": f"{state_prefix}c1",
        "merchant": f"{state_prefix}m1",
        "amount": 10.0,
    }
    builder = FeatureBuilder(DbStateStore())
    builder.build([row])
 
    with pytest.raises(LateTransactionError):
        builder.build([{**row, "step": 2}])
 
    nxt = builder.build([{**row, "step": 6}])[0]
    assert nxt["customer_transaction_count_prev_step"] == 1


def test_late_transaction_is_409_and_logged(client, good_row, state_prefix):
    row_step_5 = {
        **good_row,
        "customer": f"{state_prefix}c1",
        "merchant": f"{state_prefix}m1",
        "step": 5,
    }
    first = client.post("/v1/predict", json=row_step_5)
    assert first.status_code == 200

    row_step_2 = {**row_step_5, "step": 2}
    response = client.post("/v1/predict", json=row_step_2)

    assert response.status_code == 409
    request_id = response.headers["X-Request-ID"]

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT response_code FROM predictions WHERE request_id = %s",
            (request_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == 409