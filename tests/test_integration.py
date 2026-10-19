import os

import psycopg
import pytest

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
            "SELECT model_version, score, decision, features->>'category', response_code "
            "FROM predictions WHERE transaction_id = %s",
            (body["transaction_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == body["model_version"]
    assert row[1] == pytest.approx(body["score"])
    assert row[2] == body["decision"]
    assert row[3] == good_row["category"]
    assert row[4] == 200


def test_bad_age_is_logged(client, good_row):
    bad_row = {**good_row, "age": "10"}
    response = client.post("/v1/predict", json=bad_row)

    assert response.status_code == 422
    transaction_id = response.headers["X-Transaction-ID"]

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT model_version, score, decision, response_code "
            "FROM predictions WHERE transaction_id = %s",
            (transaction_id,),
        ).fetchone()

    assert row is not None
    assert row[1] is None
    assert row[2] is None
    assert row[3] == 422


def test_missing_field_is_logged(client, good_row):
    bad_row = dict(good_row)
    del bad_row["age"]
    response = client.post("/v1/predict", json=bad_row)
    
    assert response.status_code == 422
    transaction_id = response.headers["X-Transaction-ID"]

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT model_version, score, decision, response_code "
            "FROM predictions WHERE transaction_id = %s",
            (transaction_id,),
        ).fetchone()

    assert row is not None
    assert row[1] is None
    assert row[2] is None
    assert row[3] == 422


def test_extra_field_is_422(client, good_row):
    bad_row = {**good_row, "extra_field": 1}
    response = client.post("/v1/predict", json=bad_row)

    assert response.status_code == 422
    transaction_id = response.headers["X-Transaction-ID"]

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT model_version, score, decision, response_code "
            "FROM predictions WHERE transaction_id = %s",
            (transaction_id,),
        ).fetchone()

    assert row is not None
    assert row[1] is None
    assert row[2] is None
    assert row[3] == 422


def test_batch_prediction_is_logged(client, good_row):
    good_rows = [good_row, {**good_row, "category": "es_transportation"}]
    body = client.post("/v1/predict/batch", json={"rows": good_rows}).json()

    transaction_ids = [item["transaction_id"] for item in body]
    assert len(set(transaction_ids)) == len(transaction_ids)

    with psycopg.connect(DATABASE_URL) as conn:
        db_rows = conn.execute(
            "SELECT transaction_id, score, decision, response_code "
            "FROM predictions WHERE transaction_id = ANY(%s) "
            "ORDER BY transaction_id",
            (transaction_ids,),
        ).fetchall()

    assert len(db_rows) == len(good_rows)
    assert all(row[3] == 200 for row in db_rows)

    body_sorted = sorted(body, key=lambda row: row["transaction_id"])

    for db_row, resp_row in zip(
        db_rows,
        body_sorted,
        strict=True
    ):
        assert str(db_row[0]) == resp_row["transaction_id"]
        assert db_row[1] == pytest.approx(resp_row["score"])
        assert db_row[2] == resp_row["decision"]


def test_trash_row_in_batch_is_logged_once(client, good_row):
    bad_rows = [good_row, {**good_row, "age": 10}]
    response = client.post("/v1/predict/batch", json={"rows": bad_rows})

    assert response.status_code == 422
    transaction_id = response.headers["X-Transaction-ID"]

    with psycopg.connect(DATABASE_URL) as conn:
        db_rows = conn.execute(
            "SELECT score, decision, response_code FROM predictions WHERE transaction_id = %s",
            (transaction_id,),
        ).fetchall()

    assert len(db_rows) == 1
    assert db_rows[0][0] is None
    assert db_rows[0][1] is None
    assert db_rows[0][2] == 422
