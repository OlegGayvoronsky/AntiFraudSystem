def test_predict_smoke(client, good_row):
    r = client.post("/v1/predict", json=good_row)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert isinstance(body["decision"], str)
    assert body["latency_ms"] >= 0
    assert body["model_version"]


def test_batch_and_single_agree(client, good_row):
    s1 = client.post("/v1/predict", json=good_row).json()["score"]
    s2 = client.post("/v1/predict", json=good_row).json()["score"]
    assert abs(s1 - s2) < 1e-12

def test_predict_response_types(client, good_row):
    r = client.post("/v1/predict", json=good_row)

    assert r.status_code == 200

    body = r.json()
    assert isinstance(body["score"], float)
    assert isinstance(body["decision"], str)
    assert isinstance(body["model_version"], str)
    assert isinstance(body["transaction_id"], str)
    assert isinstance(body["latency_ms"], float)

def test_predict_returns_unique_transaction_id(client, good_row):
    r1 = client.post("/v1/predict", json=good_row).json()
    r2 = client.post("/v1/predict", json=good_row).json()

    assert r1["transaction_id"] != r2["transaction_id"]

def test_predict_batch_smoke(client, good_row):
    good_rows = [good_row, {**good_row, "gender": "M"}]
    body = client.post("/v1/predict/batch", json={"rows": good_rows}).json()
    
    transaction_ids = [item["transaction_id"] for item in body]
    assert len(set(transaction_ids)) == len(transaction_ids)

    for row in body:
        assert 0.0 <= row["score"] <= 1.0
        assert isinstance(row["decision"], str)
        assert row["latency_ms"] >= 0
        assert row["model_version"]


def test_predict_batch_response_types(client, good_row):
    good_rows = [good_row, {**good_row, "gender": "M"}]
    body = client.post("/v1/predict/batch", json={"rows": good_rows})
    
    assert body.status_code == 200

    body = body.json()

    for row in body:
        assert isinstance(row["score"], float)
        assert isinstance(row["decision"], str)
        assert isinstance(row["model_version"], str)
        assert isinstance(row["transaction_id"], str)
        assert isinstance(row["latency_ms"], float)


def test_predict_batch_and_single_agree(client, good_row):
    single = client.post("/v1/predict", json=good_row)
    batch = client.post("/v1/predict/batch", json={"rows": [good_row]})

    assert single.status_code == 200
    assert batch.status_code == 200

    single_score = single.json()["score"]
    batch_score = batch.json()[0]["score"]

    assert abs(single_score - batch_score) < 1e-12
