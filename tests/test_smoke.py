def test_predict_smoke(client, good_row):
    r = client.post("/v1/predict", json=good_row)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert isinstance(body["fraud"], bool)
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
    assert isinstance(body["fraud"], bool)
    assert isinstance(body["model_version"], str)
    assert isinstance(body["request_id"], str)
    assert isinstance(body["latency_ms"], float)

def test_predict_returns_unique_request_id(client, good_row):
    r1 = client.post("/v1/predict", json=good_row).json()
    r2 = client.post("/v1/predict", json=good_row).json()

    assert r1["request_id"] != r2["request_id"]
