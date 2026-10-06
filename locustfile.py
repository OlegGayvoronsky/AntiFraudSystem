from locust import HttpUser, between, task


class ChurnUser(HttpUser):
    wait_time = between(0.5, 1.5)

    good_row = {
      "step": 143,
      "customer": "C1216867117",
      "age": "4",
      "gender": "F",
      "merchant": "M1823072687",
      "category": "es_transportation",
      "amount": 49.77,
      "fraud": 0
    }

    @task(5)
    def predict(self):
        self.client.post(
            "/v1/predict",
            json=self.good_row,
        )

    @task(1)
    def health(self):
        self.client.get("/health")