from locust import HttpUser, task, between

class TradingBotUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def place_order(self):
        self.client.post("/api/place_order", json={
            "symbol": "sBTCUSD",
            "side": "Buy",
            "orderQty": 0.01,
            "ordType": "Market"
        })

    @task
    def get_balance(self):
        self.client.get("/api/get_balance")
