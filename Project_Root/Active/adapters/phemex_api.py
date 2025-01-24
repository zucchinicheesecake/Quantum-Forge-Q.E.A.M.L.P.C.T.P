import requests
import hmac
import hashlib
import time
from typing import Dict, Any

class PhemexAPI:
    def __init__(self, config: Dict[str, str]):
        self.api_key = config['api_key']
        self.secret = config['api_secret']
        self.base_url = config['base_url']
        
    def _generate_signature(self, query_string: str) -> str:
        return hmac.new(
            self.secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    def get_market_data(self) -> Dict[str, Any]:
        endpoint = '/v1/md/ticker/24hr'
        timestamp = str(int(time.time() * 1000))
        query_string = f"timestamp={timestamp}"
        signature = self._generate_signature(query_string)
        
        headers = {
            'x-phemex-access-token': self.api_key,
            'x-phemex-request-signature': signature,
            'x-phemex-request-timestamp': timestamp
        }
        
        response = requests.get(
            f"{self.base_url}{endpoint}",
            headers=headers
        )
        return response.json()
    
    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation for order execution
        pass
