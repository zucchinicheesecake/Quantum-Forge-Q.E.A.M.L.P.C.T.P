import requests
import hmac
import hashlib
import time
import logging
from typing import Dict, Any

class PhemexAPI:
    def __init__(self, config: Dict[str, str]):
        self.api_key = config.get('api_key')
        self.secret = config.get('api_secret')
        self.base_url = config.get('base_url', 'https://api.phemex.com')
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
    def is_connected(self) -> bool:
        """Check connection to exchange"""
        try:
            response = self.session.get(f"{self.base_url}/v1/exchangeInfo")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Connection check failed: {str(e)}")
            return False
            
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC signature for authentication"""
        return hmac.new(
            self.secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    def get_market_data(self) -> Dict[str, Any]:
        """Fetch market data from exchange"""
        endpoint = '/v1/md/ticker/24hr'
        timestamp = str(int(time.time() * 1000))
        query_string = f"timestamp={timestamp}"
        signature = self._generate_signature(query_string)
        
        headers = {
            'x-phemex-access-token': self.api_key,
            'x-phemex-request-signature': signature,
            'x-phemex-request-timestamp': timestamp
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {str(e)}")
            return {}
    
    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an order on the exchange"""
        # Implementation for order execution
        pass
