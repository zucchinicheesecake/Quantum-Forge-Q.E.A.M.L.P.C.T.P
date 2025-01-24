class PhemexTradingCore:
    def __init__(self, testnet: bool = True):  # Add testnet flag
        self.base_url = ("https://testnet-api.phemex.com" 
                        if testnet else "https://api.phemex.com")
        self.symbol_prefix = "s" if testnet else ""  # Testnet symbols: sBTCUSDT

    def get_mUSD_balance(self):
        """Fetch testnet mUSD balance"""
        return self._private_request("GET", "/accounts/accountPositions")['data']['mUSD']

class PhemexAPIError(Exception):
    """Custom exception for Phemex API errors."""
    pass

class PhemexWorker:
    def __init__(self, order=None, testnet=True):
        self.order = order
        self.testnet = testnet
        self.api = PhemexTradingCore(testnet=self.testnet)
        self.signals = self.create_signals()

    def create_signals(self):
        """Create signals for order execution."""
        class Signals:
            order_executed = Signal()  # Placeholder for signal handling
        return Signals()

    def run(self):
        """Execute the order."""
        if self.order:
            # Simulate order execution
            result = {"status": "FILLED", "id": "12345"}  # Simulated response
            self.signals.order_executed.emit(result)
        else:
            raise ValueError("No order data provided.")
