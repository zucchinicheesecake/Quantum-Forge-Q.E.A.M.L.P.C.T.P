import unittest
from src.core.trading_engine import TradingEngine

class QuantumMarketCrashTest(unittest.TestCase):
    def test_black_swan_response(self):
        simulator = MarketSimulator(volatility=500)
        bot = TradingEngine()
        result = bot.run(simulator)
        self.assertLess(result.max_drawdown, 25)
