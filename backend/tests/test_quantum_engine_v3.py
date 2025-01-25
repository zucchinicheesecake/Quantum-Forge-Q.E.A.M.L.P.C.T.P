import hypothesis.strategies as st
from hypothesis import given
import unittest
from active.core.quantum_risk_engine import QuantumRiskEngine  # Import the QuantumRiskEngine

class TestQuantumTrading(unittest.TestCase):
    @given(
        st.floats(0.01, 0.5),  # volatility
        st.floats(1000, 100000)  # balance
    )
    def test_quantum_position_sizing(self, volatility, balance):
        engine = QuantumRiskEngine()
        position = engine.calculate_position(balance, volatility)
        assert 0 <= position <= balance * 0.5  # Max 50% exposure
        
    def test_market_crash_resilience(self):
        # Assuming MarketSimulator is defined elsewhere
        simulator = MarketSimulator(volatility=1000)  # Corrected to remove the percentage sign
        bot = TradingBot(engine=QuantumRiskEngine())
        result = bot.run(simulator)
        assert result.max_drawdown < 30

if __name__ == "__main__":
    unittest.main()
