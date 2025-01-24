import numpy as np
from qiskit import QuantumCircuit
from scipy import stats

class MarketStateEncoder:
    def __init__(self, market_data):
        self.data = market_data
        self.lookback_period = 20  # Default lookback for technical indicators
        
    def to_qubits(self):
        normalized_data = self._normalize_market_data()
        technical_indicators = self._calculate_technical_indicators()
        quantum_state = self._encode_quantum_state(normalized_data, technical_indicators)
        return quantum_state
    
    def _normalize_market_data(self):
        return {
            'price': (self.data['price'] - np.mean(self.data['price'])) / np.std(self.data['price']),
            'volume': (self.data['volume'] - np.mean(self.data['volume'])) / np.std(self.data['volume'])
        }
    
    def _calculate_technical_indicators(self):
        price = self.data['price']
        
        # Calculate RSI
        returns = np.diff(price)
        gains = np.maximum(returns, 0)
        losses = -np.minimum(returns, 0)
        avg_gain = np.mean(gains[-self.lookback_period:])
        avg_loss = np.mean(losses[-self.lookback_period:])
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema12 = pd.Series(price).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(price).ewm(span=26).mean().iloc[-1]
        macd = ema12 - ema26
        
        return {
            'rsi': rsi,
            'macd': macd
        }
    
    def _encode_quantum_state(self, normalized_data, technical_indicators):
        circuit = QuantumCircuit(6)  # Increased qubits for additional features
        
        # Price and volume encoding
        circuit.ry(normalized_data['price'][-1] * np.pi, 0)
        circuit.ry(normalized_data['volume'][-1] * np.pi, 1)
        
        # Technical indicator encoding
        circuit.ry((technical_indicators['rsi']/100) * np.pi, 2)
        circuit.ry((technical_indicators['macd']+1)/2 * np.pi, 3)
        
        # Entanglement layer
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        
        return circuit
