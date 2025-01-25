import time
import numpy as np # type: ignore
from src.quantum.quantum_engine import QuantumTrader

def generate_order_books(num_books):
    """Generate a list of mock order books for testing."""
    return [{'bids': [np.random.rand()], 'asks': [np.random.rand()]} for _ in range(num_books)]

def test_stress_quantum_trader():
    """Stress test for the QuantumTrader's find_alpha method."""
    trader = QuantumTrader()
    num_books = 10000  # Simulate 10,000 order books
    order_books = generate_order_books(num_books)

    # Measure execution time
    start_time = time.time()
    trader.order_books_gpu = order_books  # Load order books into GPU
    alpha = trader.find_alpha()  # Execute the find_alpha method
    end_time = time.time()

    # Log performance metrics
    execution_time = end_time - start_time
    print(f"Execution Time for {num_books} order books: {execution_time:.4f} seconds")
    print(f"Alpha: {alpha}")

if __name__ == "__main__":
    test_stress_quantum_trader()
