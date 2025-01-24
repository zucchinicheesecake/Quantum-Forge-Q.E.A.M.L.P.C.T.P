# Quantum Edge Project

## Terminal-Based Health Check System

### 1. Project Integrity Check
```bash
# Check critical file presence
ls -l interface.py quantum_trader.py deployment.py risk_models.py

# Expected OK output:
# -rw-r--r-- 1 user group 15234 Jun 15 interface.py
# -rw-r--r-- 1 user group 29456 Jun 15 quantum_trader.py
# -rwxr-xr-x 1 user group  8192 Jun 15 deployment.py
```

### 2. Dependency Verification
```bash
# Check Python package versions
pip list | grep -E 'tensorflow|torch|pyside2|numba|pandas'

# Expected OK output:
# numpy             1.26.4
# pandas            2.2.1
# PySide2           5.15.2.1
# tensorflow        2.15.0
# torch             2.2.1
# numba             0.59.0
```

### 3. GUI Basic Functionality Test
```bash
# Start GUI in test mode
python interface.py --test 2>&1 | grep -iE 'error|warning|critical'

# Good output: (empty)
# Bad output example:
# CRITICAL | Failed to import quantum_trader: No module named 'quantum_trader'
```

### 4. Trading Engine Core Test
```bash
# Run core strategy verification
python -c "from quantum_trader import QuantumTrader; qt = QuantumTrader(); print(qt.validate_core())"

# Expected OK output:
# CORE_STATUS: STRATEGY_VALID | LATENCY 0.4ms | SHARPE 2.1
```

### 5. Latency Benchmark
```bash
# Test order execution speed (replace with your symbol)
python -c "import time; from quantum_trader import execute_test_order; \
start = time.perf_counter_ns(); execute_test_order('BTC/USDT'); \
print(f'EXECUTION_TIME: {(time.perf_counter_ns()-start)/1e6:.2f}ms')"

# Expected OK output:
# EXECUTION_TIME: 4.72ms
```

### 6. Security Audit
```bash
# Check for credential leaks
grep -r --include=*.py "API_KEY" . 
grep -r --include=*.py "SECRET" .

# Good output: (No matches found)
```

### 7. Network Resilience Test
```bash
# Simulate packet loss (Linux/Mac only)
sudo tc qdisc add dev eth0 root netem loss 30%
python quantum_trader.py --network-test
sudo tc qdisc del dev eth0 root

# Look for these in output:
# [NETWORK] Successfully failed over to dark pool after 3 retries
# [RECOVERY] Restored primary connection in 4.2s
```

### 8. Unit Test Suite
```bash
# Run verification tests (if pytest setup)
pytest tests/ -v

# Sample good output:
# tests/test_core_strategy.py::test_micro_arb PASSED
# tests/test_risk_management.py::test_max_drawdown PASSED
```

### 9. System Resource Check
```bash
# Monitor during stress test (run in separate terminal)
top -b -n 1 | grep -E "python|PID|%CPU"

# Acceptable ranges:
# CPU% < 180% (per core)
# MEM% < 45%
```

### 10. Exchange Connectivity
```bash
# Test Phemex API connectivity
python -c "import ccxt; exchange = ccxt.phemex(); print(exchange.fetch_order_book('BTC/USDT')['bids'][0])"

# Expected OK output:
# [30000.5, 1.542]  # (price, amount)
