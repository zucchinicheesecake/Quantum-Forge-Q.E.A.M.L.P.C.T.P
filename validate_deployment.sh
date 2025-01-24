#!/bin/bash

# After deployment, run:
docker exec quantum-trader \
  python -m pytest tests/integration/test_quantum_deployment.py

# Verify GPU acceleration
docker exec quantum-trader nvidia-smi
docker exec quantum-trader \
  python -c "from qiskit import QuantumCircuit; print(QuantumCircuit(2))"

# Test quantum encryption
docker exec quantum-trader \
  python -m security.test_oqs --iterations 1000
