## Quantum Docker Deployment

1. **Prerequisites**:
   - NVIDIA GPU with CUDA 12.2+
   - Docker Compose v2.20+

2. **Start Services**:
   ```bash
   docker compose -f docker-compose-quantum.yml up --build
   ```

3. **Verify Quantum Acceleration**:
   ```bash
   docker exec quantum-trader qiskit-gpu-test
   ```

4. **Troubleshooting**:
   - GPU Not Detected: Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   - OQS Errors: Run `docker system prune -af` to clear build cache
