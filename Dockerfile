# Dockerfile (Multi-Stage Build)
# ----------------------------------
# Stage 1: Quantum Base Image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as quantum-base
RUN apt-get update --no-cache && \
    apt-get install -y python3.11 pipenv ocl-icd-opencl-dev \
    libfftw3-dev libopenslide-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Open Quantum Safe Libraries
RUN git clone --branch stable https://github.com/open-quantum-safe/liboqs && \
    cd liboqs && mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && make install

# ----------------------------------
# Stage 2: Frontend Builder
FROM node:18-slim as frontend-builder
WORKDIR /app/chart
COPY chart_memory_manager/package.json .
COPY chart_memory_manager/package-lock.json .
RUN npm ci
COPY chart_memory_manager/ .
RUN npm run build

# ----------------------------------
# Stage 3: Final Image
FROM quantum-base
WORKDIR /app

# Install Python Dependencies
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy

# Copy Built Components
COPY --from=frontend-builder /app/chart/dist/ ./chart_memory_manager/dist/
COPY . .

# Quantum Runtime Config
ARG QISKIT_IBM_QUANTUM_TOKEN
ARG OQS_KEY_ENCRYPTION_KEY
ENV QISKIT_IBM_QUANTUM_TOKEN=${QISKIT_IBM_QUANTUM_TOKEN}
ENV OQS_KEY_ENCRYPTION_KEY=${OQS_KEY_ENCRYPTION_KEY}

# GPU Acceleration for Quantum Sims
ENV QISKIT_USE_CUDA=1
ENV QULACS_USE_CUDA=1

CMD ["gunicorn", "src.main:app", "-b", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker"]
