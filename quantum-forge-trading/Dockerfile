# Multi-Stage Build for Quantum Trading Bot
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as quantum-base
RUN apt-get update && \
    apt-get install -y python3.11 pipenv ocl-icd-opencl-dev \
    libfftw3-dev libopenslide-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Open Quantum Safe Libraries
RUN git clone https://github.com/open-quantum-safe/liboqs && \
    cd liboqs && mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && make install

FROM node:18-slim as frontend-builder
WORKDIR /app/ui
COPY ui/package.json .
COPY ui/package-lock.json .
RUN npm ci
COPY ui/ .
RUN npm run build

FROM quantum-base
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --from=frontend-builder /app/ui/dist ./ui/dist
COPY . .

ENV QISKIT_IBM_QUANTUM_TOKEN="your_ibm_token"
ENV OQS_KEY_ENCRYPTION_KEY="your_enc_key"
ENV QISKIT_USE_CUDA=1

CMD ["python", "-m", "src.core.trading_engine"]
