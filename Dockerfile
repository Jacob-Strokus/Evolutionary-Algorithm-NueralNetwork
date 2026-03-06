# Production Dockerfile for EA-NN Neural Ecosystem Simulation
FROM nvidia/cuda:12.4-runtime-ubuntu22.04

# Metadata
LABEL maintainer="EA-NN Team"
LABEL description="EA-NN Neural Ecosystem Simulation with GPU Acceleration"
LABEL version="2.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV FLASK_ENV=production
ENV FLASK_APP=main.py

# Install system dependencies (minimal for production)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user
RUN useradd -m -u 1000 ea-nn && \
    mkdir -p /app && \
    chown ea-nn:ea-nn /app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Copy application code
COPY --chown=ea-nn:ea-nn . .

# Switch to non-root user
USER ea-nn

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Default command
CMD ["python3", "main.py", "--web", "--production"]
