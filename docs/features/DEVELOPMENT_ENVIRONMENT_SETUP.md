# Development Environment Setup for EA-NN 🏗️

## Overview
This document outlines the development environment setup for EA-NN, including containerization strategies, virtual environments, and consistent development workflows.

## Current Environment Issues ⚠️

### Problem: Externally-Managed Environment
Modern Python installations (Python 3.12+) use PEP 668 to prevent system-wide package installations:
```
error: externally-managed-environment
× This environment is externally managed
```

### Solution: Virtual Environments & Containerization
We've implemented multiple approaches for consistent development environments.

## Current Setup Solutions 🔧

### 1. Virtual Environment (Immediate Fix)
The `setup_gpu.sh` script now automatically creates isolated environments:

```bash
# Creates virtual environment at ../ea-nn-gpu-env
./scripts/gpu/setup_gpu.sh
```

**Environment Activation:**
- **Linux/WSL:** `source ea-nn-gpu-env/bin/activate`
- **Windows:** `ea-nn-gpu-env\Scripts\activate.bat`

### 2. Manual Virtual Environment Setup
```bash
# Create environment
python3 -m venv ea-nn-env

# Activate (Linux/WSL)
source ea-nn-env/bin/activate

# Activate (Windows)
ea-nn-env\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Future Containerization Strategy 🐳

### Docker Development Environment

**Dockerfile (GPU-enabled):**
```dockerfile
FROM nvidia/cuda:12.4-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python3 -m venv venv
RUN . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Copy application
COPY . .

# Expose web interface port
EXPOSE 5000

# Activate venv and run
CMD ["/bin/bash", "-c", "source venv/bin/activate && python main.py --web"]
```

**Docker Compose (Development Stack):**
```yaml
version: '3.8'
services:
  ea-nn:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - .:/app
      - ea-nn-data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  jupyter:
    build: .
    command: bash -c "source venv/bin/activate && jupyter notebook --ip=0.0.0.0 --allow-root"
    ports:
      - "8888:8888"
    volumes:
      - .:/app
    environment:
      - CUDA_VISIBLE_DEVICES=0

volumes:
  ea-nn-data:
```

### Development Containers (VS Code)

**`.devcontainer/devcontainer.json`:**
```json
{
    "name": "EA-NN GPU Development",
    "dockerFile": "../Dockerfile",
    "runArgs": ["--gpus", "all"],
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.pylint",
                "ms-toolsai.jupyter",
                "nvidia.nsight-vscode-edition"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/app/venv/bin/python",
                "python.terminal.activateEnvironment": true
            }
        }
    },
    "forwardPorts": [5000, 8888],
    "postCreateCommand": "source venv/bin/activate && pip install -e .",
    "remoteUser": "root"
}
```

## Environment Management Best Practices 📋

### 1. Dependency Management
```bash
# Generate requirements
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### 2. GPU Environment Validation
```python
# scripts/validate_environment.py
import torch
import sys

def validate_gpu_environment():
    print("🔍 Environment Validation")
    print("=" * 30)
    
    # Python version
    print(f"Python: {sys.version}")
    
    # PyTorch version
    print(f"PyTorch: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    return cuda_available

if __name__ == "__main__":
    validate_gpu_environment()
```

### 3. Cross-Platform Compatibility
```bash
# Platform detection in setup scripts
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows (Git Bash/Cygwin)
    PLATFORM="windows"
elif [[ $(uname -r) == *microsoft* ]]; then
    # WSL
    PLATFORM="wsl"
else
    # Linux/Mac
    PLATFORM="unix"
fi
```

## Recommended Development Workflow 🚀

### For Individual Development
1. **Use Virtual Environments** (current solution)
   ```bash
   python3 -m venv ea-nn-env
   source ea-nn-env/bin/activate  # or .bat on Windows
   pip install -r requirements.txt
   ```

### For Team Development
1. **Docker Development Containers**
   - Consistent environment across all developers
   - GPU support built-in
   - VS Code integration

### For Production Deployment
1. **Kubernetes with GPU Operators**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: ea-nn-gpu
   spec:
     template:
       spec:
         containers:
         - name: ea-nn
           image: ea-nn:gpu
           resources:
             limits:
               nvidia.com/gpu: 1
   ```

## Current Status ✅

- ✅ **Virtual Environment Setup**: Automated in `setup_gpu.sh`
- ✅ **GPU Detection**: RTX 2060 support in `main.py`
- ✅ **Cross-Platform**: WSL/Linux/Windows compatibility
- 🚧 **Containerization**: Planned for Phase 3
- 🚧 **CI/CD Pipeline**: Future enhancement

## Next Steps 📈

1. **Immediate** (Phase 2 Complete):
   - ✅ Use virtual environment setup
   - ✅ Test GPU acceleration
   - ✅ Create pull request

2. **Short-term** (Phase 3):
   - 🔄 Implement Docker containers
   - 🔄 Add development container support
   - 🔄 Create automated environment validation

3. **Long-term**:
   - 🔄 Kubernetes deployment
   - 🔄 Cloud development environments
   - 🔄 Automated GPU cluster management

## Troubleshooting 🔧

### Common Issues:

1. **"externally-managed-environment"**
   - Solution: Use virtual environments (automated in setup script)

2. **CUDA not detected**
   - Check NVIDIA drivers: `nvidia-smi`
   - Verify PyTorch CUDA build: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Windows/WSL GPU issues**
   - Use Windows Python for GPU access
   - Install CUDA drivers in Windows (not WSL)

4. **Virtual environment activation**
   - Linux/WSL: `source ea-nn-env/bin/activate`
   - Windows: `ea-nn-env\Scripts\activate.bat`

---

**Author**: EA-NN Development Team  
**Last Updated**: August 3, 2025  
**Status**: Active Development - Phase 2 Complete
