#!/bin/bash
set -e

echo "================================================="
echo "== RunPod/Linux Setup Script for Gemma Fine-Tuning =="
echo "================================================="
echo ""

# 1. System Dependencies (RunPod usually runs as root, so sudo might not be needed, but check)
echo "[INFO] Updating system and installing dependencies..."
if [ "$EUID" -ne 0 ]; then 
    SUDO="sudo"
else
    SUDO=""
fi

$SUDO apt-get update
$SUDO apt-get install -y python3-venv git cmake build-essential

# 2. Virtual Environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[INFO] Virtual environment already exists."
fi

# Activate venv
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# 3. Python Packages
echo "[INFO] Installing Python packages..."
python3 -m pip install --upgrade pip

# Install dependencies from requirements.txt
# IMPORTANT: For RunPod/CUDA, we need to ensure llama-cpp-python builds with CUDA support.
# We export CMAKE_ARGS before installing to ensure it picks up CUDA.
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1

echo "[INFO] Installing requirements (compiling llama-cpp-python with CUDA support)..."
# We install other requirements first to avoid build conflicts, then specific heavy ones if needed, 
# but usually requirements.txt is fine if env vars are set.
python3 -m pip install -r requirements.txt

# 4. Clone and Build llama.cpp (for GGUF conversion tools)
if [ ! -d "llama.cpp" ]; then
    echo "[INFO] Cloning llama.cpp repository..."
    git clone https://github.com/ggml-org/llama.cpp.git
else
    echo "[INFO] llama.cpp repository already exists."
fi

# Build llama.cpp tools (useful for quantization/conversion scripts that might be run directly)
# This part is optional if only using python bindings, but good for "whole thing" completeness.
echo "[INFO] Building llama.cpp tools (C++)..."
cd llama.cpp
# Clean previous build if any to ensure CUDA config
rm -rf build
mkdir build
cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j $(nproc)
cd ../..

# 5. Download Model (Optional but convenient)
echo "[INFO] Checking for Gemma model..."
if [ ! -d "models/gemma-3n" ]; then
    echo "[INFO] Model not found. Running download script..."
    python3 download_model.py
else
    echo "[INFO] Model directory exists."
fi

# 6. Verification
echo ""
echo "[INFO] Verifying installations..."
python3 -c "import torch; print(f'[OK] PyTorch {torch.__version__} CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print('[OK] Transformers')"
python3 -c "import peft; print('[OK] PEFT')"
python3 -c "import llama_cpp; print('[OK] llama-cpp-python')"

echo ""
echo "[SUCCESS] Environment setup is complete."
echo "To activate manually: source venv/bin/activate"

