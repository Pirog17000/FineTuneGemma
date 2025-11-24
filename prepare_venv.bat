@echo off
echo Setting up virtual environment for Gemma fine-tuning...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo [INFO] Installing required Python packages...
python -m pip install --upgrade pip

REM --- Install requirements ---
REM Note: requirements.txt now contains all packages including torch with index-url
echo [INFO] Installing requirements...
python -m pip install -r requirements.txt
if errorlevel 1 goto error

REM --- Clone llama.cpp for GGUF conversion tools ---
echo [INFO] Checking for llama.cpp repository...
if not exist "llama.cpp" (
    echo [INFO] Cloning llama.cpp repository...
    git clone https://github.com/ggml-org/llama.cpp.git
    if errorlevel 1 (
        echo [ERROR] Failed to clone llama.cpp. Please ensure Git is installed and in your PATH.
        goto error
    )
) else (
    echo [INFO] llama.cpp repository already exists.
)

REM --- Verify Installations ---
echo [INFO] Verifying installations...
python -c "import torch; print(f'[OK] PyTorch {torch.__version__} CUDA: {torch.cuda.is_available()}')" || goto error
python -c "import transformers; print('[OK] Transformers')" || goto error
python -c "import peft; print('[OK] PEFT')" || goto error
python -c "import datasets; print('[OK] Datasets')" || goto error
python -c "import llama_cpp; print('[OK] llama-cpp-python')" || goto error
python -c "import bitsandbytes; print('[OK] bitsandbytes')" || goto error
python -c "import tensorboard; print('[OK] tensorboard')" || goto error

echo.
echo [SUCCESS] Virtual environment setup is complete.
echo To activate it manually, run: call venv\Scripts\activate.bat
echo.
goto :end

:error
echo.
echo [FAILED] An error occurred during setup.
echo.
goto :end

:end
pause
endlocal
