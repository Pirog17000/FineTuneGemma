@echo off
setlocal enabledelayedexpansion

echo =================================================
echo == Gemma GGUF Conversion Script                ==
echo =================================================
echo.

REM --- Configuration ---
set BASE_MODEL_PATH=models\gemma-3n
set ADAPTER_PATH=output\finetuned_model
set VENV_PATH=venv

REM --- 1. Check for Virtual Environment ---
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at '%VENV_PATH%'.
    echo Please run 'prepare_venv.bat' first.
    goto :eof
)

REM --- 2. Activate Virtual Environment ---
echo [INFO] Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate the virtual environment.
    goto :eof
)

REM --- 3. Verify Required Components ---
echo [INFO] Verifying required components...

REM Check for base model
if not exist "%BASE_MODEL_PATH%" (
    echo [ERROR] Base model not found at '%BASE_MODEL_PATH%'.
    echo Please run 'download_model.bat' to download it.
    goto :eof
)
echo [OK] Base model found.

REM Check for fine-tuned LoRA adapters
if not exist "%ADAPTER_PATH%" (
    echo [ERROR] Fine-tuned adapters directory not found at '%ADAPTER_PATH%'.
    echo Please ensure the training step has completed successfully.
    goto :eof
)
echo [OK] Fine-tuned adapters directory found.

REM --- 4. Get Quantization Type ---
set "DEFAULT_QUANT_TYPE=f16"
echo.
echo [CONFIG] Select the quantization type for the GGUF model.
echo Common options:
echo   - f16    (Highest quality, largest size)
echo   - q8_0   (Good quality, medium size)
echo   - q5_k_m (Good balance, smaller size)
echo   - q4_k_m (Recommended for quality/size ratio)
echo.
set "QUANT_TYPE="
set /p "QUANT_TYPE=Enter quantization type [%DEFAULT_QUANT_TYPE%]: "
if not defined QUANT_TYPE set "QUANT_TYPE=%DEFAULT_QUANT_TYPE%"


REM --- 5. Run GGUF Conversion ---
echo [INFO] Starting the GGUF conversion process with type '%QUANT_TYPE%'...
python convert_to_gguf.py --quant-type %QUANT_TYPE%
if errorlevel 1 (
    echo [ERROR] GGUF conversion script failed.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] GGUF conversion completed successfully!
echo Check the 'output' directory for the final model.
echo.
pause
exit /b 0
