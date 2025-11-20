@echo off
echo Downloading Gemma model from HuggingFace...

REM Check if venv exists
if not exist venv (
    echo ERROR: Virtual environment not found. Run prepare_venv.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Create models directory
if not exist models mkdir models

REM Temporarily enable online mode for download
set HF_HUB_OFFLINE=0
set HF_HUB_DISABLE_TELEMETRY=1

REM Check if model already exists
if exist models\gemma-3n (
    echo Model already exists in models\gemma-3n
    echo If you want to re-download, delete the folder first.
    goto end
)

REM Download the model
echo Downloading Gemma-3n model (this may take several minutes)...
python download_model.py

if errorlevel 1 (
    echo.
    echo ERROR: Model download failed.
    echo Check your internet connection and try again.
    pause
    exit /b 1
) else (
    echo.
    echo Model downloaded successfully!
    echo Location: models\gemma-3n
    echo.
)

:end
echo To continue with fine-tuning, run: finetune_gemma.bat
pause
exit /b 0
