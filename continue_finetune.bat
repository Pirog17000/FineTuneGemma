@echo off
setlocal enabledelayedexpansion

echo =================================================
echo == Gemma Continued Fine-Tuning Script          ==
echo =================================================
echo.

set VENV_PATH=venv

REM --- Activate Virtual Environment ---
echo [INFO] Activating virtual environment...
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run 'prepare_venv.bat'.
    goto :eof
)
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    goto :eof
)

REM --- Choose Dataset Format ---
:format_choice
echo Choose the dataset format you are training on:
echo  1. Instruct (independent instruction-response pairs)
echo  2. Conversational (multi-turn dialogues)
echo.
set "CHOICE="
set /p "CHOICE=Enter your choice (1 or 2): "

set "DATA_FORMAT="
if "%CHOICE%"=="1" set DATA_FORMAT=instruct
if "%CHOICE%"=="2" set DATA_FORMAT=conversational

if not defined DATA_FORMAT (
    echo [ERROR] Invalid choice. Please enter 1 or 2.
    echo.
    goto :format_choice
)
echo [INFO] Selected format: %DATA_FORMAT%
echo.

REM --- Configuration with user prompts ---
set "DEFAULT_ADAPTER_PATH=output\finetuned_model"
set "DEFAULT_OUTPUT_DIR=output\finetuned_model"
set "DEFAULT_NUM_EPOCHS=1"

echo [CONFIG] Please confirm the settings for continued training.
echo Press ENTER to accept the default values in brackets [].
echo.

set "ADAPTER_TO_RESUME="
set /p "ADAPTER_TO_RESUME=Enter path to the adapter to resume from [%DEFAULT_ADAPTER_PATH%]: "
if not defined ADAPTER_TO_RESUME set "ADAPTER_TO_RESUME=%DEFAULT_ADAPTER_PATH%"

set "NEW_OUTPUT_DIR="
set /p "NEW_OUTPUT_DIR=Enter the output directory to save to [%DEFAULT_OUTPUT_DIR%]: "
if not defined NEW_OUTPUT_DIR set "NEW_OUTPUT_DIR=%DEFAULT_OUTPUT_DIR%"

set "NUM_EPOCHS="
set /p "NUM_EPOCHS=Enter number of additional epochs to train for [%DEFAULT_NUM_EPOCHS%]: "
if not defined NUM_EPOCHS set "NUM_EPOCHS=%DEFAULT_NUM_EPOCHS%"
REM --- ============================================ ---

echo.
echo [INFO] Starting continued fine-tuning with the following settings:
echo   - Resuming from: %ADAPTER_TO_RESUME%
echo   - Saving to:     %NEW_OUTPUT_DIR%
echo   - Epochs:        %NUM_EPOCHS%
echo.

python continue_finetune.py ^
    --resume_from_adapter "%ADAPTER_TO_RESUME%" ^
    --output_dir "%NEW_OUTPUT_DIR%" ^
    --num_epochs %NUM_EPOCHS% ^
    --dataset_format %DATA_FORMAT%

if errorlevel 1 (
    echo [ERROR] Fine-tuning script failed.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Continued fine-tuning complete!
echo Check the '%NEW_OUTPUT_DIR%' directory for the results.
echo.
pause
exit /b 0
