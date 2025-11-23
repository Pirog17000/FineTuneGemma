@echo off
setlocal enabledelayedexpansion
title LoRA Checkpoint Merger
color 0B

:: ========================================================
:: CONFIGURATION
:: ========================================================
set "VENV_PATH=venv\Scripts\activate.bat"
set "CHECKPOINT_ROOT=output\finetuned_model"
set "MERGE_SCRIPT=merge_lora_with_model.py"

:: ========================================================
:: 1. ACTIVATE VENV
:: ========================================================
echo [INFO] Checking Virtual Environment...
if exist "%VENV_PATH%" (
    call "%VENV_PATH%"
    echo [OK] Venv activated.
) else (
    echo [ERROR] Virtual environment not found at %VENV_PATH%
    echo Please make sure you are running this from the project root.
    pause
    exit /b
)

:: ========================================================
:: 2. SCAN AND SELECT CHECKPOINT
:: ========================================================
:SELECT_CHECKPOINT
cls
echo ========================================================
echo           AVAILABLE CHECKPOINTS FOR MERGING
echo ========================================================
echo.

if not exist "%CHECKPOINT_ROOT%" (
    echo [ERROR] No checkpoints found at %CHECKPOINT_ROOT%
    echo Have you started training yet?
    pause
    exit /b
)

:: Initialize counter
set count=0

:: Loop through directories and assign numbers
for /d %%D in ("%CHECKPOINT_ROOT%\*") do (
    set /a count+=1
    set "run[!count!]=%%~nxD"
    echo [!count!] %%~nxD
)

echo.
echo ========================================================
set /p choice="Select a checkpoint number to merge: "

:: Validate choice
if "%choice%"=="" (
    echo [ERROR] No selection made. Please try again.
    timeout /t 2 >nul
    goto SELECT_CHECKPOINT
)
if not defined run[%choice%] (
    echo [ERROR] Invalid selection. Please try again.
    timeout /t 2 >nul
    goto SELECT_CHECKPOINT
)

set "SELECTED_CHECKPOINT_NAME=!run[%choice%]!"
set "SELECTED_LORA_PATH=%CHECKPOINT_ROOT%\%SELECTED_CHECKPOINT_NAME%"
echo [INFO] Selected: %SELECTED_CHECKPOINT_NAME%

:: ========================================================
:: 3. ASK TO TRANSPLANT CONFIG
:: ========================================================
echo.
set "transplant_choice="
set /p transplant_choice="Transplant original model configs? (y/n, default n): "
if /i "%transplant_choice%"=="y" (
    set "TRANSPLANT_ARG=y"
    echo [INFO] Will transplant original configs after merge.
) else (
    set "TRANSPLANT_ARG=n"
    echo [INFO] Using merged model's original configs.
)

:: ========================================================
:: 4. RUN MERGE SCRIPT
:: ========================================================
echo.
echo [INFO] Starting merge process...
echo [INFO] LORA Path: %SELECTED_LORA_PATH%
echo.

python "%MERGE_SCRIPT%" "%SELECTED_LORA_PATH%" "%TRANSPLANT_ARG%"

echo.
echo ========================================================
echo [OK] Script finished.
echo ========================================================
pause
