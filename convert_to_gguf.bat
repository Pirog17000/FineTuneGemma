@echo off
setlocal enabledelayedexpansion
title Gemma GGUF Converter ^& Quantizer
color 0A

:: ========================================================
:: CONFIGURATION
:: ========================================================
:: Folder where llama.cpp repo is cloned
set "TOOLS_DIR=llama.cpp"

:: Path to your Python environment
set "VENV_PATH=venv\Scripts\activate.bat"

:: Path to the convert script (Python) - inside llama.cpp repo
set "CONVERT_SCRIPT=%TOOLS_DIR%\convert_hf_to_gguf.py"

:: Path to the quantization script (Python wrapper)
set "QUANTIZER=quantize_model.py"

:: Input/Output paths
set "INPUT_DIR=output\ready_for_gguf"
set "BASE_GGUF=output\model_f16.gguf"
set "OUTPUT_DIR=output"

:: ========================================================
:: 1. CHECK ENVIRONMENT
:: ========================================================
if exist "%VENV_PATH%" (
    call "%VENV_PATH%"
    echo [OK] Venv activated.
) else (
    color 0C
    echo [ERROR] Virtual environment not found at "%VENV_PATH%"
    echo Please run prepare_venv.bat first.
    pause
    exit /b
)

:: Check for scripts
if not exist "%CONVERT_SCRIPT%" (
    color 0C
    echo [ERROR] Convert script not found at "%CONVERT_SCRIPT%"
    echo Please make sure llama.cpp is cloned - run prepare_venv.bat.
    pause
    exit /b
)

if not exist "%QUANTIZER%" (
    color 0C
    echo [ERROR] Quantize script not found at "%QUANTIZER%"
    echo This script should be in the project root.
    pause
    exit /b
)

:: ========================================================
:: MAIN MENU LOOP
:: ========================================================
:MENU
cls
echo ========================================================
echo           GEMMA QUANTIZATION FACTORY
echo ========================================================
echo.
echo  [1] CONVERT Base HF to GGUF (F16) -- DO THIS FIRST!
echo.
echo  --- Quantization Options (Requires Step 1) ---
echo  [2] Q8_0   (High Quality, ~8GB)
echo  [3] Q6_K   (Great Quality, ~6.5GB)
echo  [4] Q5_K_M (Balanced, ~5.5GB) - Recommended
echo  [5] Q4_K_M (Standard, ~4.5GB) - Fast
echo  [6] Q3_K_M (Small, ~3.5GB)    - IQ Loss possible
echo.
echo  [Q] Quit
echo.
echo ========================================================
set /p choice="Select an option: "

if /i "%choice%"=="1" goto CONVERT_BASE
if /i "%choice%"=="2" set "Q_TYPE=q8_0" & goto QUANTIZE
if /i "%choice%"=="3" set "Q_TYPE=q6_k" & goto QUANTIZE
if /i "%choice%"=="4" set "Q_TYPE=q5_k_m" & goto QUANTIZE
if /i "%choice%"=="5" set "Q_TYPE=q4_k_m" & goto QUANTIZE
if /i "%choice%"=="6" set "Q_TYPE=q3_k_m" & goto QUANTIZE
if /i "%choice%"=="Q" exit /b

echo Invalid choice.
pause
goto MENU

:: ========================================================
:: STEP 1: BASE CONVERSION
:: ========================================================
:CONVERT_BASE
cls
echo [INFO] converting HF model to F16 GGUF...
echo [INFO] Script: %CONVERT_SCRIPT%
echo [INFO] Input:  %INPUT_DIR%

python "%CONVERT_SCRIPT%" "%INPUT_DIR%" --outfile "%BASE_GGUF%" --outtype f16

if %errorlevel% neq 0 (
    echo [ERROR] Conversion failed!
    echo Check if 'output/ready_for_gguf' has the correct config.json transplanted!
    pause
) else (
    echo [SUCCESS] Created %BASE_GGUF%
    timeout /t 3
)
goto MENU

:: ========================================================
:: STEP 2: QUANTIZATION
:: ========================================================
:QUANTIZE
cls
echo [INFO] Checking for F16 base model...
if not exist "%BASE_GGUF%" (
    echo [ERROR] %BASE_GGUF% not found!
    echo Please run Option [1] first to create the base file.
    pause
    goto MENU
)

set "OUT_FILE=%OUTPUT_DIR%\gemma_lyrics_%Q_TYPE%.gguf"

echo.
echo [PROCESS] Quantizing to %Q_TYPE%...
echo --------------------------------------------------------
python "%QUANTIZER%" "%BASE_GGUF%" "%OUT_FILE%" %Q_TYPE%
echo --------------------------------------------------------

if %errorlevel% neq 0 (
    echo [ERROR] Quantization failed.
) else (
    echo [SUCCESS] Created %OUT_FILE%
)

pause
goto MENU
