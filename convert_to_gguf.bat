@echo off
setlocal enabledelayedexpansion
title Gemma GGUF Converter & Quantizer (Auto-Updater)
color 0A

:: ========================================================
:: CONFIGURATION
:: ========================================================
:: Folder where tools will be kept
set "TOOLS_DIR=llama.cpp"

:: Path to your Python environment
set "VENV_PATH=venv\Scripts\activate.bat"

:: Path to the convert script (Python) - usually downloaded manually or git cloned
set "CONVERT_SCRIPT=%TOOLS_DIR%\convert_hf_to_gguf.py"

:: Path to the llama-quantize executable (EXE)
set "QUANTIZER=%TOOLS_DIR%\llama-quantize.exe"

:: Input/Output paths
set "INPUT_DIR=output\ready_for_gguf"
set "BASE_GGUF=output\model_f16.gguf"
set "OUTPUT_DIR=output"

:: Create tools dir if missing
if not exist "%TOOLS_DIR%" mkdir "%TOOLS_DIR%"

:: ========================================================
:: 1. CHECK & DOWNLOAD TOOLS (THE MAGIC PART)
:: ========================================================
if exist "%QUANTIZER%" goto ACTIVATE_ENV

cls
echo [WARNING] llama-quantize.exe not found in %TOOLS_DIR%!
echo.
echo [AUTO-DOWNLOAD] Fetching the LATEST release from GitHub...
echo This may take a minute depending on your internet.
echo.

:: PowerShell one-liner to:
:: 1. Get latest release JSON from GitHub API
:: 2. Find the asset URL containing 'bin-win-avx2-x64.zip' (Most compatible/fast)
:: 3. Download it as 'llama_temp.zip'
:: 4. Unzip it into the TOOLS_DIR
:: 5. Delete the zip
powershell -Command "$ProgressPreference = 'SilentlyContinue'; Write-Host '   --> Querying GitHub API...'; $json=Invoke-RestMethod -Uri 'https://api.github.com/repos/ggerganov/llama.cpp/releases/latest'; $asset=$json.assets | Where-Object {$_.name -like '*bin-win-avx2-x64.zip'}; $url=$asset.browser_download_url; Write-Host '   --> Downloading: ' $asset.name; Invoke-WebRequest -Uri $url -OutFile 'llama_temp.zip'; Write-Host '   --> Extracting...'; Expand-Archive -Path 'llama_temp.zip' -DestinationPath '%TOOLS_DIR%' -Force; Remove-Item 'llama_temp.zip'; Write-Host '   --> Done!'"

if not exist "%QUANTIZER%" (
    color 0C
    echo.
    echo [ERROR] Download failed or file structure changed.
    echo Please download 'llama-*-bin-win-avx2-x64.zip' manually from:
    echo https://github.com/ggerganov/llama.cpp/releases
    echo and extract it to: %TOOLS_DIR%
    pause
    exit /b
) else (
    echo [OK] Tools updated successfully.
    timeout /t 2 >nul
)

:: ========================================================
:: 2. ACTIVATE VENV
:: ========================================================
:ACTIVATE_ENV
echo [INFO] Checking Virtual Environment...
if exist "%VENV_PATH%" (
    call "%VENV_PATH%"
    echo [OK] Venv activated.
) else (
    color 0C
    echo [ERROR] Virtual environment not found at "%VENV_PATH%"
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

if not exist "%CONVERT_SCRIPT%" (
    echo [ERROR] Could not find convert script at: %CONVERT_SCRIPT%
    echo Please make sure convert_hf_to_gguf.py is inside the %TOOLS_DIR% folder.
    echo (The auto-downloader only fetches the EXE binaries, not the python script).
    pause
    goto MENU
)

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
"%QUANTIZER%" "%BASE_GGUF%" "%OUT_FILE%" %Q_TYPE%
echo --------------------------------------------------------

if %errorlevel% neq 0 (
    echo [ERROR] Quantization failed.
) else (
    echo [SUCCESS] Created %OUT_FILE%
)

pause
goto MENU