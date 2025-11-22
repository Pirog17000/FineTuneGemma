@echo off
setlocal enabledelayedexpansion
title Gemma Training Monitor (TensorBoard)
color 0A

:: ========================================================
:: CONFIGURATION
:: ========================================================
set "VENV_PATH=venv\Scripts\activate.bat"
set "LOG_ROOT=output\finetuned_model\runs"
set "TB_PORT=6006"
set "TB_URL=http://localhost:%TB_PORT%"

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
:: 2. SCAN AND SELECT RUNS
:: ========================================================
cls
echo ========================================================
echo               AVAILABLE TRAINING RUNS
echo ========================================================
echo.

if not exist "%LOG_ROOT%" (
    echo [ERROR] No logs found at %LOG_ROOT%
    echo Have you started training yet?
    pause
    exit /b
)

:: Initialize counter
set count=0

:: Option 0 is always ALL
echo [0] VIEW ALL RUNS (Recommended for comparison)

:: Loop through directories and assign numbers
for /d %%D in ("%LOG_ROOT%\*") do (
    set /a count+=1
    set "run[!count!]=%%~nxD"
    echo [!count!] %%~nxD
)

echo.
echo ========================================================
set /p choice="Select a run number (default 0): "

:: Default to 0 if empty
if "%choice%"=="" set choice=0

:: Set the target path based on choice
if "%choice%"=="0" (
    set "TARGET_LOGDIR=%LOG_ROOT%"
    echo [INFO] Selected: ALL RUNS
) else (
    if defined run[%choice%] (
        set "TARGET_LOGDIR=%LOG_ROOT%\!run[%choice%]!"
        echo [INFO] Selected: !run[%choice%]!
    ) else (
        echo [ERROR] Invalid selection. Defaulting to ALL RUNS.
        set "TARGET_LOGDIR=%LOG_ROOT%"
    )
)

:: ========================================================
:: 3. LAUNCH BROWSER AND TENSORBOARD
:: ========================================================
echo.
echo [INFO] Launching Browser in 3 seconds...
echo [INFO] Starting TensorBoard on port %TB_PORT%...
echo.
echo Press CTRL+C to stop monitoring.

:: Start browser in a separate process with a slight delay to allow TB to warm up
start "" cmd /c "timeout /t 3 >nul & start %TB_URL%"

:: Start TensorBoard (This blocks the script until you close it)
tensorboard --logdir "%TARGET_LOGDIR%" --port %TB_PORT% --reload_interval 30

pause