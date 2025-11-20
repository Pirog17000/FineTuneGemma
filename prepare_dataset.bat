@echo off
setlocal enabledelayedexpansion

echo =================================================
echo == Dataset Preparation Script                  ==
echo =================================================
echo.

set VENV_PATH=venv

REM --- Activate Virtual Environment ---
echo [INFO] Activating virtual environment...
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run 'prepare_venv.bat'.
    goto :end
)
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    goto :end
)

REM --- Choose Dataset Format ---
:format_choice
echo Choose the dataset format:
echo  1. Instruct (independent instruction-response pairs)
echo  2. Conversational (multi-turn dialogues with chunking)
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
echo [DEBUG] Format chosen.
pause
if "%DATA_FORMAT%"=="conversational" (
    echo [WARNING] Conversational mode will generate a significantly larger dataset.
    echo This may take some time.
)
echo.

set "CHUNK_SIZES_ARG="
if "%DATA_FORMAT%"=="conversational" (
    set "DEFAULT_CHUNK_SIZES=4 8 16"
    echo [CONFIG] Using default chunk sizes: %DEFAULT_CHUNK_SIZES%
    set "CHUNK_SIZES_ARG=--chunk-sizes %DEFAULT_CHUNK_SIZES%"
    echo.
)


REM --- Get Input Folder ---
set "DEFAULT_INPUT_FOLDER=dataset"
set "INPUT_FOLDER="
set /p "INPUT_FOLDER=Enter path to conversation files folder [%DEFAULT_INPUT_FOLDER%]: "
if not defined INPUT_FOLDER set "INPUT_FOLDER=%DEFAULT_INPUT_FOLDER%"

if not exist "%INPUT_FOLDER%" (
    echo [ERROR] Folder '%INPUT_FOLDER%' does not exist.
    goto :end
)

REM --- Get Max Samples ---
set "DEFAULT_MAX_SAMPLES=4000"
set "MAX_SAMPLES="
set /p "MAX_SAMPLES=Enter maximum number of samples [%DEFAULT_MAX_SAMPLES%]: "
if not defined MAX_SAMPLES set "MAX_SAMPLES=%DEFAULT_MAX_SAMPLES%"

REM --- Get Validation Split ---
set "DEFAULT_VALIDATION_SPLIT=5"
set "VALIDATION_SPLIT="
set /p "VALIDATION_SPLIT=Enter validation split percentage (e.g., 5 for 5%%) [%DEFAULT_VALIDATION_SPLIT%]: "
if not defined VALIDATION_SPLIT set "VALIDATION_SPLIT=%DEFAULT_VALIDATION_SPLIT%"

REM --- Prepare Data ---
echo [INFO] Preparing '%DATA_FORMAT%' data from '%INPUT_FOLDER%'...

set "CMD=python prepare_conversation_data.py --input-folder "%INPUT_FOLDER%" --format "%DATA_FORMAT%" --max-samples %MAX_SAMPLES% --validation-split %VALIDATION_SPLIT%"
if "%DATA_FORMAT%"=="conversational" (
    set "CMD=%CMD% --chunk-sizes 4 8 16"
)

echo.
echo [DEBUG] Running command:
echo %CMD%
echo.

%CMD%

if errorlevel 1 (
    echo [ERROR] Data preparation script failed.
    goto :end
)

echo.
echo [SUCCESS] Dataset preparation complete!
echo Check the 'data' directory for the 'conversation_training.jsonl' file.
echo.
goto :end

:error
echo.
echo [FAILED] An error occurred during the process.
echo.
goto :end

:end
pause
endlocal
