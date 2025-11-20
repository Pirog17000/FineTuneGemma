@echo off
setlocal enabledelayedexpansion

echo =================================================
echo == Gemma Fine-Tuning Script                    ==
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


REM --- Choose Dataset File ---
:dataset_choice
echo Choose the TRAINING dataset file from the 'data' directory:
set "DATA_DIR=data"
set /a "file_count=0"

if not exist "%DATA_DIR%" (
    echo [ERROR] The '%DATA_DIR%' directory does not exist.
    pause
    goto :eof
)

for %%f in ("%DATA_DIR%\*.jsonl") do (
    set /a "file_count+=1"
    echo  !file_count!. %%~nxf
    set "file_!file_count!=%%~nxf"
)

if %file_count% equ 0 (
    echo [ERROR] No .jsonl files found in the '%DATA_DIR%' directory.
    pause
    goto :eof
)
echo.

set "FILE_CHOICE="
set /p "FILE_CHOICE=Enter your choice for the training dataset (1-%file_count%): "

if "!FILE_CHOICE!"=="" (
    echo [ERROR] No choice entered.
    echo.
    goto :dataset_choice
)

if not defined file_!FILE_CHOICE! (
    echo [ERROR] Invalid choice. Please enter a number between 1 and %file_count%.
    echo.
    goto :dataset_choice
)

set "TRAIN_DATASET_FILE=!file_%FILE_CHOICE%!"
echo [INFO] Selected training dataset: %TRAIN_DATASET_FILE%
echo.

REM --- Choose Evaluation Dataset File (Optional) ---
set "EVAL_ARG="
:eval_choice_prompt
set "USE_EVAL="
set /p "USE_EVAL=Do you want to use an evaluation dataset? (y/n): "

if /i "%USE_EVAL%"=="y" (
    echo.
    echo Choose the EVALUATION dataset file from the 'data' directory:
    set /a "eval_file_count=0"
    for %%f in ("%DATA_DIR%\*.jsonl") do (
        set /a "eval_file_count+=1"
        echo  !eval_file_count!. %%~nxf
        set "eval_file_!eval_file_count!=%%~nxf"
    )
    echo.
    
    set "EVAL_FILE_CHOICE="
    set /p "EVAL_FILE_CHOICE=Enter your choice for the evaluation dataset (1-!eval_file_count!): "

    if "!EVAL_FILE_CHOICE!"=="" (
        echo [ERROR] No choice entered.
        echo.
        goto :eval_choice_prompt
    )

    REM --- FIX: Check variable existence dynamically ---
    set "VALID_CHOICE=0"
    for /l %%i in (1,1,!eval_file_count!) do (
        if "%%i"=="!EVAL_FILE_CHOICE!" set "VALID_CHOICE=1"
    )

    if "!VALID_CHOICE!"=="0" (
        echo [ERROR] Invalid choice. Please enter a number between 1 and !eval_file_count!.
        echo.
        goto :eval_choice_prompt
    )

    REM --- FIX: Indirect variable expansion using a FOR loop ---
    for %%i in (!EVAL_FILE_CHOICE!) do set "EVAL_DATASET_FILE=!eval_file_%%i!"
    
    echo [INFO] Selected evaluation dataset: !EVAL_DATASET_FILE!
    
    REM --- FIX: Construct argument with clean quoting ---
    set "EVAL_ARG=--eval_dataset_file "!EVAL_DATASET_FILE!""

) else if /i not "%USE_EVAL%"=="n" (
    echo [ERROR] Invalid input. Please enter 'y' or 'n'.
    goto :eval_choice_prompt
)
echo.


REM --- Configure Training Parameters ---
echo.
echo [INFO] Configure Training Parameters (leave blank for defaults).
set "NUM_TRAIN_EPOCHS="
set /p "NUM_TRAIN_EPOCHS=Enter number of training epochs (default: 4): "
if "%NUM_TRAIN_EPOCHS%"=="" set NUM_TRAIN_EPOCHS=4

set "EVAL_STEPS="
set /p "EVAL_STEPS=Enter evaluation frequency in steps (default: 20): "
if "%EVAL_STEPS%"=="" set EVAL_STEPS=20

set "SAVE_STEPS="
set /p "SAVE_STEPS=Enter checkpoint save frequency in steps (default: 20): "
if "%SAVE_STEPS%"=="" set SAVE_STEPS=20

echo [INFO] Training with %NUM_TRAIN_EPOCHS% epochs, eval every %EVAL_STEPS% steps, save every %SAVE_STEPS% steps.
echo.


REM --- Run Fine-Tuning ---
echo [INFO] Starting fine-tuning...
python finetune_gemma.py --dataset_format %DATA_FORMAT% --train_dataset_file "%TRAIN_DATASET_FILE%" %EVAL_ARG% --num_train_epochs %NUM_TRAIN_EPOCHS% --eval_steps %EVAL_STEPS% --save_steps %SAVE_STEPS%
if errorlevel 1 (
    echo [ERROR] Fine-tuning script failed.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Fine-tuning complete!
echo Check the 'output/finetuned_model' directory for the adapter.
echo.
pause
exit /b 0

