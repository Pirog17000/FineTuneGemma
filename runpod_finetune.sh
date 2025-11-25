#!/bin/bash

# =================================================
# == Gemma Fine-Tuning Script (Linux/RunPod)    ==
# =================================================

# Colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

VENV_PATH="venv"

# --- Activate Virtual Environment ---
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo -e "${GREEN}[INFO] Activating virtual environment...${NC}"
    source "$VENV_PATH/bin/activate"
else
    echo -e "${RED}[ERROR] Virtual environment not found. Please run 'runpod_setup.sh' first.${NC}"
    exit 1
fi

# --- Choose Run Mode ---
while true; do
    echo ""
    echo "================================================="
    echo "== Select Run Mode                             =="
    echo "================================================="
    echo "1. Start NEW Training Run"
    echo "2. RESUME from Checkpoint"
    echo ""
    read -p "Enter your choice (1 or 2): " MODE_CHOICE

    RESUME_ARG=""
    if [ "$MODE_CHOICE" == "2" ]; then
        read -p "Enter path to checkpoint directory (e.g. output/finetuned_model/checkpoint-400): " CHECKPOINT_PATH
        if [ -z "$CHECKPOINT_PATH" ]; then
            echo -e "${RED}[ERROR] Checkpoint path cannot be empty.${NC}"
            continue
        fi
        RESUME_ARG="--resume_from_checkpoint \"$CHECKPOINT_PATH\""
        echo -e "${GREEN}[INFO] Will resume from: $CHECKPOINT_PATH${NC}"
        break
    elif [ "$MODE_CHOICE" == "1" ]; then
        break
    else
        echo -e "${RED}[ERROR] Invalid choice. Please enter 1 or 2.${NC}"
    fi
done

# --- Choose Dataset Format ---
while true; do
    echo ""
    echo "Choose the dataset format you are training on:"
    echo " 1. Instruct (independent instruction-response pairs)"
    echo " 2. Conversational (multi-turn dialogues)"
    echo ""
    read -p "Enter your choice (1 or 2): " FORMAT_CHOICE

    if [ "$FORMAT_CHOICE" == "1" ]; then
        DATA_FORMAT="instruct"
        break
    elif [ "$FORMAT_CHOICE" == "2" ]; then
        DATA_FORMAT="conversational"
        break
    else
        echo -e "${RED}[ERROR] Invalid choice.${NC}"
    fi
done
echo -e "${GREEN}[INFO] Selected format: $DATA_FORMAT${NC}"

# --- Choose Dataset File ---
DATA_DIR="data"
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}[ERROR] The '$DATA_DIR' directory does not exist.${NC}"
    exit 1
fi

echo ""
echo "Choose the TRAINING dataset file from the 'data' directory:"
files=("$DATA_DIR"/*.jsonl)
if [ ${#files[@]} -eq 0 ] || [ ! -e "${files[0]}" ]; then
    echo -e "${RED}[ERROR] No .jsonl files found in '$DATA_DIR'.${NC}"
    exit 1
fi

i=1
for file in "${files[@]}"; do
    filename=$(basename "$file")
    echo " $i. $filename"
    ((i++))
done

while true; do
    echo ""
    read -p "Enter your choice for the training dataset (1-$((i-1))): " FILE_INDEX
    
    if [[ "$FILE_INDEX" =~ ^[0-9]+$ ]] && [ "$FILE_INDEX" -ge 1 ] && [ "$FILE_INDEX" -lt "$i" ]; then
        TRAIN_DATASET_FILE=$(basename "${files[$((FILE_INDEX-1))]}")
        echo -e "${GREEN}[INFO] Selected training dataset: $TRAIN_DATASET_FILE${NC}"
        break
    else
        echo -e "${RED}[ERROR] Invalid choice.${NC}"
    fi
done

# --- Choose Evaluation Dataset File (Optional) ---
EVAL_ARG=""
while true; do
    echo ""
    read -p "Do you want to use an evaluation dataset? (y/n): " USE_EVAL
    if [[ "$USE_EVAL" =~ ^[yY]$ ]]; then
        echo ""
        echo "Choose the EVALUATION dataset file:"
        
        j=1
        for file in "${files[@]}"; do
            filename=$(basename "$file")
            echo " $j. $filename"
            ((j++))
        done

        read -p "Enter your choice for the evaluation dataset (1-$((j-1))): " EVAL_INDEX
        
        if [[ "$EVAL_INDEX" =~ ^[0-9]+$ ]] && [ "$EVAL_INDEX" -ge 1 ] && [ "$EVAL_INDEX" -lt "$j" ]; then
            EVAL_DATASET_FILE=$(basename "${files[$((EVAL_INDEX-1))]}")
            EVAL_ARG="--eval_dataset_file \"$EVAL_DATASET_FILE\""
            echo -e "${GREEN}[INFO] Selected evaluation dataset: $EVAL_DATASET_FILE${NC}"
            break
        else
            echo -e "${RED}[ERROR] Invalid choice.${NC}"
        fi
    elif [[ "$USE_EVAL" =~ ^[nN]$ ]]; then
        break
    else
        echo -e "${RED}[ERROR] Invalid input. Please enter 'y' or 'n'.${NC}"
    fi
done

# --- Configure Training Parameters ---
echo ""
echo "[INFO] Configure Training Parameters (leave blank for defaults)."
if [ -n "$RESUME_ARG" ]; then
    echo " [NOTE] Resuming: Ensure epochs > completed epochs."
fi

read -p "Enter number of training epochs (default: 4): " NUM_TRAIN_EPOCHS
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-4}

EVAL_STEPS=20
if [ -n "$EVAL_ARG" ]; then
    read -p "Enter evaluation frequency in steps (default: 20): " INPUT_EVAL_STEPS
    EVAL_STEPS=${INPUT_EVAL_STEPS:-20}
fi

read -p "Enter checkpoint save frequency in steps (default: 20): " SAVE_STEPS
SAVE_STEPS=${SAVE_STEPS:-20}

echo -e "${GREEN}[INFO] Training with $NUM_TRAIN_EPOCHS epochs, eval every $EVAL_STEPS steps, save every $SAVE_STEPS steps.${NC}"
echo ""

# --- Run Fine-Tuning ---
echo "[INFO] Starting fine-tuning..."
# Note: Using python3 directly as venv is activated
# We construct the command carefully to handle spaces and empty args
CMD="python3 finetune_gemma.py --dataset_format $DATA_FORMAT --train_dataset_file \"$TRAIN_DATASET_FILE\" $EVAL_ARG --num_train_epochs $NUM_TRAIN_EPOCHS --eval_steps $EVAL_STEPS --save_steps $SAVE_STEPS $RESUME_ARG"

echo "Running: $CMD"
eval $CMD

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Fine-tuning script failed.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}[SUCCESS] Fine-tuning complete!${NC}"
echo "Check the 'output/finetuned_model' directory for the adapter."
echo ""

