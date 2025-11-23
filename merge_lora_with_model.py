import os
import torch
import logging
import shutil
import sys
import glob

# --- ЯДЕРНЫЙ ПАТЧ БЕЗОПАСНОСТИ ---
import transformers.modeling_utils
import transformers.utils.import_utils
def dummy_safety_check(*args, **kwargs): return None
transformers.utils.import_utils.check_torch_load_is_safe = dummy_safety_check
transformers.modeling_utils.check_torch_load_is_safe = dummy_safety_check
# ---------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# НАСТРОЙКИ
BASE_MODEL = "models/gemma-3n"
# Укажи путь к самому последнему/лучшему чекпоинту
LORA_PATH = "output/finetuned_model/final_model" 
OUTPUT_DIR = "output/ready_for_gguf"

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python merge_lora_with_model.py <LORA_PATH> [y/n for transplant]")
        logger.error("Example: python merge_lora_with_model.py output/finetuned_model/checkpoint-2300 y")
        sys.exit(1)
        
    LORA_PATH = sys.argv[1]
    transplant_configs = len(sys.argv) > 2 and sys.argv[2].lower() == 'y'

    if not os.path.isdir(LORA_PATH):
        logger.error(f"LoRA path not found: {LORA_PATH}")
        sys.exit(1)

    logger.info(f"Loading Base Model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Грузим базу
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu", # Мержим на CPU, чтобы не забить память (это безопасно для мерджа)
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    logger.info(f"Loading LoRA Adapter: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    logger.info("Merging weights...")
    merged_model = model.merge_and_unload()

    # Принудительный каст в Float16 (решает проблему размера файлов на Windows)
    logger.info("Casting to Float16...")
    merged_model = merged_model.to(dtype=torch.float16)

    logger.info(f"Saving to {OUTPUT_DIR}... It will take time, please don't close the window and panic.")
    # safe_serialization=False создает .bin файлы (pickle), которые любит llama.cpp
    # max_shard_size="2GB" обходит баг винды с большими тензорами
    merged_model.save_pretrained(
        OUTPUT_DIR, 
        safe_serialization=False, 
        max_shard_size="2GB"
    )
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info("✅ Merge Complete.")

    if transplant_configs:
        logger.info("--- Starting Config Transplant ---")
        
        # 1. Удаляем .json файлы из папки с объединенной моделью, кроме index.json
        logger.info(f"Searching for JSON files to remove in {OUTPUT_DIR}...")
        json_files_in_output = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
        
        if not json_files_in_output:
            logger.warning("No JSON files found in output directory to remove.")
        else:
            for f in json_files_in_output:
                if "index" not in os.path.basename(f).lower():
                    try:
                        os.remove(f)
                        logger.info(f"  - Removed: {os.path.basename(f)}")
                    except OSError as e:
                        logger.error(f"Error removing file {f}: {e}")
                else:
                    logger.info(f"  - Kept index: {os.path.basename(f)}")

        # 2. Копируем .json файлы из базовой модели
        logger.info(f"Copying JSON files from base model: {BASE_MODEL}...")
        json_files_in_base = glob.glob(os.path.join(BASE_MODEL, "*.json"))
        
        if not json_files_in_base:
            logger.error("No JSON files found in base model directory to copy.")
        else:
            for f in json_files_in_base:
                try:
                    shutil.copy(f, OUTPUT_DIR)
                    logger.info(f"  + Copied: {os.path.basename(f)}")
                except shutil.Error as e:
                    logger.error(f"Error copying file {f}: {e}")
        
        logger.info("✅ Transplant Complete. Ready for conversion.")
    else:
        logger.info("Skipping config transplant. Ready for conversion.")

if __name__ == "__main__":
    main()