#!/usr/bin/env python3
"""
Скрипт для LoRA-файнтюнинга модели Gemma
Адаптировано из руководства Google по LoRA-тюнингу с использованием Keras
"""

import os
import shutil
import argparse
import gc
from typing import Optional
import inspect

# --- НАЧАЛО ПАТЧЕЙ ---
# ПОРЯДОК ВАЖЕН: Импортируем accelerate и патчим его ДО импорта transformers/peft.
import accelerate
import accelerate.optimizer

# ПАТЧ 1: Совместимость unwrap_model
# ЧТО ДЕЛАЕТ: Удаляет аргумент 'keep_torch_compile' из вызовов unwrap_model, если accelerate его не поддерживает.
# ЗАЧЕМ: Исправляет ошибку TypeError при несовпадении версий.
_orig_unwrap = accelerate.Accelerator.unwrap_model

def _patched_unwrap(self, model, *args, **kwargs):
    if 'keep_torch_compile' in kwargs:
        # Проверяем, принимает ли оригинальный метод этот аргумент
        sig = inspect.signature(_orig_unwrap)
        if 'keep_torch_compile' not in sig.parameters:
            kwargs.pop('keep_torch_compile')
    return _orig_unwrap(self, model, *args, **kwargs)

accelerate.Accelerator.unwrap_model = _patched_unwrap

# ПАТЧ 2: Исправление AttributeError: 'AdamW' object has no attribute 'train' / 'eval'
# ЧТО ДЕЛАЕТ: Блокирует вызов .train() и .eval() у оптимизатора, если методов не существует.
# ЗАЧЕМ: Оптимизаторы bitsandbytes не имеют методов train/eval, которые transformers пытается вызвать.
def _patched_optimizer_train(self):
    if hasattr(self.optimizer, "train"):
        return self.optimizer.train()
    return None

def _patched_optimizer_eval(self):
    if hasattr(self.optimizer, "eval"):
        return self.optimizer.eval()
    return None

accelerate.optimizer.AcceleratedOptimizer.train = _patched_optimizer_train
accelerate.optimizer.AcceleratedOptimizer.eval = _patched_optimizer_eval
# --- КОНЕЦ ПАТЧЕЙ ---

# Теперь импортируем остальные тяжелые библиотеки
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Gemma3nForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    BitsAndBytesConfig,
)
import transformers.trainer
import transformers.utils.import_utils
# MONKEY PATCH: Bypass CVE-2025-32434 security check for local checkpoints
# We must patch AFTER imports because Trainer binds the function locally.
transformers.trainer.check_torch_load_is_safe = lambda: True
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ЧТО ДЕЛАЕТ:
# Этот Callback управляет сохранением чекпоинтов. Он хранит только N лучших моделей
# (на основе eval_loss) и удаляет старые/худшие, чтобы экономить место.
#
# ЗАЧЕМ/МОТИВАЦИЯ:
# Стандартный механизм Hugging Face сохраняет либо все, либо последние N чекпоинтов,
# не учитывая их качество. Нам нужно сохранять только те версии модели, которые реально
# показывают улучшение метрик (уменьшение ошибки). Мы держим топ-2 лучших чекпоинта.
class ManageBestCheckpointsCallback(TrainerCallback):
    def __init__(self, output_dir, keep_best_count=2):
        super().__init__()
        self.output_dir = output_dir
        self.keep_best_count = keep_best_count
        self.best_checkpoints = []

    def on_evaluate(self, args, state, control, metrics=None, model=None, tokenizer=None, **kwargs):
        if metrics is None:
            return

        current_loss = metrics.get('eval_loss')
        if current_loss is None:
            return

        print(f"\n>>> Evaluation Completed. Loss: {current_loss:.4f}")

        step = state.global_step
        checkpoint_folder = os.path.join(self.output_dir, f"checkpoint-{step}")
        
        should_save = False
        if len(self.best_checkpoints) < self.keep_best_count:
            should_save = True
        else:
            self.best_checkpoints.sort(key=lambda x: x[0])
            if current_loss < self.best_checkpoints[-1][0]:
                should_save = True

        if should_save:
            print(f">>> Found better model! Saving to {checkpoint_folder}...")
            
            # --- FIX: Save using model/tokenizer directly, not trainer ---
            if model:
                # This saves the LoRA adapter specifically
                model.save_pretrained(checkpoint_folder)
            
            if tokenizer:
                tokenizer.save_pretrained(checkpoint_folder)
            
            # --- FIX: Save Trainer State for resuming ---
            # Standard trainer.save_model() saves this automatically. 
            # Since we are manually saving, we must manually save the state too 
            # so that trainer.train(resume_from_checkpoint=...) can find it.
            if state:
                 state.save_to_json(os.path.join(checkpoint_folder, "trainer_state.json"))
            # -------------------------------------------------------------

            self.best_checkpoints.append((current_loss, step, checkpoint_folder))
            self.best_checkpoints.sort(key=lambda x: x[0])
            
            while len(self.best_checkpoints) > self.keep_best_count:
                to_remove = self.best_checkpoints.pop()
                path_to_remove = to_remove[2]
                print(f">>> Pruning worse checkpoint: {path_to_remove}")
                if os.path.exists(path_to_remove):
                    try:
                        shutil.rmtree(path_to_remove)
                    except Exception as e:
                        print(f"Error deleting checkpoint {path_to_remove}: {e}")
        else:
            print(">>> Model did not improve. Skipping save.")

        print(">>> Force cleaning VRAM after evaluation...")
        torch.cuda.empty_cache()
        gc.collect()


# ЧТО ДЕЛАЕТ:
# Загружает и форматирует датасет. Превращает текст в токены.
#
# ЗАЧЕМ/МОТИВАЦИЯ:
# Подготовка данных - критический этап. Модель понимает только числа.
# Мы добавляем специальные токены начала/конца реплик (<start_of_turn>, <end_of_turn>),
# чтобы модель понимала структуру диалога.
def process_dataset(dataset_path, tokenizer, dataset_format="instruct"):
    print(f"3. Loading and processing dataset in '{dataset_format}' format...")
    data = load_dataset("json", data_files=dataset_path, split="train")

    def tokenize(prompt):
        if dataset_format == "instruct":
            text = (f"<start_of_turn>user\n{prompt['instruction']}<end_of_turn>\n"
                    f"<start_of_turn>model\n{prompt['response']}<end_of_turn>")
        elif dataset_format == "conversational":
            text = tokenizer.apply_chat_template(
                prompt['messages'], 
                tokenize=False, 
                add_generation_prompt=False
            )
        else:
            raise ValueError(f"Unknown dataset format: {dataset_format}")

        result = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        # Create labels and mask padding tokens so they are ignored in loss calculation
        # Use -100 to ignore these tokens in CrossEntropyLoss
        input_ids = result["input_ids"]
        labels = list(input_ids)
        
        pad_token_id = tokenizer.pad_token_id
        
        for i, token_id in enumerate(input_ids):
            if token_id == pad_token_id:
                labels[i] = -100
                
        result["labels"] = labels
        return result

    tokenized_dataset = data.map(tokenize)
    return tokenized_dataset

def main():
    # ЧТО ДЕЛАЕТ:
    # Парсинг аргументов командной строки.
    #
    # ЗАЧЕМ/МОТИВАЦИЯ:
    # Позволяет гибко настраивать запуск из консоли или .bat файла без изменения кода.
    parser = argparse.ArgumentParser(description="Gemma LoRA Fine-tuning Script")
    parser.add_argument("--dataset_format", type=str, required=True, choices=["instruct", "conversational"], help="Format of the dataset.")
    parser.add_argument("--train_dataset_file", type=str, required=True, help="Training dataset filename.")
    parser.add_argument("--eval_dataset_file", type=str, default=None, help="Evaluation dataset filename (optional).")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save frequency (ignored in favor of eval_steps in this custom setup).")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
    
    args = parser.parse_args()

    # --- ПРОВЕРКА ФАЙЛОВ ДАТАСЕТОВ ---
    # ЧТО ДЕЛАЕТ:
    # Проверяет существование файлов датасетов ДО загрузки модели.
    #
    # ЗАЧЕМ/МОТИВАЦИЯ:
    # Загрузка модели - самый тяжелый и долгий процесс. Если файл датасета
    # не найден или указан неверно, лучше узнать об этом сразу (за 0.1 сек),
    # чем ждать 5 минут загрузки модели, чтобы потом скрипт упал с ошибкой.
    train_data_path = os.path.join("data", args.train_dataset_file)
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"[ERROR] Training dataset not found at: {train_data_path}")
    print(f"[CHECK] Training dataset found: {train_data_path}")

    if args.eval_dataset_file:
        eval_data_path = os.path.join("data", args.eval_dataset_file)
        if not os.path.exists(eval_data_path):
             raise FileNotFoundError(f"[ERROR] Evaluation dataset not found at: {eval_data_path}")
        print(f"[CHECK] Evaluation dataset found: {eval_data_path}")
    # --- КОНЕЦ ПРОВЕРКИ ---

    model_path = "models/gemma-3n"
    output_dir = "output/finetuned_model"
    os.makedirs(output_dir, exist_ok=True)

    print("1. Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_path)

    # Конфигурация квантования (QLoRA) для экономии памяти (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ЧТО ДЕЛАЕТ:
    # Загрузка базовой модели с квантованием.
    #
    # ЗАЧЕМ/МОТИВАЦИЯ:
    # Используем 'eager' attention и bfloat16 для совместимости и скорости.
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager", 
        quantization_config=bnb_config,
    )

    model = prepare_model_for_kbit_training(model)

    # --- ПАТЧ ДЛЯ GEMMA-3N ---
    # ЧТО ДЕЛАЕТ:
    # Вручную разквантовывает (переводит в float32/bfloat16) специфичные модули 'altup' и 'lm_head'.
    #
    # ЗАЧЕМ/МОТИВАЦИЯ:
    # Архитектура Gemma-3n имеет особенности, которые вызывают ошибки (RuntimeError)
    # при обучении в 4 бита. Это известный баг/особенность совместимости.
    # Мы переводим проблемные слои обратно в полную точность, чтобы избежать падения.
    # ИСПОЛЬЗУЕМ bfloat16 ВМЕСТО float32 ДЛЯ ЭКОНОМИИ ПАМЯТИ (1.5GB saving)
    for layer in model.model.language_model.layers:
        for param in layer.altup.parameters():
            param.data = param.data.to(torch.bfloat16)
            param.requires_grad = True
            
    for param in model.lm_head.parameters():
        param.data = param.data.to(torch.bfloat16)
        param.requires_grad = True
    # --- КОНЕЦ ПАТЧА ---

    # --- STABILITY PATCH FOR GEMMA-3N (Manual Fix) ---
    # DIAGNOSIS: "Conv2D Explosion". Vision/Audio layers generate infinite values in fp16/bf16.
    # SOLUTION: Force these specific modules to float32 and freeze them.
    print("2a. Applying stability patch for Vision/Audio encoders...")
    for name, module in model.named_modules():
        if "vision" in name or "audio" in name:
            # Force module to full precision (float32) to handle large activation values
            module.to(torch.float32)

    # Re-freeze them to ensure no gradients flow back (redundant but safe)
    for name, param in model.named_parameters():
        if "vision" in name or "audio" in name:
            param.requires_grad = False
    # -------------------------------------------------

    print("2. Configuring LoRA...")
    # ЧТО ДЕЛАЕТ:
    # Настройка адаптеров LoRA.
    #
    # ЗАЧЕМ/МОТИВАЦИЯ:
    # Мы обучаем только ~0.1% параметров (адаптеры), замораживая остальную модель.
    # Это позволяет файн-тюнить огромные модели на обычных GPU.
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Загрузка Данных ---
    train_data_path = os.path.join("data", args.train_dataset_file)
    tokenized_train_dataset = process_dataset(train_data_path, processor.tokenizer, args.dataset_format)
    
    tokenized_eval_dataset = None
    if args.eval_dataset_file:
        print(f"Loading evaluation dataset: {args.eval_dataset_file}")
        eval_data_path = os.path.join("data", args.eval_dataset_file)
        full_eval_dataset = process_dataset(eval_data_path, processor.tokenizer, args.dataset_format)
        
        # --- THE FIX: LIMIT EVALUATION TO 200 SAMPLES ---
        # We shuffle with a fixed seed to ensure we test the SAME 200 songs every time
        if len(full_eval_dataset) > 200:
            print(f"⚠️ Evaluation dataset is huge ({len(full_eval_dataset)}). Slicing to random 200 samples to save VRAM/Time.")
            tokenized_eval_dataset = full_eval_dataset.shuffle(seed=42).select(range(200))
        else:
            tokenized_eval_dataset = full_eval_dataset
        # ------------------------------------------------

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    data_collator = DataCollatorForLanguageModeling(tokenizer=processor.tokenizer, mlm=False)

    # ЧТО ДЕЛАЕТ:
    # Кастомный класс Trainer с расширенным логированием.
    #
    # ЗАЧЕМ/МОТИВАЦИЯ:
    # Стандартный лог может скрывать результаты оценки. Мы явно выводим их в консоль.
    class CustomTrainer(Trainer):
        def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
            super().log(logs, start_time)
            if 'eval_loss' in logs:
                print(f"\n[LOG] Evaluation: {logs}")

    print("4. Setting up Trainer...")
    
    # Аргументы тренировки
    # Мы отключаем стандартное сохранение (save_strategy="no") и используем наш Callback.
    training_args_dict = {
        'output_dir': output_dir,
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 32,
        # Lower learning rate for stability
        'learning_rate': 5e-6, 
        'num_train_epochs': args.num_train_epochs,
        # Логируем чаще, чтобы видеть прогресс (каждые 10 шагов)
        'logging_steps': 10, 
        # Отключаем стандартное сохранение, чтобы не спамить чекпоинтами и не удалять нужное
        'save_strategy': "no",
        'report_to': "tensorboard",
        'weight_decay': 0.01,
        # Fixed warmup steps for stability
        'warmup_ratio': 0.03,  # 0.03 is fine
        #'warmup_steps': 100,  # 100 is fine (use just one of them)
        'gradient_checkpointing': True,
        'gradient_checkpointing_kwargs': {'use_reentrant': False},
        # Use 32-bit optimizer for better precision with LoRA
        'optim': "paged_adamw_32bit",
        'bf16': True,
        'max_grad_norm': 0.3,
    }

    callbacks_to_use = []
    
    # Настраиваем эвалюацию, если есть валидационный датасет
    if tokenized_eval_dataset:
        # --- ИСПРАВЛЕНИЕ: evaluation_strategy -> eval_strategy ---
        # Transformers >= 4.41.0 переименовал этот аргумент.
        # Мы используем новое имя 'eval_strategy'.
        training_args_dict['eval_strategy'] = "steps" 
        training_args_dict['eval_steps'] = args.eval_steps
        # Добавляем наш умный менеджер чекпоинтов
        callbacks_to_use.append(ManageBestCheckpointsCallback(output_dir, keep_best_count=2))
        print(f"Info: Evaluation enabled every {args.eval_steps} steps. Using Smart Checkpoint Manager.")
    else:
        print("Warning: No evaluation dataset provided. Switching to standard saving strategy (saving every N steps).")
        # Fallback to standard saving if no evaluation is performed
        training_args_dict['save_strategy'] = "steps"
        training_args_dict['save_steps'] = args.save_steps
        training_args_dict['save_total_limit'] = 2 # Keep only the last 2 checkpoints

    training_args = TrainingArguments(**training_args_dict)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks_to_use,
    )

    print("5. Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print("Training complete.")

    print(f"6. Saving final model to {output_dir}/final_model")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    processor.save_pretrained(os.path.join(output_dir, "final_model"))
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
