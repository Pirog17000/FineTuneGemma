# python rescue.py --dataset_format instruct --train_dataset_file your_dataset.json --resume_checkpoint_path output/finetuned_model/checkpoint-400

### Extreme case restart
# Why this will work (and not freeze)
#
#    No Evaluation: The script never allocates that extra memory buffer for validation data. You stay strictly in the "Training" VRAM footprint.
#
#    Dirty Resume: It bypasses the error about missing trainer_state.json. It loads the brain (weights) but gives it a fresh cup of coffee (new optimizer).
#
#    Stability: By keeping the batch size 1 and accumulation 16, you stay in the safe zone.


#!/usr/bin/env python3
import os
import argparse
import gc
import inspect
import torch
import shutil
# PATCHES START
import accelerate
import accelerate.optimizer

# Patch unwrap_model
_orig_unwrap = accelerate.Accelerator.unwrap_model
def _patched_unwrap(self, model, *args, **kwargs):
    if 'keep_torch_compile' in kwargs:
        sig = inspect.signature(_orig_unwrap)
        if 'keep_torch_compile' not in sig.parameters:
            kwargs.pop('keep_torch_compile')
    return _orig_unwrap(self, model, *args, **kwargs)
accelerate.Accelerator.unwrap_model = _patched_unwrap

# Patch optimizer
def _patched_optimizer_train(self):
    if hasattr(self.optimizer, "train"): return self.optimizer.train()
    return None
def _patched_optimizer_eval(self):
    if hasattr(self.optimizer, "eval"): return self.optimizer.eval()
    return None
accelerate.optimizer.AcceleratedOptimizer.train = _patched_optimizer_train
accelerate.optimizer.AcceleratedOptimizer.eval = _patched_optimizer_eval
# PATCHES END

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Gemma3nForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import PeftModel, prepare_model_for_kbit_training

def process_dataset(dataset_path, tokenizer, dataset_format="instruct"):
    print(f"3. Loading dataset...")
    data = load_dataset("json", data_files=dataset_path, split="train")

    def tokenize(prompt):
        if dataset_format == "instruct":
            text = (f"<start_of_turn>user\n{prompt['instruction']}<end_of_turn>\n"
                    f"<start_of_turn>model\n{prompt['response']}<end_of_turn>")
        elif dataset_format == "conversational":
            text = tokenizer.apply_chat_template(prompt['messages'], tokenize=False, add_generation_prompt=False)
        else:
            raise ValueError(f"Unknown format")

        result = tokenizer(text, truncation=True, max_length=512, padding="max_length")
        
        input_ids = result["input_ids"]
        labels = list(input_ids)
        pad_token_id = tokenizer.pad_token_id
        for i, token_id in enumerate(input_ids):
            if token_id == pad_token_id:
                labels[i] = -100
        result["labels"] = labels
        return result

    return data.map(tokenize)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_format", type=str, required=True)
    parser.add_argument("--train_dataset_file", type=str, required=True)
    # Point this to your last good checkpoint folder
    parser.add_argument("--resume_checkpoint_path", type=str, required=True) 
    
    args = parser.parse_args()

    model_path = "models/gemma-3n"
    # We save to a NEW folder to avoid corrupting old data
    output_dir = "output/finetuned_model_rescue" 
    os.makedirs(output_dir, exist_ok=True)

    print("1. Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("2. Loading Base Model...")
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager", 
        quantization_config=bnb_config,
    )

    model = prepare_model_for_kbit_training(model)

    # --- DIRTY RESUME: Load Weights manually ---
    print(f"3. LOADING ADAPTER WEIGHTS from {args.resume_checkpoint_path}...")
    # This loads the learned Rank 64 weights onto the base model
    model = PeftModel.from_pretrained(model, args.resume_checkpoint_path, is_trainable=True)
    
    # Re-apply patches because loading PeftModel might reset some flags
    print("3a. Re-applying stability patches...")
    for layer in model.model.language_model.layers:
        for param in layer.altup.parameters():
            param.data = param.data.to(torch.float32)
            param.requires_grad = True     
    for param in model.lm_head.parameters():
        param.data = param.data.to(torch.float32)
        param.requires_grad = True
    for name, module in model.named_modules():
        if "vision" in name or "audio" in name:
            module.to(torch.float32)
    for name, param in model.named_parameters():
        if "vision" in name or "audio" in name:
            param.requires_grad = False
            
    model.print_trainable_parameters()

    train_data_path = os.path.join("data", args.train_dataset_file)
    tokenized_train_dataset = process_dataset(train_data_path, processor.tokenizer, args.dataset_format)
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    data_collator = DataCollatorForLanguageModeling(tokenizer=processor.tokenizer, mlm=False)

    print("4. Setting up Trainer (NO EVALUATION)...")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16, # Keep this high!
        learning_rate=5e-6,             # Low LR to help optimizer reset gently
        num_train_epochs=4,             # Set this to how many MORE epochs you want
        logging_steps=5, 
        
        # STANDARD SAVING - Safer than custom callbacks
        save_strategy="steps",
        save_steps=200,                 # Save every 200 steps
        save_total_limit=3,             # Keep last 3
        
        report_to="tensorboard",
        warmup_steps=50,                # Quick warmup for the new optimizer
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        bf16=True,
        max_grad_norm=0.3,
        
        # DISABLE EVALUATION
        eval_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator,
    )

    print("5. STARTING RESCUE TRAINING...")
    # We do NOT use resume_from_checkpoint=True because the state files are missing.
    # We essentially start a "new" training run, but starting with the smart weights.
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    processor.save_pretrained(os.path.join(output_dir, "final_model"))
    print("Done.")

if __name__ == "__main__":
    main()