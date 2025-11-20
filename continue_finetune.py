#!/usr/bin/env python3
"""
Gemma LoRA Fine-tuning Script using Hugging Face Transformers.
Allows for continuing training from an existing adapter.
"""
import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(base_model_path, adapter_path=None):
    """Loads the base model and tokenizer, and applies an existing or new LoRA adapter."""
    logger.info(f"1. Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    if adapter_path:
        logger.info(f"2. Resuming training from existing adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        logger.info("2. Initializing new LoRA adapter for training.")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model, tokenizer

def process_dataset(dataset_path, tokenizer, dataset_format="instruct"):
    """Loads and processes the dataset based on the specified format."""
    logger.info(f"3. Loading and processing dataset in '{dataset_format}' format...")
    data = load_dataset("json", data_files=dataset_path, split="train")

    def tokenize(prompt):
        # Format the prompt based on the dataset format
        if dataset_format == "instruct":
            text = (f"<start_of_turn>user\n{prompt['instruction']}<end_of_turn>\n"
                    f"<start_of_turn>model\n{prompt['response']}<end_of_turn>")
        elif dataset_format == "conversational":
            # The tokenizer will apply the template for a list of messages
            text = tokenizer.apply_chat_template(
                prompt['messages'], 
                tokenize=False, 
                add_generation_prompt=False
            )
        else:
            raise ValueError(f"Unknown dataset format: {dataset_format}")

        # Tokenize the formatted text
        result = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_dataset = data.map(tokenize)
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser(description="Continue fine-tuning a Gemma model with LoRA.")
    parser.add_argument("--base_model_path", type=str, default="models/gemma-3n", help="Path to the base Gemma model.")
    parser.add_argument("--dataset_path", type=str, default="data/conversation_training.jsonl", help="Path to the training dataset.")
    parser.add_argument("--dataset_format", type=str, default="instruct", choices=["instruct", "conversational"], help="Format of the dataset.")
    parser.add_argument("--resume_from_adapter", type=str, required=True, help="Path to a LoRA adapter to continue training from.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the new adapter.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of additional training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="The initial learning rate.")

    args = parser.parse_args()

    # --- Main script ---
    model, tokenizer = setup_model_and_tokenizer(args.base_model_path, args.resume_from_adapter)
    tokenized_dataset = process_dataset(args.dataset_path, tokenizer, args.dataset_format)

    logger.info("4. Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_strategy="epoch",  # Save a checkpoint at the end of each epoch
        report_to="none",
        # Regularization
        weight_decay=0.01,
        warmup_ratio=0.03,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("5. Starting training...")
    trainer.train()

    logger.info("Training complete.")
    logger.info(f"6. Saving final model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    logger.info("Model saved successfully.")

if __name__ == "__main__":
    main()
