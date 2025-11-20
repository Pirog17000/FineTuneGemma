#!/usr/bin/env python3
"""
Convert fine-tuned Gemma model to GGUF format
"""

import os
import subprocess
import sys
from pathlib import Path
import logging
import argparse # Import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_llama_cpp():
    """Check if llama.cpp tools are available"""
    try:
        import llama_cpp
        logger.info("llama_cpp available")
        return True
    except ImportError:
        logger.warning("llama_cpp not available, trying system tools")
        return False

def merge_lora_adapters(base_model_path, lora_path, output_path):
    """Merge LoRA adapters with base model"""
    logger.info("Merging LoRA adapters with base model...")

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Try to import specific Gemma3n class if available (to ensure custom layers are loaded)
        try:
            from transformers import Gemma3nForConditionalGeneration
            ModelClass = Gemma3nForConditionalGeneration
            logger.info("Using Gemma3nForConditionalGeneration class")
        except ImportError:
            ModelClass = AutoModelForCausalLM
            logger.info("Using AutoModelForCausalLM class")

        # Load base model and tokenizer
        logger.info(f"Loading base model from {base_model_path}")
        base_model = ModelClass.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # Load LoRA model
        logger.info(f"Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(base_model, lora_path)

        # Merge and unload
        logger.info("Merging adapter into the base model...")
        merged_model = model.merge_and_unload()
        
        # --- DEBUG: Check for potential missing specific tensors ---
        # Gemma 3n might have 'per_layer_token_embd'. 
        # If it's missing from state_dict, we might need to ensure it's preserved.
        keys = set(merged_model.state_dict().keys())
        if "per_layer_token_embd.weight" in keys:
            logger.info("Confirmed: 'per_layer_token_embd.weight' is present in merged model.")
        else:
            logger.warning("'per_layer_token_embd.weight' NOT found in merged model keys.")
            # Check if it was in base model but lost?
            # If we used the correct class, it should be there.
            pass
            
        # COMPATIBILITY HACK: 
        # If this is Gemma-3n, and we want to force compatibility or fix metadata
        # We assume the model structure is correct now that we used the right class.
        # -----------------------------------------------------------

        # CAST TO FLOAT16 to avoid size issues and ensure compatibility
        # This fixes the "Array length must be >= 0" overflow error by reducing model size
        logger.info("Casting merged model to float16 to prevent size overflows...")
        import torch
        merged_model = merged_model.to(dtype=torch.float16)

        # Save merged model and tokenizer
        logger.info(f"Saving merged model and tokenizer to {output_path}")
        
        # WINDOWS FIX: safetensors has a bug on Windows with >2GB tensors (Integer Overflow).
        # We force safe_serialization=False to use standard PyTorch .bin files which work fine.
        # llama.cpp supports .bin files correctly.
        merged_model.save_pretrained(
            output_path, 
            max_shard_size="2GB", 
            safe_serialization=False
        )
        tokenizer.save_pretrained(output_path)
        logger.info(f"Successfully saved merged model and tokenizer.")

        return output_path

    except Exception as e:
        logger.error(f"Failed to merge LoRA adapters: {e}")
        return None

def convert_with_llama_cpp(model_path, output_path, quant_type="f16"):
    """Convert model to GGUF using the convert.py script from llama.cpp repo."""
    logger.info(f"Converting to GGUF using llama.cpp/convert_hf_to_gguf.py with quantization: {quant_type}...")

    llama_cpp_dir = Path("llama.cpp")
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        # Fallback for the newer name
        convert_script = llama_cpp_dir / "convert.py"
        if not convert_script.exists():
            logger.error(f"Conversion script not found in {llama_cpp_dir}")
            logger.info("Please run 'prepare_venv.bat' to clone/update the llama.cpp repository.")
            return False

    command = [
        sys.executable,  # Use the current python interpreter
        str(convert_script),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", quant_type
    ]

    logger.info(f"Running command: {' '.join(command)}")

    try:
        # Use Popen to capture and print output in real-time
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8'
        )
        
        # Read and print output line by line
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
        
        process.stdout.close()
        return_code = process.wait()

        if return_code == 0:
            logger.info(f"GGUF conversion completed successfully: {output_path}")
            return True
        else:
            logger.error(f"GGUF conversion script failed with return code {return_code}")
            return False

    except Exception as e:
        logger.error(f"An exception occurred during GGUF conversion: {e}")
        return False

def convert_with_gguf_tools(model_path, output_path):
    """Fallback conversion using gguf tools"""
    logger.info("Attempting conversion with gguf tools...")

    try:
        # Try using the gguf package
        import gguf

        # This is a simplified approach - gguf conversion can be complex
        # In practice, you might need to use llama.cpp's convert.py script
        logger.warning("Direct gguf conversion not implemented. Please use llama.cpp convert.py manually")
        return False

    except ImportError:
        logger.error("gguf tools not available")
        return False

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and convert to GGUF.")
    parser.add_argument(
        "--quant-type", 
        type=str, 
        default="f16", 
        help="The quantization type to use for GGUF conversion (e.g., f16, q8_0, q4_k_m)."
    )
    args = parser.parse_args()

    base_model_path = "models/gemma-3n"
    lora_path = "output/finetuned_model"
    merged_path = "output/merged_model"
    gguf_output = "output/finetuned_gemma.gguf"

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Check if LoRA adapters exist
    if not os.path.exists(lora_path):
        logger.error(f"LoRA adapters path not found: {lora_path}")
        sys.exit(1)
        
    # Check if adapter_config.json is in the root of lora_path
    if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        logger.info(f"adapter_config.json not found in root of {lora_path}. Searching subfolders...")
        
        # Search for subfolders containing adapter_config.json
        candidates = []
        for entry in os.scandir(lora_path):
            if entry.is_dir() and os.path.exists(os.path.join(entry.path, "adapter_config.json")):
                candidates.append(entry.path)
        
        if not candidates:
            logger.error(f"No subfolders with adapter_config.json found in {lora_path}")
            sys.exit(1)
            
        # Prioritize 'final_model'
        final_model_path = os.path.join(lora_path, "final_model")
        if final_model_path in candidates:
            logger.info(f"Found 'final_model', using it.")
            lora_path = final_model_path
        else:
            # Otherwise, sort by checkpoint number (assuming 'checkpoint-N')
            def get_step(path):
                try:
                    return int(path.split("checkpoint-")[-1])
                except ValueError:
                    return -1
            
            candidates.sort(key=get_step, reverse=True)
            best_candidate = candidates[0]
            logger.info(f"Using latest checkpoint: {best_candidate}")
            lora_path = best_candidate

    # Merge LoRA adapters
    merged_model_path = merge_lora_adapters(base_model_path, lora_path, merged_path)
    if not merged_model_path:
        logger.error("Failed to merge LoRA adapters")
        sys.exit(1)

    # Convert to GGUF
    success = convert_with_llama_cpp(merged_model_path, gguf_output, args.quant_type)

    if success:
        logger.info("GGUF conversion completed successfully!")
        file_size = os.path.getsize(gguf_output) / (1024 * 1024)  # MB
        logger.info(f"Final model size: {file_size:.2f} MB")
    else:
        logger.error("GGUF conversion failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
