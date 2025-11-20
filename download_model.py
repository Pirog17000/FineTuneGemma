#!/usr/bin/env python3
"""
Download Gemma model from HuggingFace
"""

import os
from huggingface_hub import snapshot_download

def download_gemma_model():
    """Download the Gemma model from HuggingFace Hub"""
    model_name = "google/gemma-3n-E4B-it"
    model_dir = "models/gemma-3n"

    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        print(f"Downloading Gemma model '{model_name}' from HuggingFace Hub...")
        os.makedirs(model_dir, exist_ok=True)

        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                local_dir_use_symlinks=False  # Use False on Windows
            )
            print(f"Model downloaded successfully to {model_dir}")
        except Exception as e:
            print(f"Failed to download model: {e}")
            # Clean up partial download
            if os.path.exists(model_dir):
                import shutil
                shutil.rmtree(model_dir)
            exit(1)
    else:
        print("Model directory already exists and is not empty. Skipping download.")

if __name__ == "__main__":
    download_gemma_model()
