import sys
import ctypes
import os
from llama_cpp import llama_model_quantize, llama_model_quantize_params

# Enum values from llama.h / llama_cpp.py
# Maps CLI string to llama_ftype enum
FTYPE_MAP = {
    "q8_0": 7,    # LLAMA_FTYPE_MOSTLY_Q8_0
    "q6_k": 18,   # LLAMA_FTYPE_MOSTLY_Q6_K
    "q5_k_m": 17, # LLAMA_FTYPE_MOSTLY_Q5_K_M
    "q4_k_m": 15, # LLAMA_FTYPE_MOSTLY_Q4_K_M
    "q3_k_m": 12, # LLAMA_FTYPE_MOSTLY_Q3_K_M
}

GGML_TYPE_COUNT = 30

def main():
    if len(sys.argv) != 4:
        print("Usage: python quantize_model.py <input_gguf> <output_gguf> <quant_type>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    q_type = sys.argv[3].lower()

    if q_type not in FTYPE_MAP:
        print(f"Error: Unknown quantization type '{q_type}'")
        print("Available types:", ", ".join(FTYPE_MAP.keys()))
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    ftype = FTYPE_MAP[q_type]
    
    # Create params
    params = llama_model_quantize_params()
    params.nthread = 0 # use all threads (or 0 for default)
    params.ftype = ftype
    params.output_tensor_type = GGML_TYPE_COUNT
    params.token_embedding_type = GGML_TYPE_COUNT
    params.allow_requantize = False
    params.quantize_output_tensor = True
    params.only_copy = False
    params.pure = False
    params.keep_split = False
    
    print(f"Quantizing {input_path} to {output_path} with type {q_type} (ftype={ftype})...")
    
    # Ensure paths are bytes
    fname_inp = input_path.encode('utf-8')
    fname_out = output_path.encode('utf-8')
    
    result = llama_model_quantize(
        fname_inp,
        fname_out,
        ctypes.byref(params)
    )

    if result == 0:
        print(f"Success! Quantized model saved to {output_path}")
    else:
        print(f"Quantization failed with error code {result}")
        sys.exit(1)

if __name__ == "__main__":
    main()

