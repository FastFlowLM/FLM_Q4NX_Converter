# FLM Q4NX Converter

A utility for converting GGUF model files into the Q4NX format. This tool supports converting both language and vision model weights.

## Supported Models
Based on the configuration, the converter supports several model architectures, including:
- Gemma 3
- GPT-OSS
- LFM 2
- LLaMA
- Phi-4
- Qwen 2 / Qwen 2.5
- Qwen 2 VL
- Qwen 3
- Qwen 3 VL

## Setup

The project includes a setup script that automatically creates a Python virtual environment, installs required dependencies (like `torch`, `amd-quark`, `transformers`), and sets up the `gguf` package from `llama.cpp`.

1. Make the setup script executable (if it isn't already) and run it:
   ```bash
   chmod +x setup_venv.sh
   ./setup_venv.sh
   ```

2. Activate the environment using the provided activation script. **Note:** Always use `activate.sh` instead of directly sourcing the venv, as it also sets up necessary environment variables like `HF_HOME` and CUDA paths.
   ```bash
   source activate.sh
   ```

## Usage

The main entry point for conversion is `convert.py`. 

### Basic Syntax
```bash
python convert.py [input_file] [output_folder] [-t TYPE]
```

### Arguments
- `input_file` or `-i`, `--input`: Path to the input `.gguf` file.
- `output_folder` or `-o`, `--output`: Path to the output folder. The output file will always be named `model.q4nx` (or `vision_weights.q4nx` depending on the type) inside this folder. Defaults to the input file's directory.
- `-t`, `--type`: Type of weights to convert. Choices are `language` (default) or `vision`.

### Examples

**1. Convert a language model (positional arguments):**
```bash
python convert.py model.gguf output_folder
```

**2. Convert a language model (flag arguments):**
```bash
python convert.py -i unsloth_gpt-oss-20b-Q4_0.gguf -o unsloth-gotoss20b-q40
```

**3. Convert a vision model:**
```bash
python convert.py -i qwen3vl-4b-mmproj-BF16.gguf -o unsloth-qwen3vl-vision -t vision
```

## Project Structure
- `convert.py`: The main CLI script for running conversions.
- `setup_venv.sh`: Initializes the Python virtual environment and installs dependencies.
- `activate.sh`: Activates the virtual environment and sets up environment variables.
- `q4nx/`: Core package containing the conversion logic, gguf tensor parsing, and model-specific implementations.
- `configs/`: JSON configuration files for supported model architectures.


## Known Issues
- The converter currently only supports either Q4_0 or Q4_1 quantization format based on the setting in config files for each model.
- For GPT-OSS:20B models, the converter currently uses the original `model.embed_tokens.weight` from the safetensors from OpenAI due to issues with Q4_1 quantization (from our experience, Q4_1 quantization messes up the quantization of the embedding layer for this model). 
  - **Workaround:** Place the [`model-00001-of-00001.safetensors`](https://huggingface.co/openai/gpt-oss-20b/tree/main) file in the root directory of this project before running the conversion for GPT-OSS. If the file is not found, the converter will print a warning and skip replacing the embedding weights. 
