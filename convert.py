#!./venv/bin/python3
import argparse
import os
from pathlib import Path
from q4nx import create_converter


def convert_gguf_to_q4nx(gguf_path: str, q4nx_path: str):
    model = create_converter(gguf_path)
    model.convert(q4nx_path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert GGUF model files to Q4NX format (output always named model.q4nx)',
        epilog='Examples:\n'
               '  python convert.py -i model.gguf\n'
               '  python convert.py -i model.gguf -o output_folder\n'
               '  python convert.py model.gguf output_folder\n'
               '  python convert.py model.gguf .',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add support for both flag-based and positional arguments
    parser.add_argument('input_file', nargs='?', help='Input GGUF file (positional)')
    parser.add_argument('output_folder', nargs='?', help='Output folder (positional, optional)')
    parser.add_argument('-i', '--input', dest='input_flag', help='Input GGUF file')
    parser.add_argument('-o', '--output', dest='output_flag', help='Output folder (optional, defaults to input file directory)')
    
    args = parser.parse_args()
    
    # Determine input file (prioritize flag, then positional)
    input_path = args.input_flag or args.input_file
    
    if not input_path:
        parser.error('Input file is required. Use -i <file> or provide as positional argument.')
    
    # Determine output folder (prioritize flag, then positional)
    output_folder = args.output_flag or args.output_folder
    
    # Check if input file exists
    if not os.path.exists(input_path):
        parser.error(f'Input file does not exist: {input_path}')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_folder)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Converting {input_path} to {output_folder}...")
    convert_gguf_to_q4nx(input_path, output_folder)
    print(f"[INFO] Conversion complete! Output saved to {output_folder}")


if __name__ == "__main__":
    main()
