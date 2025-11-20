import argparse
import os
from pathlib import Path
from q4nx import create_converter


def convert_gguf_to_q4nx(gguf_path: str, q4nx_path: str):
    model = create_converter(gguf_path)
    model.convert(q4nx_path)


def get_output_path(input_path: str, output_path: str = None) -> str:
    """
    Determine the output path based on input path and optional output path.
    
    Args:
        input_path: Input GGUF file path
        output_path: Optional output file path
        
    Returns:
        Output file path with proper .q4nx extension
    """
    if output_path is None:
        # Use same name as input but change suffix to .q4nx
        input_file = Path(input_path)
        return str(input_file.with_suffix('.q4nx'))
    else:
        # Check if output_path has a suffix
        output_file = Path(output_path)
        if not output_file.suffix:
            # No suffix, add .q4nx
            return str(output_file) + '.q4nx'
        else:
            return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert GGUF model files to Q4NX format',
        epilog='Examples:\n'
               '  python convert.py -i model.gguf\n'
               '  python convert.py -i model.gguf -o output.q4nx\n'
               '  python convert.py model.gguf output.q4nx\n'
               '  python convert.py model.gguf output',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add support for both flag-based and positional arguments
    parser.add_argument('input_file', nargs='?', help='Input GGUF file (positional)')
    parser.add_argument('output_file', nargs='?', help='Output Q4NX file (positional, optional)')
    parser.add_argument('-i', '--input', dest='input_flag', help='Input GGUF file')
    parser.add_argument('-o', '--output', dest='output_flag', help='Output Q4NX file (optional)')
    
    args = parser.parse_args()
    
    # Determine input file (prioritize flag, then positional)
    input_path = args.input_flag or args.input_file
    
    if not input_path:
        parser.error('Input file is required. Use -i <file> or provide as positional argument.')
    
    # Determine output file (prioritize flag, then positional)
    output_path = args.output_flag or args.output_file
    
    # Get the final output path with proper extension
    final_output_path = get_output_path(input_path, output_path)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        parser.error(f'Input file does not exist: {input_path}')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(final_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting {input_path} to {final_output_path}...")
    convert_gguf_to_q4nx(input_path, final_output_path)
    print(f"Conversion complete! Output saved to {final_output_path}")


if __name__ == "__main__":
    main()
