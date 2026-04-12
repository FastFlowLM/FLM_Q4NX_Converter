#!./venv/bin/python3
import argparse
import os
from pathlib import Path
from q4nx import create_converter


def convert_gguf_to_q4nx(gguf_path: str, q4nx_path: str, override_model_arch:str, weights_type: str = 'language'):
    model = create_converter(gguf_path, override_model_arch)
    model.convert(q4nx_path=q4nx_path, weights_type=weights_type)


def main():
    parser = argparse.ArgumentParser(
        description='Convert GGUF model files to Q4NX format (output always named model.q4nx)',
        epilog='Examples:\n'
               '  python convert.py -i model.gguf\n'
               '  python convert.py -i model.gguf -o output_folder\n'
               '  python convert.py model.gguf output_folder\n'
               '  python convert.py model.gguf .\n'
               '  python convert.py -i vision_model.gguf -o output_folder -t vision',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add support for both flag-based and positional arguments
    parser.add_argument('input_file', nargs='?', help='Input GGUF file (positional)')
    parser.add_argument('output_folder', nargs='?', help='Output folder (positional, optional)')
    parser.add_argument('-i', '--input', dest='input_flag', help='Input GGUF file')
    parser.add_argument('-o', '--output', dest='output_flag', help='Output folder (optional, defaults to input file directory)')
    parser.add_argument('-t', '--type', dest='weights_type', default='language', help='Type of weights to convert (default: language)',
                        choices=['language', 'vision', 'audio'])
    parser.add_argument('-f', '--force', dest='force_model_type', default="", help="Model type. Empty string for automatic recognition from gguf file")
    
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
    convert_gguf_to_q4nx(input_path, output_folder, args.force_model_type, weights_type=args.weights_type)
    print(f"[INFO] Conversion complete! Output saved to {output_folder}")



if __name__ == "__main__":
    # for debug, give the path and ouptut path here by directly set the command line args
    import sys
    # sys.argv = ['convert.py', '-i', 'unsloth_gpt-oss-20b-Q4_0.gguf', '-o', 'unsloth-gotoss20b-q40']
            
    # sys.argv = ['convert.py', '-i', 'unsloth_gpt-oss-20b-Q4_1.gguf', '-o', 'unsloth-gotoss20b-q41']            
            
    # import sys
    # # sys.argv = ['convert.py', '-i', 'gemma-3-4b-it-Q4_1.gguf', '-o', 'unsloth-gemma3-q41']    
    # # main()


    # # sys.argv = ['convert.py', '-i', 'gemma3-mmproj-BF16.gguf', '-o', 'unsloth-gemma3-vision', '-t', 'vision']
    # # main()
    
    

    # # sys.argv = ['convert.py', '-i'
    # , 'medgemma3-mmproj-BF16.gguf', '-o', 'unsloth-medgemma3-vision', '-t', 'vision']
    # # main()
    # sys.argv = ['convert.py', '-i', 'Qwen3-VL-4B-Instruct-Q4_1.gguf', '-o', 'unsloth-qwen3vl-4b-q41' ]
    # main()             
    
    # sys.argv = ['convert.py', '-i', 'Qwen3-4B-Q4_1.gguf', '-o', 'unsloth-qwen3-4b-q41' ]
    # main()                 
    # sys.argv = ['convert.py', '-i', 'qwen3vl-4b-mmproj-BF16.gguf', '-o', 'unsloth-qwen3vl-vision', '-t', 'vision']
    # main()        
    
    
    
    #sys.argv = ['convert.py', '-i', 'qwen3_5vl-4b-mmproj-BF16.gguf', '-o', 'unsloth-qwen3_5vl-vision', '-t', 'vision']
     
    #sys.argv = ['convert.py', '-i', 'qwen3_5vl-9bmmproj-BF16.gguf', '-o', 'unsloth-qwen3_5_9bvl-vision', '-t', 'vision'] 
    
    
    #sys.argv = ['convert.py', '-i', 'Qwen3.5-4B-Q4_1.gguf', '-o', 'unsloth-qwen3_5_4bq41'] 
    
    #sys.argv = ['convert.py', '-i', 'Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf', '-o', 'unsloth-qwen3_59b_uncensored', "-f", "qwen3.5-9B"]     
    # sys.argv = ['convert.py', '-i', 'Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf', '-o', 'unsloth-qwen3_59b_uncensored', "-f", "qwen3.5-9B"]     
    
    # sys.argv = ['convert.py', '-i', 'Qwen3.5-9B-Q4_1.gguf', '-o', 'unsloth-qwen3_5_9bq41']     
    
    
    
    #sys.argv = ['convert.py', '-i', 'gemma-4-E2B-it-Q4_1.gguf', '-o', 'unsloth-gemma4-2b-it-q41']    
    
    #sys.argv = ['convert.py', '-i', 'gemma4-2b-mmproj.gguf', '-o', 'unsloth-gemma4-2b-vision', '-t', 'vision']     
    
    sys.argv = ['convert.py', '-i', 'gemma4-2b-mmproj.gguf', '-o', 'unsloth-gemma4-2b-audio', '-t', 'audio']         
    # sys.argv = ['convert.py', '-i', 'debug_gemma4e2b_model.gguf', '-o', 'debug-gemma4-2b-audio', '-t', 'audio', '-f', 'gemma4']           
    main()
