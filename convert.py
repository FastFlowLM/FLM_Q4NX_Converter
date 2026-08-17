#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from q4nx import create_converter, create_hf_converter
from q4nx.model_assets import assemble_model_assets, assemble_model_assets_hf, get_default_flm_version, find_repo_gguf


def is_hf_repo_id(path: str) -> bool:
    """HF hub repo id like 'org/name' (not a local path, not a .gguf)."""
    if not path or path.endswith(".gguf") or os.path.exists(path):
        return False
    if path.startswith(("http://", "https://", "file:")):
        return False
    # org/name with no filesystem separators beyond the single slash
    parts = path.split("/")
    return len(parts) == 2 and all(parts) and "\\" not in path


def is_hf_repo_id(path: str) -> bool:
    """HF hub repo id like 'org/name' (not a local path, not a .gguf)."""
    if not path or path.endswith(".gguf") or os.path.exists(path):
        return False
    if path.startswith(("http://", "https://", "file:")):
        return False
    # org/name with no filesystem separators beyond the single slash
    parts = path.split("/")
    return len(parts) == 2 and all(parts) and "\\" not in path


def is_hf_source(path: str) -> bool:
    if os.path.isdir(path):
        return (
            os.path.exists(os.path.join(path, "model.safetensors"))
            or os.path.exists(os.path.join(path, "model.safetensors.index.json"))
        )
    return is_hf_repo_id(path)


def convert_gguf_to_q4nx(gguf_path: str, q4nx_path: str, override_model_arch:str, weights_type: str = 'language', source_model: str = None, flm_version: str = None, deploy_tag: str = None, deploy_from: str = None, deploy_name: str = None):
    if flm_version is None:
        flm_version = get_default_flm_version()
    # The weight source is always -i: an HF dir / repo id (Darwin-style HF
    # path) or a GGUF. --source-model only supplies tokenizer/config assets and
    # is never treated as a weight source, so it can't trigger a weights
    # download when converting from GGUF.
    #
    # New: -i <hf-repo-id> prefers a quantized GGUF shipped in the repo itself,
    # chosen in a family-preferred order (default q4_1, then q4_0, then q8_0;
    # e.g. LFM prefers q4_0 first, gpt-oss also accepts mxfp4 last). The order
    # is driven by -f when given, otherwise by a best-effort repo-id/filename
    # match. The chosen GGUF is downloaded via the HF cache; if the repo has
    # none, we fall back to the HF-safetensors source path below.
    hf_input = None
    source_file = None
    if is_hf_repo_id(gguf_path):
        repo_id = gguf_path
        found = find_repo_gguf(repo_id, override_model_arch)
        if found is not None:
            gguf_path, source_file = found
            source_model = source_model or repo_id
        else:
            hf_input = repo_id
    elif is_hf_source(gguf_path):
        hf_input = gguf_path

    if hf_input is not None:
        model = create_hf_converter(hf_input, override_model_arch)
        model.convert(q4nx_path=q4nx_path, weights_type=weights_type)
        assemble_model_assets_hf(
            model.hf_source,
            model.q4nx_config,
            q4nx_path,
            source_model=source_model or hf_input,
            flm_version=flm_version,
            source_file=source_file,
        )
    else:
        model = create_converter(gguf_path, override_model_arch)
        model.convert(q4nx_path=q4nx_path, weights_type=weights_type)
        assemble_model_assets(
            model.gguf_reader,
            model.q4nx_config,
            q4nx_path,
            source_model=source_model,
            flm_version=flm_version,
            source_file=source_file,
        )
    if deploy_tag:
        from q4nx.deploy import deploy_model
        deploy_model(
            q4nx_path,
            deploy_tag,
            model.model_arch,
            model_dir_name=deploy_name,
            deploy_from=deploy_from,
        )
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Convert GGUF or HF-safetensors model files to Q4NX format (output always named model.q4nx). '
                    '-i also accepts an HF repo id: the repo is searched for a quantized GGUF, chosen in a '
                    'family-preferred order (default q4_1 / q4_0 / q8_0; e.g. LFM prefers q4_0 first). Pass -f to '
                    'force the family and override the auto-detected source GGUF.',
        epilog='Examples:\n'
               '  python convert.py -i model.gguf\n'
               '  python convert.py -i model.gguf -o output_folder\n'
               '  python convert.py model.gguf output_folder\n'
               '  python convert.py -i Qwen/Qwen3.5-9B -o output_folder     (HF repo: picks a q4_1/q4_0/q8_0 GGUF from it)\n'
               '  python convert.py -i LiquidAI/LFM2-1.2B -o out -f lfm2      (force LFM family -> prefers q4_0 source)\n'
               '  python convert.py -i /path/to/hf_model_dir -o output_folder   (HF safetensors source, e.g. Darwin-36B-Opus)\n'
               '  python convert.py -i vision_model.gguf -o output_folder -t vision',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add support for both flag-based and positional arguments
    parser.add_argument('input_file', nargs='?', help='Input GGUF file (positional)')
    parser.add_argument('output_folder', nargs='?', help='Output folder (positional, optional)')
    parser.add_argument('-i', '--input', dest='input_flag', help='Input GGUF file, or an HF repo id (a quantized GGUF is auto-selected in family-preferred order; use -f to force the family)')
    parser.add_argument('-o', '--output', dest='output_flag', help='Output folder (optional, defaults to input file directory)')
    parser.add_argument('-t', '--type', dest='weights_type', default='language', help='Type of weights to convert (default: language)',
                        choices=['language', 'vision', 'audio'])
    parser.add_argument('-f', '--force', dest='force_model_type', default="", help="Model type. Empty string for automatic recognition from gguf file")
    parser.add_argument('-s', '--source-model', dest='source_model', default=None,
                        help="Source HF/ModelScope model for tokenizer/config assets. A local dir, an HF cache repo name, or a repo id like 'Qwen/Qwen3.5-9B'. If omitted, the GGUF's provenance metadata is followed (local HF cache first, then SDK download).")
    parser.add_argument('--flm-version', dest='flm_version', default=None,
                        help="flm_version to write into the generated config.json (default: detected from `flm --version`)" )
    parser.add_argument('-d', '--deploy', dest='deploy_tag', default=None, metavar='NAME:SIZE',
                        help="Deploy the converted model into flm's models directory and register it under this tag (e.g. 'qwen3.5-claude:9b'). Uses a user-level model_list.json (point FLM_CONFIG_PATH at it to make `flm run` see the tag).")
    parser.add_argument('--deploy-from', dest='deploy_from', default=None, metavar='SOURCE_TAG',
                        help="Official registry entry to copy defaults from (e.g. 'qwen3.5:9b'). Auto-detected from the model architecture if omitted.")
    parser.add_argument('--deploy-name', dest='deploy_name', default=None, metavar='DIR',
                        help="Directory name inside flm's models dir (default: derived from the deploy tag, e.g. Qwen3.5-Claude-9B-NPU2).")
    
    args = parser.parse_args()
    
    # Determine input file (prioritize flag, then positional)
    input_path = args.input_flag or args.input_file
    
    if not input_path:
        parser.error('Input file is required. Use -i <file> or provide as positional argument.')
    
    # Determine output folder (prioritize flag, then positional)
    output_folder = args.output_flag or args.output_folder
    
    # Local paths must exist; HF repo ids are resolved later by the converter.
    if not is_hf_repo_id(input_path) and not os.path.exists(input_path):
        parser.error(f'Input file does not exist: {input_path}')

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_folder)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Converting {input_path} to {output_folder}...")
    convert_gguf_to_q4nx(input_path, output_folder, args.force_model_type, weights_type=args.weights_type, source_model=args.source_model, flm_version=args.flm_version, deploy_tag=args.deploy_tag, deploy_from=args.deploy_from, deploy_name=args.deploy_name)
    print(f"[INFO] Conversion complete! Output saved to {output_folder}")



if __name__ == "__main__":
    # for debug, give the path and ouptut path here by directly set the command line args
    import sys
    main()
