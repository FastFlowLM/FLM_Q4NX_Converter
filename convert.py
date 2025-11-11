from q4nx import create_converter

def convert_gguf_to_q4nx(gguf_path: str, q4nx_path: str):
    model = create_converter(gguf_path)
    model.convert(q4nx_path)


if __name__ == "__main__":
    convert_gguf_to_q4nx("gguf_files/Qwen3-VL-4B-Instruct-Q4_1.gguf", "q4nx_files/Qwen3-VL-4B-Instruct-Q4_1.q4nx")
