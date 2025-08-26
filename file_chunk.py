import os
import json
import glob
import argparse
import tiktoken

from hashlib import md5
def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()
def chunk_documents(
    docs,
    model_name="cl100k_base",
    max_token_size=512,
    overlap_token_size=64,
):
    ENCODER = tiktoken.get_encoding(model_name)
    tokens_list = ENCODER.encode_batch(docs, num_threads=16)

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token_ids = []
        lengths = []

        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk = tokens[start : start + max_token_size]
            chunk_token_ids.append(chunk)
            lengths.append(len(chunk))

        chunk_texts = ENCODER.decode_batch(chunk_token_ids)

        for i, text in enumerate(chunk_texts):
            results.append({
                "hash_code": compute_mdhash_id(text),
                "text": text.strip().replace("\n", ""),
            })

    return results

def chunk_files(input_directory, output_directory, max_token_size, overlap_token_size):
    os.makedirs(output_directory, exist_ok=True)

    print(f"Chunking files in {input_directory} and saving to {output_directory}")

    json_files = glob.glob(os.path.join(input_directory, "*.json"))
    print(f"Found {len(json_files)} JSON files.")

    for file_path in json_files:
        filename = os.path.basename(file_path)
        prefix = filename.partition('_unique_contexts.json')[0]
        output_path = os.path.join(output_directory, prefix)
        os.makedirs(output_path, exist_ok=True)
        output_filename = os.path.join(output_path, "chunk.json")

        print(f"Processing file: {filename} -> {output_filename}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = chunk_documents(
            data,
            max_token_size=max_token_size,
            overlap_token_size=overlap_token_size,
        )
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print("All files have been processed.")

if __name__ == "__main__":
    max_token_size=1024
    overlap_token_size=128

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, default="datasets/unique_contexts")
    parser.add_argument(
        "-o", "--output_dir", type=str, default="datatest"
    )
    args = parser.parse_args()

    chunk_files(args.input_dir, args.output_dir, max_token_size, overlap_token_size)



