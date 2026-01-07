#!/usr/bin/env python3
"""Tokenizer compression ratio experiments."""

import random
import time
from pathlib import Path
import json


def sample_documents(file_path: Path, num_samples: int = 10, lines_per_doc: int = 10) -> list[str]:
    """
    Sample documents from a file.

    Args:
        file_path: Path to the input file
        num_samples: Number of documents to sample
        lines_per_doc: Number of lines per document

    Returns:
        List of sampled document strings
    """
    # Read all lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Sample random starting positions
    max_start = len(lines) - lines_per_doc
    if max_start < 0:
        # File is too small, just return what we have
        return [''.join(lines)]

    documents = []
    start_positions = random.sample(range(max_start), min(num_samples, max_start))

    for start_pos in start_positions:
        doc_lines = lines[start_pos:start_pos + lines_per_doc]
        document = ''.join(doc_lines)
        documents.append(document)

    return documents


def calculate_compression_ratio(text: str, token_ids: list[int]) -> float:
    """
    Calculate compression ratio as bytes/token.

    Args:
        text: Original text
        token_ids: Encoded token IDs

    Returns:
        Compression ratio (bytes per token)
    """
    num_bytes = len(text.encode('utf-8'))
    num_tokens = len(token_ids)

    if num_tokens == 0:
        return 0.0

    return num_bytes / num_tokens


def main():
    print("=" * 80)
    print("Tokenizer Compression Ratio Experiments")
    print("=" * 80)

    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    output_dir = script_dir / 'output'

    # Check if trained tokenizers exist
    tinystories_vocab = output_dir / 'tinystories_vocab.json'
    tinystories_merges = output_dir / 'tinystories_merges.txt'
    openwebtext_vocab = output_dir / 'openwebtext_vocab.json'
    openwebtext_merges = output_dir / 'openwebtext_merges.txt'

    tinystories_available = tinystories_vocab.exists() and tinystories_merges.exists()
    openwebtext_available = openwebtext_vocab.exists() and openwebtext_merges.exists()

    if not tinystories_available and not openwebtext_available:
        print("ERROR: No trained tokenizers found!")
        print(f"\nExpected TinyStories tokenizer at:")
        print(f"  {tinystories_vocab}")
        print(f"  {tinystories_merges}")
        print(f"\nExpected OpenWebText tokenizer at:")
        print(f"  {openwebtext_vocab}")
        print(f"  {openwebtext_merges}")
        print("\nPlease train at least one tokenizer first.")
        return 1

    # Import tokenizer
    from cs336_basics.tokenizer import Tokenizer

    print("\n" + "=" * 80)
    print("Loading Tokenizers")
    print("=" * 80)

    tokenizers = {}

    # Load TinyStories tokenizer
    if tinystories_available:
        print(f"\nLoading TinyStories tokenizer...")
        tinystories_tokenizer = Tokenizer.from_files(
            str(tinystories_vocab),
            str(tinystories_merges),
            special_tokens=["<|endoftext|>"]
        )
        print(f"  Vocab size: {len(tinystories_tokenizer.vocab)}")
        tokenizers['tinystories'] = tinystories_tokenizer

    # Load OpenWebText tokenizer
    if openwebtext_available:
        print(f"\nLoading OpenWebText tokenizer...")
        openwebtext_tokenizer = Tokenizer.from_files(
            str(openwebtext_vocab),
            str(openwebtext_merges),
            special_tokens=["<|endoftext|>"]
        )
        print(f"  Vocab size: {len(openwebtext_tokenizer.vocab)}")
        tokenizers['openwebtext'] = openwebtext_tokenizer

    # Sample documents
    print("\n" + "=" * 80)
    print("Sampling Documents")
    print("=" * 80)

    datasets = {}

    tinystories_file = data_dir / 'TinyStoriesV2-GPT4-train.txt'
    if tinystories_file.exists():
        print(f"\nSampling from TinyStories...")
        tinystories_docs = sample_documents(tinystories_file, num_samples=10, lines_per_doc=10)
        print(f"  Sampled {len(tinystories_docs)} documents")
        datasets['tinystories'] = tinystories_docs
    else:
        print(f"  TinyStories file not found at {tinystories_file}")

    openwebtext_file = data_dir / 'openwebtext' / 'train.txt'
    if openwebtext_file.exists():
        print(f"\nSampling from OpenWebText...")
        openwebtext_docs = sample_documents(openwebtext_file, num_samples=10, lines_per_doc=10)
        print(f"  Sampled {len(openwebtext_docs)} documents")
        datasets['openwebtext'] = openwebtext_docs
    else:
        print(f"  OpenWebText file not found at {openwebtext_file}")

    if not datasets:
        print("\nERROR: No dataset files found!")
        return 1

    # Analyze compression ratios
    print("\n" + "=" * 80)
    print("Compression Ratio Analysis")
    print("=" * 80)

    results = {}

    # Test all combinations of tokenizers and datasets
    for tokenizer_name, tokenizer in tokenizers.items():
        for dataset_name, docs in datasets.items():
            print(f"\n--- {tokenizer_name.title()} Tokenizer on {dataset_name.title()} Dataset ---")

            ratios = []
            throughputs = []
            for i, doc in enumerate(docs, 1):
                num_bytes = len(doc.encode('utf-8'))

                # Measure encoding time
                start_time = time.time()
                token_ids = tokenizer.encode(doc)
                encoding_time = time.time() - start_time

                ratio = calculate_compression_ratio(doc, token_ids)
                throughput = num_bytes / encoding_time if encoding_time > 0 else 0

                ratios.append(ratio)
                throughputs.append(throughput)

                num_tokens = len(token_ids)
                print(f"  Doc {i:2d}: {num_bytes:6d} bytes, {num_tokens:5d} tokens, "
                      f"ratio: {ratio:.3f} bytes/token, throughput: {throughput/1024/1024:.2f} MB/s")

            avg_ratio = sum(ratios) / len(ratios)
            avg_throughput = sum(throughputs) / len(throughputs)
            print(f"\n  Average compression ratio: {avg_ratio:.3f} bytes/token")
            print(f"  Average throughput: {avg_throughput/1024/1024:.2f} MB/s")

            # Store results
            key = f"{tokenizer_name}_on_{dataset_name}"
            results[key] = {
                "tokenizer": tokenizer_name,
                "dataset": dataset_name,
                "vocab_size": len(tokenizer.vocab),
                "num_samples": len(docs),
                "compression_ratios": ratios,
                "average_ratio": avg_ratio,
                "throughputs_bytes_per_sec": throughputs,
                "average_throughput_bytes_per_sec": avg_throughput,
                "average_throughput_mb_per_sec": avg_throughput / 1024 / 1024
            }

    # Save results
    results_file = output_dir / 'compression_ratios.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for key, data in results.items():
        tokenizer_name = data['tokenizer']
        dataset_name = data['dataset']
        vocab_size = data['vocab_size']
        avg_ratio = data['average_ratio']
        avg_throughput_mb = data['average_throughput_mb_per_sec']

        print(f"\n{tokenizer_name.title()} tokenizer (vocab={vocab_size}) on {dataset_name.title()}:")
        print(f"  Compression ratio: {avg_ratio:.3f} bytes/token (lower is better)")
        print(f"  Throughput: {avg_throughput_mb:.2f} MB/s (higher is better)")

    return 0


if __name__ == "__main__":
    exit(main())
