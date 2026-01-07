#!/usr/bin/env python3
"""Train BPE on TinyStories dataset"""

import sys
import os
from pathlib import Path
import time
import json

# Change to script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

from tests.adapters import run_train_bpe


def main():
    print("=" * 70)
    print("BPE Training on TinyStories Dataset")
    print("=" * 70)

    # Configuration
    input_path = Path('data/TinyStoriesV2-GPT4-train.txt')
    # input_path = Path('/home/ubuntu/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt')

    vocab_size = 10000  # Common vocab size for small models
    special_tokens = ["<|endoftext|>"]

    # Check if file exists
    if not input_path.exists():
        print(f"\n✗ Error: Input file not found at {input_path}")
        print("\nPlease ensure the TinyStories dataset is in the correct location.")
        print("Expected path: data/TinyStoriesV2-GPT4-train.txt")
        return 1

    # Show file info
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"\nInput file: {input_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Target vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")

    # Train BPE
    print("\n" + "-" * 70)
    print("Starting BPE training...")
    print("-" * 70)

    start_time = time.time()

    try:
        vocab, merges = run_train_bpe(
            input_path=str(input_path),
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        end_time = time.time()
        training_time = end_time - start_time

        print(f"\n{'='*70}")
        print("Training completed successfully!")
        print(f"{'='*70}")
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Final vocab size: {len(vocab)}")
        print(f"Number of merges: {len(merges)}")
        print(f"Tokens per second: {len(merges)/training_time:.2f}")

        # Save results
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)

        vocab_file = output_dir / 'tinystories_vocab_0105.json'
        merges_file = output_dir / 'tinystories_merges_0105.txt'

        # Import GPT-2 encoding function
        import sys
        sys.path.insert(0, str(script_dir / 'tests'))
        from common import gpt2_bytes_to_unicode

        # Get byte-to-unicode mapping
        byte_encoder = gpt2_bytes_to_unicode()

        # Save vocab in GPT-2 format (convert bytes to GPT-2 unicode strings)
        vocab_serializable = {}
        for token_id, token_bytes in vocab.items():
            # Convert each byte in token_bytes to its GPT-2 unicode representation
            gpt2_token = ''.join(byte_encoder[b] for b in token_bytes)
            vocab_serializable[gpt2_token] = token_id

        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_serializable, f, ensure_ascii=False)
        print(f"\n✓ Vocabulary saved to: {vocab_file}")

        # Save merges in GPT-2 format
        with open(merges_file, 'w', encoding='utf-8') as f:
            for token1, token2 in merges:
                # Convert bytes to GPT-2 unicode strings
                gpt2_token1 = ''.join(byte_encoder[b] for b in token1)
                gpt2_token2 = ''.join(byte_encoder[b] for b in token2)
                f.write(f"{gpt2_token1} {gpt2_token2}\n")
        print(f"✓ Merges saved to: {merges_file}")

        # Show some statistics
        print(f"\n{'='*70}")
        print("Statistics")
        print(f"{'='*70}")

        # First 10 merges
        print("\nFirst 10 merges:")
        for i, (token1, token2) in enumerate(merges[:10], 1):
            try:
                # Try to decode as ASCII for readability
                t1_str = token1.decode('ascii', errors='replace')
                t2_str = token2.decode('ascii', errors='replace')
                merged_str = (token1 + token2).decode('ascii', errors='replace')
                print(f"  {i:2d}. {repr(t1_str):8s} + {repr(t2_str):8s} = {repr(merged_str)}")
            except:
                print(f"  {i:2d}. {token1.hex()} + {token2.hex()} = {(token1+token2).hex()}")

        # Last 10 merges
        print("\nLast 10 merges:")
        for i, (token1, token2) in enumerate(merges[-10:], len(merges)-9):
            try:
                t1_str = token1.decode('ascii', errors='replace')
                t2_str = token2.decode('ascii', errors='replace')
                merged_str = (token1 + token2).decode('ascii', errors='replace')
                print(f"  {i:4d}. {repr(t1_str):15s} + {repr(t2_str):15s} = {repr(merged_str)}")
            except:
                print(f"  {i:4d}. {token1.hex()} + {token2.hex()} = {(token1+token2).hex()}")

        # Check for special tokens
        print("\nSpecial tokens:")
        for token_id, token_bytes in vocab.items():
            if token_bytes == b"<|endoftext|>":
                print(f"  ✓ '<|endoftext|>' found at token ID {token_id}")

        # Sample some longer tokens
        print("\nSample merged tokens (length >= 4 bytes):")
        count = 0
        for token_id, token_bytes in sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True):
            if len(token_bytes) >= 4 and count < 10:
                try:
                    decoded = token_bytes.decode('utf-8', errors='replace')
                    print(f"  ID {token_id:4d}: {repr(decoded):30s} ({len(token_bytes)} bytes)")
                    count += 1
                except:
                    pass

        return 0

    except Exception as e:
        print(f"\n✗ Error during training:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
