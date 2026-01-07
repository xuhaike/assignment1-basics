#!/usr/bin/env python3
"""Simple test script for BPE training"""

import sys
import os
from pathlib import Path
import time

# Change to script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

from tests.adapters import run_train_bpe

def main():
    print("=" * 60)
    print("Testing BPE Training Implementation")
    print("=" * 60)

    input_path = Path('tests/fixtures/corpus.en')

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"\nInput file: {input_path}")
    print(f"Target vocab size: 500")
    print(f"Special tokens: ['<|endoftext|>']")
    print("\nStarting training...")

    start_time = time.time()
    try:
        vocab, merges = run_train_bpe(
            input_path=str(input_path),
            vocab_size=500,
            special_tokens=["<|endoftext|>"],
        )
        end_time = time.time()

        print(f"\n{'='*60}")
        print("Training completed successfully!")
        print(f"{'='*60}")
        print(f"Training time: {end_time - start_time:.3f} seconds")
        print(f"Vocab size: {len(vocab)}")
        print(f"Number of merges: {len(merges)}")

        # Check if we're within the time limit
        if end_time - start_time < 1.5:
            print(f"✓ Training time is within the 1.5 second limit")
        else:
            print(f"✗ Training took longer than 1.5 seconds (too slow)")

        print(f"\nFirst 5 merges:")
        for i, (token1, token2) in enumerate(merges[:5], 1):
            print(f"  {i}. {token1} + {token2} = {token1 + token2}")

        print(f"\nLast 5 merges:")
        for i, (token1, token2) in enumerate(merges[-5:], len(merges)-4):
            print(f"  {i}. {token1} + {token2} = {token1 + token2}")

        # Check for special token in vocab
        special_token_found = False
        for token_id, token_bytes in vocab.items():
            if token_bytes == b"<|endoftext|>":
                print(f"\n✓ Special token '<|endoftext|>' found at ID {token_id}")
                special_token_found = True
                break

        if not special_token_found:
            print(f"\n✗ Special token '<|endoftext|>' NOT found in vocabulary")

        # Save results in GPT-2 format
        import json
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)

        vocab_file = output_dir / 'corpus_vocab_0105.json'
        merges_file = output_dir / 'corpus_merges_0105.txt'

        # Import GPT-2 encoding function
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

    except Exception as e:
        print(f"\n✗ Error during training:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
