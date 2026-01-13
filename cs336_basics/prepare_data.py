"""
Utility script to prepare training data in memmap format.

This script tokenizes text data and saves it as a memory-mapped numpy array
for efficient data loading during training.
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


# Global variables for multiprocessing
_tokenizer = None
_tokenizer_vocab_path = None
_tokenizer_merges_path = None
_special_tokens = None


def _init_worker(vocab_path, merges_path, special_tokens):
    """Initialize tokenizer in worker process."""
    global _tokenizer, _tokenizer_vocab_path, _tokenizer_merges_path, _special_tokens
    import pickle
    from cs336_basics.tokenizer import Tokenizer

    _tokenizer_vocab_path = vocab_path
    _tokenizer_merges_path = merges_path
    _special_tokens = special_tokens

    # Load tokenizer
    if vocab_path.endswith('.json') or merges_path.endswith('.txt'):
        _tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    else:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_path, 'rb') as f:
            merges = pickle.load(f)
        _tokenizer = Tokenizer(vocab, merges, special_tokens or [])


def _encode_chunk(chunk):
    """Encode a single chunk using the worker's tokenizer."""
    global _tokenizer
    return _tokenizer.encode(chunk)


def prepare_data(
    input_path: str,
    output_path: str,
    tokenizer_vocab_path: str,
    tokenizer_merges_path: str,
    special_tokens: list[str] = None,
    dtype=np.uint16,
    chunk_size: int = 1024 * 1024,  # Process 1MB at a time
    num_workers: int = None,  # Number of workers for multiprocessing (None = auto)
):
    """
    Tokenize text data and save as memmap.

    Args:
        input_path: Path to input text file
        output_path: Path to output memmap file
        tokenizer_vocab_path: Path to tokenizer vocabulary file
        tokenizer_merges_path: Path to tokenizer merges file
        special_tokens: List of special tokens
        dtype: Data type for tokens (np.uint16 for vocab_size <= 65536)
        chunk_size: Size of text chunks to process at once
        num_workers: Number of worker processes (None = auto-detect, 0 = single-threaded)
    """
    import pickle
    from cs336_basics.tokenizer import Tokenizer

    # Load tokenizer
    print("Loading tokenizer...")

    # Try loading with Tokenizer.from_files first (supports JSON/txt format)
    if tokenizer_vocab_path.endswith('.json') or tokenizer_merges_path.endswith('.txt'):
        print(f"  Loading from JSON/text format...")
        print(f"    Vocab: {tokenizer_vocab_path}")
        print(f"    Merges: {tokenizer_merges_path}")
        tokenizer = Tokenizer.from_files(
            tokenizer_vocab_path,
            tokenizer_merges_path,
            special_tokens=special_tokens
        )
    else:
        # Fallback to pickle format
        print(f"  Loading from pickle format...")
        print(f"    Vocab: {tokenizer_vocab_path}")
        print(f"    Merges: {tokenizer_merges_path}")
        with open(tokenizer_vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(tokenizer_merges_path, 'rb') as f:
            merges = pickle.load(f)
        tokenizer = Tokenizer(vocab, merges, special_tokens or [])

    print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")

    # Read input file size
    input_file_size = Path(input_path).stat().st_size
    print(f"Input file size: {input_file_size / (1024**2):.2f} MB")

    # Read file and split into chunks
    print("Reading file...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"File length: {len(text):,} characters")

    # Split into chunks for parallel processing
    num_chunks = max(1, len(text) // chunk_size)
    print(f"Splitting into {num_chunks} chunks for parallel processing...")

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)

    print(f"Created {len(chunks)} chunks")

    # Tokenize in parallel
    if num_workers == 0:
        # Single-threaded mode
        print("Tokenizing (single-threaded)...")
        all_tokens = []
        try:
            from tqdm import tqdm
            chunks_iter = tqdm(chunks, desc="Tokenizing", unit="chunk")
        except ImportError:
            chunks_iter = chunks

        for chunk in chunks_iter:
            tokens = tokenizer.encode(chunk)
            all_tokens.extend(tokens)

        print(f"Total tokens: {len(all_tokens):,}")

    else:
        # Multi-threaded mode
        print("Tokenizing with multiprocessing...")

        try:
            from multiprocessing import Pool, cpu_count

            if num_workers is None:
                num_processes = min(cpu_count(), len(chunks))
            else:
                num_processes = min(num_workers, len(chunks))

            print(f"  Using {num_processes} processes")

            # Try to use tqdm for progress bar
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                use_tqdm = False

            with Pool(
                processes=num_processes,
                initializer=_init_worker,
                initargs=(tokenizer_vocab_path, tokenizer_merges_path, special_tokens)
            ) as pool:
                if use_tqdm:
                    # Show progress bar
                    results = list(tqdm(
                        pool.imap(_encode_chunk, chunks),
                        total=len(chunks),
                        desc="Tokenizing",
                        unit="chunk"
                    ))
                else:
                    results = pool.map(_encode_chunk, chunks)

            # Flatten results
            all_tokens = []
            for tokens in results:
                all_tokens.extend(tokens)

            print(f"Total tokens: {len(all_tokens):,}")

        except Exception as e:
            print(f"Multiprocessing failed ({e}), falling back to single-threaded...")

            # Fallback to single-threaded processing
            all_tokens = []
            for i, chunk in enumerate(chunks, 1):
                tokens = tokenizer.encode(chunk)
                all_tokens.extend(tokens)

                if i % 100 == 0:
                    print(f"Processed {i}/{len(chunks)} chunks, {len(all_tokens):,} tokens so far...")

            print(f"Total tokens: {len(all_tokens):,}")

    # Convert to numpy array
    print("Converting to numpy array...")
    tokens_array = np.array(all_tokens, dtype=dtype)

    # Save as memmap
    print(f"Saving to {output_path}...")
    memmap_array = np.memmap(output_path, dtype=dtype, mode='w+', shape=tokens_array.shape)
    memmap_array[:] = tokens_array[:]
    memmap_array.flush()

    print("Done!")
    print(f"Saved {len(tokens_array):,} tokens to {output_path}")
    print(f"Output file size: {Path(output_path).stat().st_size / (1024**2):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data in memmap format")

    parser.add_argument("--input", type=str, required=True,
                        help="Path to input text file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output memmap file")
    parser.add_argument("--vocab", type=str, required=True,
                        help="Path to tokenizer vocabulary (pickle or JSON file)")
    parser.add_argument("--merges", type=str, required=True,
                        help="Path to tokenizer merges (pickle or txt file)")
    parser.add_argument("--special_tokens", type=str, nargs='*', default=None,
                        help="Special tokens (space-separated)")
    parser.add_argument("--dtype", type=str, default="uint16",
                        help="Data type (uint16 or uint32)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes (None=auto, 0=single-threaded)")

    args = parser.parse_args()

    # Convert dtype string to numpy dtype
    dtype_map = {
        "uint16": np.uint16,
        "uint32": np.uint32,
    }
    dtype = dtype_map.get(args.dtype, np.uint16)

    prepare_data(
        input_path=args.input,
        output_path=args.output,
        tokenizer_vocab_path=args.vocab,
        tokenizer_merges_path=args.merges,
        special_tokens=args.special_tokens,
        dtype=dtype,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
