import re
from collections import Counter
from typing import List, Tuple, Dict


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str] = None
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to a text file with BPE tokenizer training data
        vocab_size: Maximum final vocabulary size
        special_tokens: List of special tokens to add to vocabulary

    Returns:
        vocab: Mapping from token ID to token bytes
        merges: List of BPE merges in order of creation
    """
    if special_tokens is None:
        special_tokens = []

    # Step 1: Read and encode text as UTF-8
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Step 2: Pretokenization - split by special tokens
    words = pretokenize(text, special_tokens)

    # Step 3: Initialize byte-level vocabulary (0-255)
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    # Step 4: Convert words to sequences of bytes
    # Each word becomes a list of individual byte tokens initially
    word_tokens = []
    for word in words:
        word_bytes = word.encode('utf-8')
        # Each byte becomes a separate token (as bytes object)
        tokens = [bytes([b]) for b in word_bytes]
        word_tokens.append(tokens)

    merges = []

    # Step 5: Run BPE until we reach vocab_size (accounting for special tokens)
    target_merges = vocab_size - len(vocab) - len(special_tokens)

    for _ in range(target_merges):
        # Count all adjacent pairs
        pair_counts = count_pairs(word_tokens)

        if not pair_counts:
            break

        # Find most frequent pair
        most_frequent_pair = max(pair_counts, key=pair_counts.get)

        # Merge the most frequent pair
        word_tokens = merge_pair(word_tokens, most_frequent_pair)

        # Record the merge
        merges.append(most_frequent_pair)

        # Add merged token to vocabulary
        merged_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[next_token_id] = merged_token
        next_token_id += 1

    # Step 6: Add special tokens to vocabulary
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode('utf-8')
        next_token_id += 1

    return vocab, merges


def pretokenize(text: str, special_tokens: List[str]) -> List[str]:
    """
    Pretokenize text by splitting on special tokens.

    Args:
        text: Input text
        special_tokens: List of special tokens to split on

    Returns:
        List of text chunks
    """
    if not special_tokens:
        return [text] if text else []

    # Build regex pattern that splits on special tokens
    # Escape special regex characters and join with |
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = '|'.join(escaped_tokens)

    # Use re.split to split by special tokens, keeping the delimiters
    # Then filter out the special tokens themselves and empty strings
    parts = re.split(f'({pattern})', text)

    # Keep only non-empty parts that aren't special tokens
    special_tokens_set = set(special_tokens)
    words = [part for part in parts if part and part not in special_tokens_set]

    return words


def count_pairs(word_tokens: List[List[bytes]]) -> Counter:
    """
    Count frequency of all adjacent token pairs.

    Args:
        word_tokens: List of words, each word is a list of byte tokens

    Returns:
        Counter of (token1, token2) pairs
    """
    pair_counts = Counter()

    for tokens in word_tokens:
        # Count adjacent pairs in this word's token sequence
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += 1

    return pair_counts


def merge_pair(
    word_tokens: List[List[bytes]],
    pair: Tuple[bytes, bytes]
) -> List[List[bytes]]:
    """
    Merge all occurrences of a token pair.

    Args:
        word_tokens: List of words, each word is a list of byte tokens
        pair: Tuple of (token1, token2) to merge

    Returns:
        Updated word_tokens with pair merged
    """
    token1, token2 = pair
    merged_token = token1 + token2

    new_word_tokens = []
    for tokens in word_tokens:
        new_seq = []
        i = 0
        while i < len(tokens):
            # Check if current and next token match the pair
            if i < len(tokens) - 1 and tokens[i] == token1 and tokens[i + 1] == token2:
                new_seq.append(merged_token)
                i += 2
            else:
                new_seq.append(tokens[i])
                i += 1
        new_word_tokens.append(new_seq)

    return new_word_tokens
