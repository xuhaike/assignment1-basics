"""BPE Tokenizer implementation."""

from __future__ import annotations
import json
import re
import regex
from typing import Iterable, Iterator


class Tokenizer:
    """Byte-Pair Encoding (BPE) Tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and special tokens.

        Args:
            vocab: Mapping from token ID to token bytes
            merges: List of BPE merges, each merge is (token1, token2)
            special_tokens: Optional list of special token strings
        """
        # Copy vocab to avoid modifying the original
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        # Add special tokens to vocab if not already present
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode('utf-8')
            # Check if this special token is already in vocab
            if special_token_bytes not in self.vocab.values():
                self.vocab[next_id] = special_token_bytes
                next_id += 1

        # Build reverse vocab: bytes -> token_id (for encoding)
        self.vocab_reverse = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        # Build merge rules: (token1, token2) -> merged_token
        # This will be used during encoding to know which pairs to merge
        self.merge_rules = {}
        for token1, token2 in self.merges:
            merged = token1 + token2
            self.merge_rules[(token1, token2)] = merged

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """
        Construct a Tokenizer from serialized vocabulary and merges files.

        Args:
            vocab_filepath: Path to vocabulary JSON file
            merges_filepath: Path to merges text file
            special_tokens: Optional list of special token strings

        Returns:
            Tokenizer instance
        """
        # Load vocabulary from JSON
        # The vocab file uses GPT-2 format: {token_string: token_id}
        # We need to convert it to {token_id: token_bytes}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        # Import GPT-2 encoding function
        try:
            from .common import gpt2_bytes_to_unicode
        except ImportError:
            # Fallback for standalone usage
            from common import gpt2_bytes_to_unicode

        # Get the reverse mapping: unicode char -> byte value
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        # Convert GPT-2 format to our format
        vocab = {}
        for token_str, token_id in vocab_data.items():
            # Convert GPT-2 unicode string back to bytes
            token_bytes = bytes([byte_decoder[c] for c in token_str])
            vocab[token_id] = token_bytes

        # Load merges from text file
        # Each line is: "token1 token2" in GPT-2 unicode format
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ')
                if len(parts) != 2:
                    continue

                # Convert from GPT-2 unicode format to bytes
                token1_str, token2_str = parts
                token1 = bytes([byte_decoder[c] for c in token1_str])
                token2 = bytes([byte_decoder[c] for c in token2_str])
                merges.append((token1, token2))

        return cls(vocab, merges, special_tokens)

    def _pretokenize(self, text: str) -> list[tuple[str, bool]]:
        """
        Pretokenize text by splitting into words using GPT-2 pattern.
        Handles special tokens by splitting on them.

        Args:
            text: Input text

        Returns:
            List of (word, is_special_token) tuples
        """
        # GPT-2 pre-tokenization pattern
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Split by special tokens if any
        if self.special_tokens:
            # Sort special tokens by length (longest first) to handle prefix cases
            # E.g., if we have both "<|end|>" and "<|endoftext|>", we want to match
            # "<|endoftext|>" first, not split it into "<|end|>" + "oftext|>"
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_special_tokens]
            special_pattern = '|'.join(escaped_tokens)
            # Split by special tokens to avoid merging across boundaries
            parts = re.split(f'({special_pattern})', text)
            special_tokens_set = set(self.special_tokens)
        else:
            parts = [text] if text else []
            special_tokens_set = set()

        # Apply GPT-2 regex pattern to each part
        # Track whether each token is a special token
        words = []
        for part in parts:
            if not part:
                continue

            # Check if this part is a special token
            if part in special_tokens_set:
                words.append((part, True))  # Mark as special token
            else:
                # Apply regex pattern to regular text
                for match in regex.finditer(PAT, part):
                    words.append((match.group(), False))  # Mark as regular token

        return words

    def _apply_bpe(self, word_bytes: bytes) -> list[bytes]:
        """
        Apply BPE merges to a word (byte sequence) following the historical merge order.

        Args:
            word_bytes: Word as bytes

        Returns:
            List of token bytes after applying BPE merges
        """
        # Start with individual bytes as tokens
        tokens = [bytes([b]) for b in word_bytes]

        # Apply merges in order until no more merges can be applied
        while len(tokens) > 1:
            # Find the first merge that applies to adjacent tokens
            min_merge_idx = float('inf')
            best_pair_idx = -1

            # Check all adjacent pairs
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                # Check if this pair has a merge rule
                if pair in self.merge_rules:
                    # Find the index of this merge in the original merge list
                    try:
                        merge_idx = self.merges.index(pair)
                        # Keep track of the earliest merge
                        if merge_idx < min_merge_idx:
                            min_merge_idx = merge_idx
                            best_pair_idx = i
                    except ValueError:
                        pass

            # If no merge found, we're done
            if best_pair_idx == -1:
                break

            # Apply the merge at best_pair_idx
            pair = (tokens[best_pair_idx], tokens[best_pair_idx + 1])
            merged_token = self.merge_rules[pair]

            # Replace the pair with the merged token
            tokens = tokens[:best_pair_idx] + [merged_token] + tokens[best_pair_idx + 2:]

        return tokens

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        token_ids = []

        # Step 1: Pretokenize the text
        words = self._pretokenize(text)

        # Step 2: For each word, apply BPE and convert to token IDs
        for word, is_special in words:
            if is_special:
                # Special tokens are encoded directly without BPE
                word_bytes = word.encode('utf-8')
                if word_bytes in self.vocab_reverse:
                    token_ids.append(self.vocab_reverse[word_bytes])
                else:
                    raise ValueError(f"Special token {word} not found in vocabulary")
            else:
                # Convert word to bytes
                word_bytes = word.encode('utf-8')

                # Apply BPE merges
                tokens = self._apply_bpe(word_bytes)

                # Convert token bytes to token IDs
                for token_bytes in tokens:
                    if token_bytes in self.vocab_reverse:
                        token_ids.append(self.vocab_reverse[token_bytes])
                    else:
                        # Token not in vocabulary - this shouldn't happen with proper BPE
                        # But if it does, we could handle it by encoding as individual bytes
                        raise ValueError(f"Token {token_bytes} not found in vocabulary")

        return token_ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
        show_progress: bool = True
    ) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.

        Note: Multiprocessing is not used here due to pickling constraints with instance methods.
        For large-scale parallel processing, consider batching externally.

        Args:
            iterable: Iterable of strings (e.g., file handle)
            show_progress: Whether to show a progress bar

        Yields:
            Token IDs
        """
        # Convert to list if we want to show progress bar
        if show_progress:
            try:
                from tqdm import tqdm
                # Try to get length if possible
                try:
                    total = len(iterable)
                    iterable = tqdm(iterable, desc="Encoding", unit="chunk", total=total)
                except TypeError:
                    # Iterable doesn't support len(), use tqdm without total
                    iterable = tqdm(iterable, desc="Encoding", unit="chunk")
            except ImportError:
                # tqdm not available, just process without progress bar
                pass

        for text_chunk in iterable:
            # Encode this chunk and yield each token ID
            token_ids = self.encode(text_chunk)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string
        """
        # Concatenate all token bytes
        result_bytes = b''
        for token_id in ids:
            if token_id in self.vocab:
                result_bytes += self.vocab[token_id]
            else:
                # Unknown token ID
                raise ValueError(f"Token ID {token_id} not found in vocabulary")

        # Decode bytes to string, using replacement character for invalid UTF-8
        return result_bytes.decode('utf-8', errors='replace')
