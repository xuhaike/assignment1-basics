# TinyStories Data Preparation Guide

## Quick Start

To prepare the TinyStories dataset for training:

```bash
./prepare_tinystories.sh
```

This will:
1. Load your trained tokenizer (vocab + merges)
2. Tokenize the training and validation text files
3. Save them as memory-mapped binary files for efficient loading

## What It Does

### Input Files

- **Training text**: `/home/ubuntu/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-train.txt` (~2.1 GB)
- **Validation text**: `/home/ubuntu/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-valid.txt` (~22 MB)
- **Vocabulary**: `/home/ubuntu/assignment1-basics/output/tinystories_vocab.json` (JSON format)
- **Merges**: `/home/ubuntu/assignment1-basics/output/tinystories_merges.txt` (text format)

### Output Files

- **Training data**: `data/tinystories_train.bin` (memmap format)
- **Validation data**: `data/tinystories_val.bin` (memmap format)

The binary files use `uint16` data type (supports vocab sizes up to 65,536).

## Manual Usage

If you need to customize the preparation:

```bash
# Process training data
uv run python cs336_basics/prepare_data.py \
    --input /path/to/train.txt \
    --output data/train.bin \
    --vocab /path/to/vocab.json \
    --merges /path/to/merges.txt \
    --dtype uint16

# Process validation data
uv run python cs336_basics/prepare_data.py \
    --input /path/to/val.txt \
    --output data/val.bin \
    --vocab /path/to/vocab.json \
    --merges /path/to/merges.txt \
    --dtype uint16
```

### Parameters

- `--input`: Path to input text file (required)
- `--output`: Path to output memmap file (required)
- `--vocab`: Path to vocabulary file - supports `.json` or `.pkl` (required)
- `--merges`: Path to merges file - supports `.txt` or `.pkl` (required)
- `--dtype`: Data type - "uint16" or "uint32" (default: uint16)
- `--num_workers`: Number of processes for parallel tokenization (default: auto-detect, 0 for single-threaded)
- `--special_tokens`: Special tokens to add (optional)

### Performance

The script uses **multiprocessing** to parallelize tokenization across multiple CPU cores:
- Automatically detects number of CPU cores
- Shows progress bar with `tqdm` if installed
- Fallback to single-threaded if multiprocessing fails
- Use `--num_workers 0` to force single-threaded mode

## File Formats

### Vocabulary JSON Format

```json
{
  "token_string": token_id,
  "Ġ": 32,
  "!": 33,
  "the": 256,
  ...
}
```

Mapping from token strings to integer IDs.

### Merges Text Format

```
token1 token2
Ġ t
h e
Ġt he
...
```

Each line contains two space-separated tokens that were merged during BPE training.

### Output Memmap Format

Binary file containing token IDs as uint16 (or uint32) integers:
```
[token_id_0, token_id_1, token_id_2, ...]
```

Can be loaded efficiently with:
```python
import numpy as np
data = np.memmap('data/tinystories_train.bin', dtype=np.uint16, mode='r')
```

## After Preparation

Once data is prepared, use it for training:

```bash
# Baseline 17M parameter model
uv run python cs336_basics/train_with_logging.py \
    --experiment_name "baseline_17m_tinystories" \
    --description "Baseline 17M model on TinyStories" \
    --tags "baseline,17m,tinystories" \
    --train_data_path data/tinystories_train.bin \
    --val_data_path data/tinystories_val.bin \
    --vocab_size 50257 \
    --context_length 512 \
    --d_model 512 \
    --num_layers 8 \
    --num_heads 8 \
    --d_ff 2048 \
    --batch_size 8 \
    --max_iters 50000 \
    --checkpoint_dir checkpoints/baseline_17m
```

## Troubleshooting

### Error: Input file not found

Make sure the TinyStories data files are in the correct location:
```bash
ls -lh /home/ubuntu/assignment1-basics/cs336_basics/data/TinyStories*.txt
```

### Error: Vocabulary/merges not found

Ensure you've trained your tokenizer first and the output files exist:
```bash
ls -lh /home/ubuntu/assignment1-basics/output/tinystories_*
```

### Out of memory during tokenization

The script processes files in chunks, but if you still run out of memory:
1. Reduce chunk size in `prepare_tinystories_data.py`
2. Close other applications
3. Use a machine with more RAM

### Vocabulary size mismatch

If your vocabulary size differs from 50257:
1. Count the actual size:
   ```bash
   python -c "import json; print(len(json.load(open('output/tinystories_vocab.json'))))"
   ```
2. Use this value for `--vocab_size` when training

## Performance

With **multiprocessing enabled** (default), typical preparation times:
- **Training data** (~2.1 GB): 1-3 minutes (with 8+ cores)
- **Validation data** (~22 MB): 5-10 seconds

Single-threaded mode (`--num_workers 0`) may take 2-3x longer.

Output file sizes are much smaller than input (typically 10-20% of original size).

### Optimization Tips

- More CPU cores = faster processing
- Install `tqdm` for progress bars: `pip install tqdm`
- Use `--num_workers N` to control parallelism
- For very large files, increase `chunk_size` in code

## Next Steps

1. **Prepare data**: `./prepare_tinystories.sh`
2. **Verify data**: Check output files exist in `data/`
3. **Train model**: Use the training command above
4. **Monitor progress**: Check experiment logs in `experiments/`

See also:
- `TRAINING_GUIDE.md` - Complete training guide
- `EXPERIMENT_GUIDE.md` - Experiment tracking
- `EXPERIMENT_LOG_TEMPLATE.md` - Document your experiments
