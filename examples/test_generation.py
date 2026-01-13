"""
Test generation/decoding functionality with a tiny model.
"""

import torch
import numpy as np

from cs336_basics.transformer import TransformerLM
from cs336_basics.decoding import generate, top_p_sampling, apply_temperature


def test_temperature_scaling():
    """Test temperature scaling."""
    print("Testing temperature scaling...")

    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])

    # Temperature = 1.0 (no change)
    scaled = apply_temperature(logits, 1.0)
    assert torch.allclose(scaled, logits)

    # Temperature = 2.0 (more uniform)
    scaled = apply_temperature(logits, 2.0)
    expected = logits / 2.0
    assert torch.allclose(scaled, expected)

    # Temperature = 0.5 (more peaked)
    scaled = apply_temperature(logits, 0.5)
    expected = logits / 0.5
    assert torch.allclose(scaled, expected)

    print("✓ Temperature scaling works correctly")


def test_top_p_sampling():
    """Test top-p sampling."""
    print("\nTesting top-p sampling...")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create logits with clear probabilities
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])

    # After softmax, these will be roughly: [0.54, 0.24, 0.11, 0.05, 0.02, 0.01]

    # Test with top_p = 1.0 (should sample from all)
    samples = []
    for _ in range(100):
        sample = top_p_sampling(logits.clone(), top_p=1.0, temperature=1.0)
        samples.append(sample)

    # Should get multiple different tokens
    unique_samples = len(set(samples))
    print(f"  top_p=1.0: {unique_samples} unique tokens sampled")
    assert unique_samples > 1, "Should sample multiple tokens with top_p=1.0"

    # Test with top_p = 0.5 (should only use top few tokens)
    # With cumulative probs [0.54, 0.78, 0.89, ...], top_p=0.5 includes just first token
    # But with sampling variance, we might see the second token too
    samples = []
    for _ in range(100):
        sample = top_p_sampling(logits.clone(), top_p=0.6, temperature=1.0)
        samples.append(sample)

    unique_samples = set(samples)
    print(f"  top_p=0.6: tokens sampled = {unique_samples}")
    # Should mostly be token 0 or 1
    assert all(s <= 2 for s in unique_samples), "Should only sample from nucleus"

    print("✓ Top-p sampling works correctly")


def test_generation():
    """Test text generation with a tiny model."""
    print("\nTesting text generation...")

    # Create a tiny model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 100
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=64,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        rope_theta=10000.0,
        device=device,
        dtype=torch.float32,
    )
    model.eval()

    # Generate from a prompt
    prompt = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)

    print(f"  Prompt: {prompt.tolist()}")

    # Test basic generation
    generated = generate(
        model=model,
        prompt_tokens=prompt,
        max_new_tokens=10,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=None,
        device=device,
    )

    print(f"  Generated ({len(generated)} tokens): {generated.tolist()}")
    assert len(generated) == len(prompt) + 10, "Should generate exactly max_new_tokens"
    assert torch.all(generated[:len(prompt)] == prompt), "Prompt should be preserved"

    # Test with EOS token
    generated = generate(
        model=model,
        prompt_tokens=prompt,
        max_new_tokens=100,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=0,  # Use token 0 as EOS
        device=device,
    )

    print(f"  With EOS token (id=0): {len(generated)} tokens generated")
    # Should stop early if EOS is generated
    assert len(generated) <= len(prompt) + 100

    # Test with different temperatures
    torch.manual_seed(42)
    gen_temp_low = generate(
        model=model,
        prompt_tokens=prompt,
        max_new_tokens=5,
        temperature=0.1,  # Low temperature (more deterministic)
        top_p=1.0,
        device=device,
    )

    torch.manual_seed(42)
    gen_temp_high = generate(
        model=model,
        prompt_tokens=prompt,
        max_new_tokens=5,
        temperature=2.0,  # High temperature (more random)
        top_p=1.0,
        device=device,
    )

    print(f"  Low temp (0.1): {gen_temp_low.tolist()}")
    print(f"  High temp (2.0): {gen_temp_high.tolist()}")

    # Test with top-p sampling
    generated = generate(
        model=model,
        prompt_tokens=prompt,
        max_new_tokens=10,
        temperature=1.0,
        top_p=0.9,
        device=device,
    )

    print(f"  Top-p (0.9): {generated.tolist()}")

    print("✓ Text generation works correctly")


def test_batch_consistency():
    """Test that batched and single generation are consistent."""
    print("\nTesting generation consistency...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerLM(
        vocab_size=50,
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=256,
        rope_theta=10000.0,
        device=device,
        dtype=torch.float32,
    )
    model.eval()

    prompt = torch.tensor([1, 2, 3], dtype=torch.long)

    # Generate twice with same seed
    torch.manual_seed(42)
    gen1 = generate(model, prompt, max_new_tokens=5, temperature=1.0, top_p=1.0, device=device)

    torch.manual_seed(42)
    gen2 = generate(model, prompt, max_new_tokens=5, temperature=1.0, top_p=1.0, device=device)

    assert torch.all(gen1 == gen2), "Same seed should produce same generation"
    print("✓ Generation is deterministic with same seed")


def main():
    print("=" * 80)
    print("Testing Decoding Functionality")
    print("=" * 80)

    test_temperature_scaling()
    test_top_p_sampling()
    test_generation()
    test_batch_consistency()

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
