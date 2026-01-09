"""
Quick script to check mean pooling values
"""
import numpy as np
from einops import reduce
from letter_e_playground import create_letter_e, display_image_terminal

# Create high-res letter 'e'
letter_e = create_letter_e(size=64)
print(f"Original shape: {letter_e.shape}")
print(f"Original unique values: {np.unique(letter_e)}")

# Downsample with mean pooling
down_sampled_e = reduce(letter_e, "(h 4) (w 4) -> h w", "mean")
print(f"\nDownsampled shape: {down_sampled_e.shape}")
print(f"Downsampled unique values: {sorted(np.unique(down_sampled_e))}")
print(f"Number of unique values: {len(np.unique(down_sampled_e))}")

# Show the actual array to see intermediate values
print("\nDownsampled array (showing intermediate values if any):")
print(down_sampled_e)

# Count how many pixels have intermediate values
intermediate_count = np.sum((down_sampled_e > 0) & (down_sampled_e < 1))
print(f"\nPixels with intermediate values (0 < val < 1): {intermediate_count}")

# Visualize
display_image_terminal(down_sampled_e, "Downsampled (4x) - may look binary", method="ansi")

# Try different downsampling to get more intermediate values
print("\n" + "="*70)
print("Trying 3x3 downsampling instead (doesn't align with strokes):")
print("="*70)

# Pad to make it divisible by 3
# 64 is not divisible by 3, but we can crop to 63
letter_e_crop = letter_e[:63, :63]
down_sampled_3x = reduce(letter_e_crop, "(h 3) (w 3) -> h w", "mean")
print(f"\nDownsampled 3x shape: {down_sampled_3x.shape}")
print(f"Unique values: {len(np.unique(down_sampled_3x))}")
print(f"Sample unique values: {sorted(np.unique(down_sampled_3x))[:20]}")  # Show first 20

# Count intermediate values
intermediate_count_3x = np.sum((down_sampled_3x > 0) & (down_sampled_3x < 1))
print(f"Pixels with intermediate values: {intermediate_count_3x}")

display_image_terminal(down_sampled_3x, "Downsampled (3x) - should show gradients", method="ansi")

# Try 5x5 downsampling
print("\n" + "="*70)
print("Trying 5x5 downsampling:")
print("="*70)

letter_e_crop2 = letter_e[:60, :60]
down_sampled_5x = reduce(letter_e_crop2, "(h 5) (w 5) -> h w", "mean")
print(f"\nDownsampled 5x shape: {down_sampled_5x.shape}")
print(f"Unique values: {len(np.unique(down_sampled_5x))}")
print(f"Pixels with intermediate values: {np.sum((down_sampled_5x > 0) & (down_sampled_5x < 1))}")

display_image_terminal(down_sampled_5x, "Downsampled (5x) - should show gradients", method="ansi")
