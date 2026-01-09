"""
Letter 'e' Image for einops Experimentation
Create a numpy array with the letter 'e' pattern for practicing einops operations
"""

import numpy as np
from einops import rearrange, reduce, repeat

# ==============================================================================
# TERMINAL VISUALIZATION FUNCTION
# ==============================================================================

def display_image_terminal(array, title="", method="ansi"):
    """Display a numpy array as an image in the terminal."""
    if title:
        print(f"\n{title}")

    if method == "ansi":
        arr = np.array(array)
        if arr.ndim == 3:
            # RGB image
            h, w = arr.shape[:2]
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)

            for i in range(0, h, 2):
                row_str = ""
                for j in range(w):
                    r1, g1, b1 = arr[i, j, :3].astype(int)
                    if i + 1 < h:
                        r2, g2, b2 = arr[i+1, j, :3].astype(int)
                    else:
                        r2, g2, b2 = r1, g1, b1
                    row_str += f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀\033[0m"
                print(row_str)
        else:
            # Grayscale image
            arr = np.array(array)
            if arr.max() > 1.0:
                arr = arr / arr.max()

            h, w = arr.shape
            for i in range(0, h, 2):
                row_str = ""
                for j in range(w):
                    val1 = arr[i, j]
                    val2 = arr[i+1, j] if i + 1 < h else val1
                    gray1 = int(val1 * 255)
                    gray2 = int(val2 * 255)
                    row_str += f"\033[38;2;{gray1};{gray1};{gray1}m\033[48;2;{gray2};{gray2};{gray2}m▀\033[0m"
                print(row_str)

    elif method == "ascii":
        ascii_chars = " .:-=+*#%@"
        arr = np.array(array)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        if arr.max() > 1.0:
            arr = arr / arr.max()

        h, w = arr.shape
        for i in range(h):
            row_str = ""
            for j in range(w):
                val = arr[i, j]
                char_idx = int(val * (len(ascii_chars) - 1))
                row_str += ascii_chars[char_idx] * 2
            print(row_str)

    print()


# ==============================================================================
# CREATE LETTER 'e' IMAGE
# ==============================================================================

def create_letter_e(size=16):
    """
    Create a numpy array with the letter 'e' pattern.

    Args:
        size: Size of the square image (default: 16x16)

    Returns:
        numpy array with shape (size, size) containing the letter 'e'
    """
    img = np.zeros((size, size))

    # Scale dimensions based on size
    margin = size // 8
    thickness = max(1, size // 8)

    # Top horizontal line
    img[margin:margin+thickness, margin:-margin] = 1.0

    # Middle horizontal line
    mid = size // 2
    img[mid-thickness//2:mid+thickness//2+1, margin:-margin] = 1.0

    # Bottom horizontal line
    img[-margin-thickness:-margin, margin:-margin] = 1.0

    # Left vertical line
    img[margin:-margin, margin:margin+thickness] = 1.0

    return img


# ==============================================================================
# MAIN PLAYGROUND
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LETTER 'e' EINOPS PLAYGROUND")
    print("=" * 70)

    # Create the letter 'e' image - now in high resolution!
    letter_e = create_letter_e(size=64)

    print("\nOriginal letter 'e' (64x64 - high resolution):")
    print(f"Shape: {letter_e.shape}")
    display_image_terminal(letter_e, "Letter 'e' (64x64)", method="ansi")

    # Also show it as a raw array
    print("\nRaw array (1 = white, 0 = black):")
    print(letter_e.astype(int))

    print("\n" + "=" * 70)
    print("EXPERIMENT IDEAS - TRY THESE EINOPS OPERATIONS!")
    print("=" * 70)

    print("""
Try these einops operations on 'letter_e':

1. TRANSPOSE:
   e_transposed = rearrange(letter_e, 'h w -> w h')
   display_image_terminal(e_transposed, "Transposed e")

2. FLIP VERTICALLY:
   e_flipped = letter_e[::-1, :]
   display_image_terminal(e_flipped, "Flipped e")

3. SPLIT INTO 4 QUADRANTS:
   quadrants = rearrange(letter_e, '(h1 h2) (w1 w2) -> (h1 w1) h2 w2', h2=32, w2=32)
   for i in range(4):
       display_image_terminal(quadrants[i], f"Quadrant {i}")

4. DOWNSAMPLE (4x4 max pooling to 16x16):
   downsampled_4x = reduce(letter_e, '(h 4) (w 4) -> h w', 'max')
   display_image_terminal(downsampled_4x, "Downsampled 4x (16x16)")

5. DOWNSAMPLE (8x8 mean pooling to 8x8):
   downsampled_8x = reduce(letter_e, '(h 8) (w 8) -> h w', 'mean')
   display_image_terminal(downsampled_8x, "Downsampled 8x (8x8)")

6. DOWNSAMPLE then UPSAMPLE (lossy compression demo):
   # Downsample 4x
   small = reduce(letter_e, '(h 4) (w 4) -> h w', 'mean')
   # Upsample 4x
   reconstructed = repeat(small, 'h w -> (h 4) (w 4)')
   display_image_terminal(reconstructed, "Downsample->Upsample (blocky)")

7. EXTREME COMPRESSION (16x downsample to 4x4, then back):
   tiny = reduce(letter_e, '(h 16) (w 16) -> h w', 'mean')
   blown_up = repeat(tiny, 'h w -> (h 16) (w 16)')
   display_image_terminal(blown_up, "Extreme compression (very blocky)")

8. CREATE A BATCH OF 3:
   batch = repeat(letter_e, 'h w -> b h w', b=3)
   print(f"Batch shape: {batch.shape}")

9. EXTRACT PATCHES (8x8 patches):
   patches = rearrange(letter_e, '(h p1) (w p2) -> (h w) p1 p2', p1=8, p2=8)
   print(f"Patches shape: {patches.shape}")
   for i in range(min(4, patches.shape[0])):
       display_image_terminal(patches[i], f"Patch {i}")

10. ROTATE 90 DEGREES:
    rotated = rearrange(letter_e, 'h w -> w h')[:, ::-1]
    display_image_terminal(rotated, "Rotated 90° clockwise")

11. CREATE RGB VERSION (colored 'e'):
    rgb_e = repeat(letter_e, 'h w -> h w c', c=3)
    rgb_e[:, :, 0] *= 1.0  # Red channel
    rgb_e[:, :, 1] *= 0.5  # Green channel
    rgb_e[:, :, 2] *= 0.2  # Blue channel
    display_image_terminal(rgb_e, "Colored e (reddish)")

12. FLATTEN TO 1D:
    flattened = rearrange(letter_e, 'h w -> (h w)')
    print(f"Flattened shape: {flattened.shape}")
    print(f"Number of white pixels: {flattened.sum()}")

""")

    print("=" * 70)
    print("YOUR EXPERIMENTS BELOW:")
    print("=" * 70)
    print()

    # ===== YOUR CODE HERE =====
    # Try out different einops operations!

    # # Example 1: Transpose
    # e_transposed = rearrange(letter_e, 'h w -> w h')
    # display_image_terminal(e_transposed, "1. Transposed e", method="ansi")

    # # Example 2: Downsample with max pooling
    # downsampled = reduce(letter_e, '(h h2) (w w2) -> h w', 'max', h2=2, w2=2)
    # display_image_terminal(downsampled, "2. Downsampled e (8x8)", method="ansi")

    # # Example 3: Create RGB colored version
    # rgb_e = np.stack([letter_e, letter_e * 0.5, letter_e * 0.2], axis=-1)
    # display_image_terminal(rgb_e, "3. Colored e (orange/red)", method="ansi")


    print(letter_e.shape)

    display_image_terminal(letter_e, "0. original e", method="ansi")

    e_transposed = rearrange(letter_e, "h w -> w h")
    display_image_terminal(e_transposed, "1. Transposed e", method="ansi")

    down_sampled_e = reduce(letter_e, "(h 2) (w 2) -> h w", "mean")
    display_image_terminal(down_sampled_e, "down sampled e", method="ansi")

    max_row_e = reduce(letter_e, "h w -> h ", "max")
    max_row_e_recover = repeat(max_row_e, "h -> h w", w=letter_e.shape[1])
    display_image_terminal(max_row_e_recover, "max_row_e_recover", method="ansi")

    # repeat_e = repeat(letter_e, "h w -> h (w repeat)", repeat=3)
    # display_image_terminal(repeat_e, "repeat_e", method="ansi")

    repeat_e = repeat(letter_e, "h w -> h 2 w 2")
    # display_image_terminal(repeat_e, "repeat_e", method="ansi")

    print(repeat_e.shape)

    dedup_e = reduce(repeat_e, "h 2 w 2 -> h w", "max")
    display_image_terminal(dedup_e, "dedup_e", method="ansi")

    down_sampled_e = reduce(letter_e, "(h 3) (w 3) -> h w", "mean")
    display_image_terminal(down_sampled_e, "down_sampled_e", method="ansi")
    up_sampled_e = repeat(down_sampled_e, "h w -> (h 3) (w 3)")
    display_image_terminal(up_sampled_e, "up_sampled_e", method="ansi")

    # Add your own experiments here!
    # ...

    print("\nHappy experimenting with einops! 🎨")
