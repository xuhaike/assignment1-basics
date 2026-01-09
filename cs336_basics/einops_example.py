# Examples are given for numpy. This code also setups ipython/jupyter
# so that numpy arrays in the output are displayed as images
import numpy as np
from einops import rearrange, reduce, repeat
import sys

# ==============================================================================
# TERMINAL VISUALIZATION SETUP
# ==============================================================================
# Option 1: Use matplotlib to display images in pop-up windows (requires X11)
# Option 2: Use ANSI colors to display images in terminal using colored blocks
# Option 3: Use ASCII art with grayscale characters

def display_image_terminal(array, title="", method="ansi"):
    """
    Display a numpy array as an image in the terminal.

    Args:
        array: 2D numpy array (grayscale) or 3D array with shape (H, W, 3) for RGB
        title: Optional title to print above the image
        method: "ansi" for colored blocks, "ascii" for grayscale ASCII art, "matplotlib" for pop-up
    """
    if title:
        print(f"\n{title}")

    if method == "matplotlib":
        # Display in a pop-up window (requires X11 in MobaXterm)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 6))
            if array.ndim == 2:
                plt.imshow(array, cmap='viridis')
            else:
                plt.imshow(array)
            plt.title(title)
            plt.colorbar()
            plt.show(block=False)
            plt.pause(0.1)
        except ImportError:
            print("matplotlib not available, falling back to ANSI display")
            display_image_terminal(array, "", "ansi")

    elif method == "ansi":
        # Display using ANSI colored blocks (▀▄█ characters)
        # Normalize to 0-1 range
        arr = np.array(array)
        if arr.ndim == 3:
            # RGB image - average to grayscale for simplicity, or use true color
            # We'll use true color ANSI (24-bit color)
            h, w = arr.shape[:2]
            # Normalize to 0-255
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)

            for i in range(0, h, 2):  # Process 2 rows at a time
                row_str = ""
                for j in range(w):
                    # Get colors for top and bottom pixels
                    r1, g1, b1 = arr[i, j, :3].astype(int)
                    if i + 1 < h:
                        r2, g2, b2 = arr[i+1, j, :3].astype(int)
                    else:
                        r2, g2, b2 = r1, g1, b1

                    # Use upper half block with top pixel as foreground, bottom as background
                    row_str += f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀\033[0m"
                print(row_str)
        else:
            # Grayscale image
            arr = np.array(array)
            if arr.max() > 1.0:
                arr = arr / arr.max()

            h, w = arr.shape
            for i in range(0, h, 2):  # Process 2 rows at a time
                row_str = ""
                for j in range(w):
                    val1 = arr[i, j]
                    val2 = arr[i+1, j] if i + 1 < h else val1

                    # Convert to 0-255 for RGB
                    gray1 = int(val1 * 255)
                    gray2 = int(val2 * 255)

                    # Use upper half block
                    row_str += f"\033[38;2;{gray1};{gray1};{gray1}m\033[48;2;{gray2};{gray2};{gray2}m▀\033[0m"
                print(row_str)

    elif method == "ascii":
        # ASCII art using grayscale characters
        ascii_chars = " .:-=+*#%@"
        arr = np.array(array)

        # Convert to grayscale if RGB
        if arr.ndim == 3:
            arr = arr.mean(axis=2)

        # Normalize to 0-1
        if arr.max() > 1.0:
            arr = arr / arr.max()

        h, w = arr.shape
        for i in range(h):
            row_str = ""
            for j in range(w):
                val = arr[i, j]
                char_idx = int(val * (len(ascii_chars) - 1))
                row_str += ascii_chars[char_idx] * 2  # Double width for better aspect ratio
            print(row_str)

    print()  # Empty line after image

# Choose visualization method
# "ansi" - colored blocks in terminal (best for MobaXterm)
# "ascii" - grayscale ASCII art
# "matplotlib" - pop-up windows (requires X11, may need `export DISPLAY=:0`)
VIS_METHOD = "ansi"  # Change this to switch methods

# ==============================================================================
# EINOPS TUTORIAL: Elegant Tensor Operations Made Easy
# ==============================================================================
# einops is a library that provides a clear, readable syntax for tensor operations
# using Einstein-inspired notation. It works with NumPy, PyTorch, TensorFlow, etc.
#
# Three main operations:
# 1. rearrange - reshape, transpose, flatten, split, merge axes
# 2. reduce - perform reductions (mean, sum, max, min) along axes
# 3. repeat - duplicate data along axes

# ==============================================================================
# 1. REARRANGE: Reshaping and Transposing Tensors
# ==============================================================================

print("=" * 70)
print("1. REARRANGE EXAMPLES")
print("=" * 70)

# Example 1a: Simple transpose - Create a gradient image
x = np.linspace(0, 1, 64).reshape(8, 8)
print("\nOriginal gradient image (8, 8) - horizontal gradient:")
display_image_terminal(x, "Original horizontal gradient", VIS_METHOD)

# Transpose using einops - much more readable than x.T
x_transposed = rearrange(x, 'h w -> w h')
print("\nTransposed to (8, 8) using 'h w -> w h' - now vertical gradient:")
display_image_terminal(x_transposed, "Transposed to vertical gradient", VIS_METHOD)

# Example 1b: Creating a checkerboard pattern
h, w = 8, 8
x_check = (np.arange(h)[:, None] + np.arange(w)[None, :]) % 2
print("\n\nCheckerboard pattern (8, 8):")
display_image_terminal(x_check, "Checkerboard pattern", VIS_METHOD)

# Example 1c: RGB image - create colored stripes
h, w = 16, 16
red_channel = np.linspace(0, 1, h)[:, None] * np.ones((h, w))
green_channel = np.linspace(0, 1, w)[None, :] * np.ones((h, w))
blue_channel = np.ones((h, w)) * 0.5
rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
print(f"\n\nRGB gradient image shape: {rgb_image.shape}")
print("RGB image (red gradient vertical, green horizontal):")
display_image_terminal(rgb_image, "RGB gradient (red↓ green→)", VIS_METHOD)

# Example 1d: Splitting into patches - using the gradient image
x = np.linspace(0, 1, 64).reshape(8, 8)
print(f"\n\nOriginal 8x8 gradient:")
print(x)

# Split into 4 patches of 4x4 each
patches = rearrange(x, '(h p1) (w p2) -> (h w) p1 p2', p1=4, p2=4)
print(f"\nSplit into 4 patches of 4x4 using '(h p1) (w p2) -> (h w) p1 p2':")
print(f"Patches shape: {patches.shape}")
print("Patch 0 (top-left):")
print(patches[0])
print("Patch 3 (bottom-right):")
print(patches[3])

# Example 1e: Create a circular pattern
size = 32
y, x = np.ogrid[:size, :size]
center = size // 2
radius = size // 3
circle = ((x - center)**2 + (y - center)**2 <= radius**2).astype(float)
print(f"\n\nCircular pattern ({size}, {size}):")
display_image_terminal(circle, "Circle pattern", VIS_METHOD)

# Convert to channels-first by adding a channel dimension
circle_chw = rearrange(circle, 'h w -> 1 h w')
print(f"\nChannels-first format: {circle_chw.shape}")

# Example 1f: Vision Transformer patches - create a grid pattern to visualize
size = 16
grid = np.zeros((size, size))
grid[::4, :] = 1  # Horizontal lines
grid[:, ::4] = 1  # Vertical lines
print(f"\n\nGrid pattern ({size}, {size}):")
print(grid)

# Split into 4x4 patches (will get 16 patches)
patches = rearrange(grid, '(h p1) (w p2) -> (h w) p1 p2', p1=4, p2=4)
print(f"\nSplit into patches: {patches.shape}")
print("First patch:")
print(patches[0])

# ==============================================================================
# 2. REDUCE: Aggregating Data Along Axes
# ==============================================================================

print("\n" + "=" * 70)
print("2. REDUCE EXAMPLES")
print("=" * 70)

# Example 2a: Mean pooling on a noisy gradient
h, w = 8, 8
gradient = np.linspace(0, 1, h)[:, None] * np.ones((h, w))
noise = np.random.randn(h, w) * 0.1
noisy_gradient = gradient + noise
print(f"\nNoisy gradient image ({h}, {w}):")
print(noisy_gradient)

# Downsample by averaging 2x2 blocks
downsampled = reduce(noisy_gradient, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2)
print(f"\nDownsampled to {downsampled.shape} using 2x2 mean pooling:")
print(downsampled)

# Example 2b: RGB to grayscale conversion
h, w = 16, 16
# Create an RGB image with different channel values
rgb = np.zeros((h, w, 3))
rgb[:, :, 0] = np.linspace(0, 1, h)[:, None]  # Red increases vertically
rgb[:, :, 1] = np.linspace(0, 1, w)[None, :]  # Green increases horizontally
rgb[:, :, 2] = 0.3  # Blue is constant
print(f"\n\nRGB image shape: {rgb.shape}")
print("RGB image:")
print(rgb)

# Average across color channels to get grayscale
grayscale = reduce(rgb, 'h w c -> h w', 'mean')
print(f"\nGrayscale (average of channels) shape: {grayscale.shape}")
print(grayscale)

# Example 2c: Max pooling on a pattern with peaks
pattern = np.zeros((8, 8))
pattern[1, 1] = pattern[1, 6] = pattern[6, 1] = pattern[6, 6] = 1.0
print(f"\n\nPattern with peaks at corners of inner region:")
print(pattern)

# 2x2 max pooling - will capture the peaks
pooled = reduce(pattern, '(h h2) (w w2) -> h w', 'max', h2=2, w2=2)
print(f"\n2x2 max pooling result (peaks preserved):")
print(pooled)

# Example 2d: Comparing different reductions on same image
waves = np.sin(np.linspace(0, 4*np.pi, 16))[:, None] * np.ones((16, 16))
print(f"\n\nSine wave pattern (16, 16):")
print(waves)

print(f"\nMean pooling (2x2):")
print(reduce(waves, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2))

print(f"\nMax pooling (2x2):")
print(reduce(waves, '(h h2) (w w2) -> h w', 'max', h2=2, w2=2))

print(f"\nMin pooling (2x2):")
print(reduce(waves, '(h h2) (w w2) -> h w', 'min', h2=2, w2=2))

# ==============================================================================
# 3. REPEAT: Duplicating Data Along Axes
# ==============================================================================

print("\n" + "=" * 70)
print("3. REPEAT EXAMPLES")
print("=" * 70)

# Example 3a: Creating stripes by repeating
stripe = np.array([0, 1])
print(f"\nOriginal stripe pattern: {stripe}")

# Repeat to create horizontal stripes
h_stripes = repeat(stripe, 'w -> h w', h=8)
print(f"\nHorizontal stripes (8, 2):")
print(h_stripes)

# Repeat to create vertical stripes
stripe_col = np.array([[0], [1]])
v_stripes = repeat(stripe_col, 'h 1 -> h w', w=8)
print(f"\nVertical stripes (2, 8):")
print(v_stripes)

# Example 3b: Upsampling an image by pixel repetition
small_img = np.array([[0.2, 0.8], [0.9, 0.1]])
print(f"\n\nSmall 2x2 image:\n{small_img}")
display_image_terminal(small_img, "Original 2x2 image", VIS_METHOD)

# Upsample by repeating each pixel 4x4
upsampled = repeat(small_img, 'h w -> (h h2) (w w2)', h2=4, w2=4)
print(f"\nUpsampled to 8x8 by repeating each pixel 4x4:")
display_image_terminal(upsampled, "Upsampled 8x8 (nearest neighbor)", VIS_METHOD)

# Example 3c: Creating a tiled pattern from a small motif
# Create a simple 2x2 motif
motif = np.array([[1, 0], [0, 1]])  # Diagonal pattern
print(f"\n\n2x2 motif (diagonal):\n{motif}")

# Tile it to create a larger checkerboard
tiled = repeat(motif, 'h w -> (h t1) (w t2)', t1=4, t2=4)
print(f"\nTiled to 8x8:")
print(tiled)

# Example 3d: Broadcasting for batch processing
# Create a 4x4 kernel/filter
kernel = np.outer(np.linspace(1, 0, 4), np.linspace(1, 0, 4))
print(f"\n\n4x4 kernel (smooth falloff from top-left):\n{kernel}")

# Repeat to create a batch of 3 identical kernels
batch = repeat(kernel, 'h w -> b h w', b=3)
print(f"\nBatch of 3 identical kernels: {batch.shape}")
print(f"First kernel in batch:\n{batch[0]}")

# ==============================================================================
# 4. PRACTICAL DEEP LEARNING EXAMPLES
# ==============================================================================

print("\n" + "=" * 70)
print("4. PRACTICAL DEEP LEARNING EXAMPLES")
print("=" * 70)

# Example 4a: Image channel manipulation
h, w = 8, 8
# Create 3 different channel patterns
channel1 = np.linspace(0, 1, h)[:, None] * np.ones((h, w))  # Vertical gradient
channel2 = np.linspace(0, 1, w)[None, :] * np.ones((h, w))  # Horizontal gradient
channel3 = (np.arange(h)[:, None] + np.arange(w)[None, :]) % 2  # Checkerboard

# Stack into RGB image
rgb = np.stack([channel1, channel2, channel3], axis=-1)
print(f"\nRGB image with different patterns per channel: {rgb.shape}")
print("Red channel (vertical gradient):")
print(rgb[:, :, 0])
print("Green channel (horizontal gradient):")
print(rgb[:, :, 1])
print("Blue channel (checkerboard):")
print(rgb[:, :, 2])

# Rearrange to channels-first (common in PyTorch)
rgb_chw = rearrange(rgb, 'h w c -> c h w')
print(f"\nChannels-first format: {rgb_chw.shape}")

# Example 4b: Rearranging feature maps (like in CNNs)
# Simulate feature maps from a convolutional layer
batch = 2
height, width = 8, 8
channels = 16

# Create feature maps with different patterns
feature_maps = np.zeros((batch, height, width, channels))
for c in range(channels):
    # Each channel has a different frequency pattern
    freq = (c + 1) / 4
    feature_maps[0, :, :, c] = np.sin(np.linspace(0, freq * 2 * np.pi, height))[:, None]
    feature_maps[1, :, :, c] = np.cos(np.linspace(0, freq * 2 * np.pi, width))[None, :]

print(f"\n\nFeature maps shape (BHWC): {feature_maps.shape}")
print("First batch, first channel (low frequency):")
print(feature_maps[0, :, :, 0])
print("First batch, last channel (high frequency):")
print(feature_maps[0, :, :, -1])

# Rearrange to channels-first for PyTorch-style processing
feature_maps_chw = rearrange(feature_maps, 'b h w c -> b c h w')
print(f"\nChannels-first (BCHW): {feature_maps_chw.shape}")

# Example 4c: Grouped convolutions - split channels into groups
groups = 4
grouped = rearrange(feature_maps, 'b h w (g c) -> b g c h w', g=groups)
print(f"\n\nGrouped for grouped convolution:")
print(f"Original: {feature_maps.shape}")
print(f"Grouped into {groups} groups: {grouped.shape}")
print(f"({channels//groups} channels per group)")

# ==============================================================================
# 5. COMBINING OPERATIONS
# ==============================================================================

print("\n" + "=" * 70)
print("5. COMBINING OPERATIONS")
print("=" * 70)

# Example 5a: Creating complex patterns by combining operations
# Start with a simple motif
motif = np.array([[1, 0], [0, 0.5]])
print(f"\nOriginal 2x2 motif:")
print(motif)

# First, tile it
tiled = repeat(motif, 'h w -> (h t1) (w t2)', t1=4, t2=4)
print(f"\nTiled to 8x8:")
print(tiled)

# Then downsample with mean pooling
downsampled = reduce(tiled, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2)
print(f"\nDownsampled to 4x4:")
print(downsampled)

# Finally, rearrange into a different shape
final = rearrange(downsampled, 'h w -> (h w)')
print(f"\nFlattened to 1D: {final.shape}")
print(final)

# Example 5b: Image pyramid (multi-scale representation)
# Create a gradient image
img = np.linspace(0, 1, 16)[:, None] * np.linspace(0, 1, 16)[None, :]
print(f"\n\nOriginal 16x16 image:")
print(img)

# Create pyramid by repeated downsampling
level1 = reduce(img, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2)
print(f"\nPyramid level 1 (8x8):")
print(level1)

level2 = reduce(level1, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2)
print(f"\nPyramid level 2 (4x4):")
print(level2)

level3 = reduce(level2, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2)
print(f"\nPyramid level 3 (2x2):")
print(level3)

# Example 5c: Pixel shuffle / depth to space operation
# Common in super-resolution networks
h, w = 4, 4
depth = 4  # Will reshape to 2x upscale
# Create different patterns in each depth slice
deep_tensor = np.zeros((h, w, depth))
deep_tensor[:, :, 0] = 1.0
deep_tensor[:, :, 1] = 0.75
deep_tensor[:, :, 2] = 0.5
deep_tensor[:, :, 3] = 0.25

print(f"\n\nDeep tensor (4x4x4):")
print(f"Shape: {deep_tensor.shape}")
print("Depth slice 0:")
print(deep_tensor[:, :, 0])

# Pixel shuffle: depth to space
upscaled = rearrange(deep_tensor, 'h w (h2 w2) -> (h h2) (w w2)', h2=2, w2=2)
print(f"\nPixel shuffle to (8x8) - depth converted to spatial resolution:")
print(upscaled)

# ==============================================================================
# 6. BONUS: MORE VISUAL PATTERNS
# ==============================================================================

print("\n" + "=" * 70)
print("6. BONUS VISUAL PATTERNS")
print("=" * 70)

# Example 6a: Creating a radial gradient
size = 16
y, x = np.ogrid[:size, :size]
center = size // 2
distances = np.sqrt((x - center)**2 + (y - center)**2)
radial = 1 - (distances / distances.max())
print("\nRadial gradient (bright center, dark edges):")
display_image_terminal(radial, "Radial gradient", VIS_METHOD)

# Example 6b: Rotating an image using rearrange (90 degree rotations)
image = np.linspace(0, 1, 16).reshape(4, 4)
print(f"\n\nOriginal image (4x4 gradient):")
print(image)

# Rotate 90 degrees clockwise: transpose then flip horizontally
rotated_90 = rearrange(image, 'h w -> w h')[:, ::-1]
print("\nRotated 90° clockwise:")
print(rotated_90)

# Example 6c: Creating a border
img = np.ones((6, 6)) * 0.5
print("\n\nImage with uniform gray:")
print(img)

# Add a border by selecting regions
img_with_border = img.copy()
img_with_border[0, :] = img_with_border[-1, :] = 1.0  # Top and bottom
img_with_border[:, 0] = img_with_border[:, -1] = 1.0  # Left and right
print("\nImage with white border:")
print(img_with_border)

# Example 6d: Interleaving two images (like for stereo vision)
left_img = np.zeros((4, 4))
left_img[:, :2] = 1.0  # Left half bright
right_img = np.zeros((4, 4))
right_img[:, 2:] = 1.0  # Right half bright

print("\n\nLeft image:")
print(left_img)
print("Right image:")
print(right_img)

# Stack and interleave rows
stacked = np.stack([left_img, right_img], axis=0)
interleaved = rearrange(stacked, 'n h w -> (h n) w')
print("\nInterleaved (alternating rows):")
print(interleaved)

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. REARRANGE: Use for reshaping, transposing, splitting, merging axes
   - Syntax: 'input_axes -> output_axes'
   - Use parentheses for merging: '(h w)' or splitting with constraints

2. REDUCE: Use for aggregations (mean, sum, max, min, prod)
   - Removes specified axes by aggregating over them
   - Common for pooling operations

3. REPEAT: Use for duplicating data along axes
   - Explicit alternative to broadcasting
   - Good for creating batches or tiling

4. ADVANTAGES:
   - Readable: Code is self-documenting
   - Safe: Runtime checks ensure shape compatibility
   - Flexible: Works with NumPy, PyTorch, TensorFlow, JAX
   - Concise: Complex operations in one line

5. COMMON PATTERNS:
   - Image formats: 'b h w c -> b c h w' (BHWC to BCHW)
   - Flattening: 'b h w c -> b (h w c)'
   - Patches: '(h p1) (w p2) c -> (h w) (p1 p2 c)'
   - Pooling: '(h h2) (w w2) -> h w' with reduce
   - Upsampling: 'h w -> (h r1) (w r2)' with repeat
   - Pixel shuffle: 'h w (r1 r2) -> (h r1) (w r2)'

VISUALIZATION OPTIONS:
   Change VIS_METHOD at the top of the file:
   - "ansi": Colored blocks in terminal (works in MobaXterm!)
   - "ascii": Grayscale ASCII art
   - "matplotlib": Pop-up windows (requires: export DISPLAY=:0 in MobaXterm)
""")