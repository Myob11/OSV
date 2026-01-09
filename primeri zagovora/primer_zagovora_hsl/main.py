#%%
"""
Image Processing Script: RGB to HSL Conversion and Hue Channel Filtering

This script performs color space transformations on an image:
1. Loads and standardizes an RGB image (z-score normalization per channel)
2. Converts RGB to HSL color space
3. Applies filtering to the hue channel
4. Converts back to RGB for visualization

Author: Image Processing Exercise
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread


def standardize_image(image):
    """
    Standardizes an RGB image by applying z-score normalization per channel.
    
    For each color channel (R, G, B), this function:
    - Computes the mean and standard deviation
    - Normalizes: (pixel_value - mean) / std
    
    This centers each channel around 0 with unit variance.
    
    Args:
        image: Input RGB image (numpy array of shape H x W x 3)
    
    Returns:
        Standardized image (same shape as input, dtype=float)
    """
    image = image.astype(float)
    standardized = np.copy(image)
    
    # Normalize each color channel independently
    for channel in range(3):
        channel_data = image[:, :, channel]
        mean = channel_data.mean()
        std = channel_data.std()
        standardized[:, :, channel] = (channel_data - mean) / std
    
    return standardized


def rgb2hsl(image):
    """
    Converts an RGB image to HSL (Hue, Saturation, Lightness) color space.
    
    The input image should have pixel values in the range [0, 1].
    
    Algorithm:
    - V (value) = max(R, G, B)
    - C (chroma) = V - min(R, G, B)
    - L (lightness) = V - C/2
    - H (hue) is calculated based on which channel is dominant
    - S (saturation) = C / (1 - |2L - 1|)
    
    Args:
        image: Input RGB image with values in [0, 1] (numpy array of shape H x W x 3)
    
    Returns:
        HSL image (numpy array of shape H x W x 3)
        - H: Hue in degrees [0, 360]
        - S: Saturation [0, 1]
        - L: Lightness [0, 1]
    """
    # Extract RGB channels
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    
    # Calculate value (maximum of RGB) and chroma (difference between max and min)
    V = np.max(image, axis=2)
    C = V - np.min(image, axis=2)
    
    # Calculate lightness: average of max and min RGB values
    L = V - C / 2
    
    # Calculate hue based on which RGB channel is dominant
    H = np.zeros_like(V)
    # When red is dominant
    H[V == R] = (((G - B) / C))[V == R]
    # When green is dominant
    H[V == G] = ((2 + (B - R) / C))[V == G]
    # When blue is dominant
    H[V == B] = ((4 + (R - G) / C))[V == B]
    # Handle grayscale (no chroma)
    H[C == 0] = 0
    # Convert to degrees [0, 360]
    H = H * 60
    
    # Calculate saturation per HSL definition
    S = np.zeros_like(V)
    # Avoid division by zero when lightness is 0 or 1
    valid_mask = (L != 0) & (L != 1)
    S[valid_mask] = (C / (1 - np.abs(2 * L - 1)))[valid_mask]
    
    return np.stack((H, S, L), axis=2)


def hsl2rgb(image):
    """
    Converts an HSL image back to RGB color space.
    
    This is the inverse operation of rgb2hsl().
    
    Algorithm:
    - C (chroma) = (1 - |2L - 1|) * S
    - H' = H / 60 (normalized hue)
    - X = C * (1 - |H' mod 2 - 1|)
    - RGB values are assigned based on which sextant H' falls into
    - m = L - C/2 (adjustment to match lightness)
    - Final RGB = (R', G', B') + m
    
    Args:
        image: Input HSL image (numpy array of shape H x W x 3)
               - H: Hue in degrees [0, 360]
               - S: Saturation [0, 1]
               - L: Lightness [0, 1]
    
    Returns:
        RGB image with values in [0, 1] (numpy array of shape H x W x 3)
    """
    H = image[:, :, 0]
    S = image[:, :, 1]
    L = image[:, :, 2]
    
    # Calculate chroma from saturation and lightness
    C = (1 - np.abs(2 * L - 1)) * S
    
    # Normalize hue to [0, 6) range
    H_normalized = H / 60
    # Intermediate value for RGB calculation
    X = C * (1 - np.abs(H_normalized % 2 - 1))
    
    # Initialize RGB channels
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)
    
    # Determine which sextant of the color wheel each pixel falls into
    # and assign RGB values accordingly
    mask0 = (0 <= H_normalized) & (H_normalized < 1)  # Red to Yellow
    mask1 = (1 <= H_normalized) & (H_normalized < 2)  # Yellow to Green
    mask2 = (2 <= H_normalized) & (H_normalized < 3)  # Green to Cyan
    mask3 = (3 <= H_normalized) & (H_normalized < 4)  # Cyan to Blue
    mask4 = (4 <= H_normalized) & (H_normalized < 5)  # Blue to Magenta
    mask5 = (5 <= H_normalized) & (H_normalized < 6)  # Magenta to Red
    
    # Assign RGB values based on hue sextant
    R[mask0] = C[mask0]; G[mask0] = X[mask0]; B[mask0] = 0
    R[mask1] = X[mask1]; G[mask1] = C[mask1]; B[mask1] = 0
    R[mask2] = 0;        G[mask2] = C[mask2]; B[mask2] = X[mask2]
    R[mask3] = 0;        G[mask3] = X[mask3]; B[mask3] = C[mask3]
    R[mask4] = X[mask4]; G[mask4] = 0;        B[mask4] = C[mask4]
    R[mask5] = C[mask5]; G[mask5] = 0;        B[mask5] = X[mask5]
    
    # Adjust for lightness
    m = L - C / 2
    return np.stack((R + m, G + m, B + m), axis=2)


def filter_h_channel(image):
    """
    Applies filtering to the hue channel of an HSL image.
    
    For hue values between 10 and 200 degrees, the hue is divided by 10,
    effectively compressing the hue range in that region.
    
    Args:
        image: Input HSL image (numpy array of shape H x W x 3)
    
    Returns:
        Filtered HSL image (same shape as input)
    """
    # Work on a copy to avoid mutating the input image
    H = np.copy(image[:, :, 0])
    
    # Apply filtering to hue values in the specified range
    mask = (H >= 10) & (H <= 200)
    H[mask] = H[mask] / 10
    
    # Create output image with filtered hue channel
    filtered_image = np.copy(image)
    filtered_image[:, :, 0] = H
    
    return filtered_image


def normalize_to_01(image):
    """
    Normalizes an image to the range [0, 1] using min-max scaling.
    
    Args:
        image: Input image (numpy array)
    
    Returns:
        Normalized image with values in [0, 1]
    """
    img_min = image.min()
    img_max = image.max()
    # Avoid division by zero for constant images
    if img_max == img_min:
        return np.zeros_like(image)
    return (image - img_min) / (img_max - img_min)


if __name__ == "__main__":
    # ========================================================================
    # 1. Load and display original image
    # ========================================================================
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planina-uint8.jpeg")
    
    img = imread(img_path)
    print(f"Original image - Shape: {img.shape}, Mean: {img.mean():.2f}, Std: {img.std():.2f}")
    
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()
    
    # ========================================================================
    # 2. Standardize the image (z-score normalization per channel)
    # ========================================================================
    std_image = standardize_image(img)
    print(f"Standardized image - Shape: {std_image.shape}, Mean: {std_image.mean():.2f}, Std: {std_image.std():.2f}")
    
    # Normalize to [0, 1] for visualization (standardized images have negative values)
    std_image_normalized = normalize_to_01(std_image)
    
    plt.imshow(std_image_normalized)
    plt.title("Standardized Image (Normalized for Display)")
    plt.axis("off")
    plt.show()
    
    # ========================================================================
    # 3. Convert RGB to HSL color space
    # ========================================================================
    
    hsl_image = rgb2hsl(std_image_normalized)
    
    plt.imshow(hsl_image)
    plt.title("HSL Image")
    plt.axis("off")
    plt.show()
    
    # ========================================================================
    # 4. Apply filtering to hue channel
    # ========================================================================
    hsl_image_filtered = filter_h_channel(hsl_image.copy())
    
    plt.imshow(hsl_image_filtered)
    plt.title("Filtered HSL Image (Hue Channel Modified)")
    plt.axis("off")
    plt.show()
    
    # ========================================================================
    # 5. Convert HSL back to RGB (original)
    # ========================================================================
    rgb_image = hsl2rgb(hsl_image)
    rgb_image = normalize_to_01(rgb_image)
    
    plt.imshow(rgb_image)
    plt.title("RGB Image (Reconstructed from HSL)")
    plt.axis("off")
    plt.show()
    
    # ========================================================================
    # 6. Convert filtered HSL back to RGB
    # ========================================================================
    rgb_image_filtered = hsl2rgb(hsl_image_filtered)
    rgb_image_filtered = normalize_to_01(rgb_image_filtered)
    
    plt.imshow(rgb_image_filtered)
    plt.title("RGB Image (Reconstructed from Filtered HSL)")
    plt.axis("off")
    plt.show()



