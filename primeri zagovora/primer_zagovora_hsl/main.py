#%%
import matplotlib.pyplot as plt
import numpy as np


path = r"primeri zagovora\primer_zagovora_hsl\planina-uint8.jpeg"
def standardize_image(iImage):
    """
    Standardizes the intensity of an RGB image.

    Args:
        iImage: Input RGB image.

    Returns:
        Standardized RGB image.
    """
    # Convert image to float type for calculation
    oImage = iImage.astype(np.float64)
    
    # Standardize each channel separately
    for i in range(oImage.shape[2]):
        channel = oImage[:, :, i]
        mean = np.mean(channel)
        std = np.std(channel)
        # Avoid division by zero if a channel is uniform
        if std > 0:
            oImage[:, :, i] = (channel - mean) / std
        else:
            oImage[:, :, i] = channel - mean
        
    return oImage

# 1. Load the image
try:
    # The user wants to use plt.imread
        img = plt.imread(path)
except FileNotFoundError:
    print("Error: 'planina-uint8.jpeg' not found. Make sure the image is in the same directory as the script.")
    exit()

# Standardize the input image
img_std = standardize_image(img)

# Display the standardized image
# For visualization, we scale the image to the [0, 1] range
img_display = (img_std - np.min(img_std)) / (np.max(img_std) - np.min(img_std))
# Guard against tiny floating point overshoots
img_display = np.clip(img_display, 0.0, 1.0)

plt.figure(figsize=(6, 5))
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(img_display)
plt.title('Standardized Image')
plt.axis('off')
plt.show()

def rgb2hsl(iRGB):
    """
    Converts an RGB image to HSL.
    """
    iRGB = (iRGB - np.min(iRGB)) / (np.max(iRGB) - np.min(iRGB))
    oHSL = np.zeros_like(iRGB, dtype=np.float64)
    
    r = iRGB[:, :, 0]
    g = iRGB[:, :, 1]
    b = iRGB[:, :, 2]
    
    v = np.max(iRGB, axis=2)
    c = v - np.min(iRGB, axis=2)
    
    l = (v + np.min(iRGB, axis=2)) / 2

    s = np.zeros_like(l)
    # Avoid division by zero
    mask = (l > 0) & (l < 1)
    s[mask] = c[mask] / (1 - np.abs(2 * l[mask] - 1))

    h = np.zeros_like(l)
    
    # Hue calculation
    mask_c_zero = c == 0
    h[mask_c_zero] = 0

    mask_v_is_r = (c != 0) & (v == r)
    h[mask_v_is_r] = 60 * ((g[mask_v_is_r] - b[mask_v_is_r]) / c[mask_v_is_r])

    mask_v_is_g = (c != 0) & (v == g)
    h[mask_v_is_g] = 60 * (2 + (b[mask_v_is_g] - r[mask_v_is_g]) / c[mask_v_is_g])

    mask_v_is_b = (c != 0) & (v == b)
    h[mask_v_is_b] = 60 * (4 + (r[mask_v_is_b] - g[mask_v_is_b]) / c[mask_v_is_b])

    # Normalize h to [0, 360]
    h[h < 0] += 360

    oHSL[:, :, 0] = h
    oHSL[:, :, 1] = s
    oHSL[:, :, 2] = l
    
    return oHSL

def hsl2rgb(iHSL):
    """Converts an HSL image (H in degrees, S/L in [0,1]) back to RGB in [0,1]."""
    h = iHSL[:, :, 0] / 60.0  # scale hue to [0,6)
    s = iHSL[:, :, 1]
    l = iHSL[:, :, 2]

    c = (1 - np.abs(2 * l - 1)) * s
    x = c * (1 - np.abs((h % 2) - 1))
    m = l - c / 2

    r1 = np.zeros_like(h)
    g1 = np.zeros_like(h)
    b1 = np.zeros_like(h)

    mask = (h >= 0) & (h < 1)
    r1[mask], g1[mask], b1[mask] = c[mask], x[mask], 0

    mask = (h >= 1) & (h < 2)
    r1[mask], g1[mask], b1[mask] = x[mask], c[mask], 0

    mask = (h >= 2) & (h < 3)
    r1[mask], g1[mask], b1[mask] = 0, c[mask], x[mask]

    mask = (h >= 3) & (h < 4)
    r1[mask], g1[mask], b1[mask] = 0, x[mask], c[mask]

    mask = (h >= 4) & (h < 5)
    r1[mask], g1[mask], b1[mask] = x[mask], 0, c[mask]

    mask = (h >= 5) & (h < 6)
    r1[mask], g1[mask], b1[mask] = c[mask], 0, x[mask]

    oRGB = np.zeros_like(iHSL)
    oRGB[:, :, 0] = r1 + m
    oRGB[:, :, 1] = g1 + m
    oRGB[:, :, 2] = b1 + m

    return np.clip(oRGB, 0.0, 1.0)

# Convert the standardized image to HSL
img_hsl = rgb2hsl(img_std)

# Transform hue channel: scale values in [10,200) by 1/10
h_slice = img_hsl[:, :, 0].copy()
mask_h = (h_slice >= 10) & (h_slice < 200)
h_slice[mask_h] = h_slice[mask_h] / 10.0

img_hsl_transformed = img_hsl.copy()
img_hsl_transformed[:, :, 0] = h_slice

# For visualization, we need to scale the HSL image channels to the [0, 1] range
img_hsl_display = np.zeros_like(img_hsl)
img_hsl_display[:, :, 0] = img_hsl[:, :, 0] / 360.0  # Hue is in [0, 360]
img_hsl_display[:, :, 1] = img_hsl[:, :, 1]          # Saturation is already [0, 1]
img_hsl_display[:, :, 2] = img_hsl[:, :, 2]          # Lightness is already [0, 1]
img_hsl_display = np.clip(img_hsl_display, 0.0, 1.0)

plt.figure(figsize=(6, 5))
plt.imshow(img_hsl_display)
plt.title('HSL Image')
plt.axis('off')
plt.show()

# Display transformed HSL image
img_hsl_transformed_display = np.zeros_like(img_hsl_transformed)
img_hsl_transformed_display[:, :, 0] = img_hsl_transformed[:, :, 0] / 360.0
img_hsl_transformed_display[:, :, 1] = img_hsl_transformed[:, :, 1]
img_hsl_transformed_display[:, :, 2] = img_hsl_transformed[:, :, 2]
img_hsl_transformed_display = np.clip(img_hsl_transformed_display, 0.0, 1.0)

plt.figure(figsize=(6, 5))
plt.imshow(img_hsl_transformed_display)
plt.title('Transformed HSL Image')
plt.axis('off')
plt.show()

# Convert back to RGB and display
img_rgb_from_hsl = hsl2rgb(img_hsl)
img_rgb_from_hsl_transformed = hsl2rgb(img_hsl_transformed)

plt.figure(figsize=(6, 5))
plt.imshow(img_rgb_from_hsl)
plt.title('RGB from HSL')
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(img_rgb_from_hsl_transformed)
plt.title('RGB from Transformed HSL')
plt.axis('off')
plt.show()


# %%