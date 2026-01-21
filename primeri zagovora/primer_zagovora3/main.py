import numpy as np
from matplotlib import pyplot as plt


I = plt.imread(r"primeri zagovora\primer_zagovora3\travnik-uint8.jpeg")
plt.imshow(I)
plt.axis('on')
plt.show()


def normalizeImage(iImage):
    """
    Normalize image intensities from arbitrary range [Imin, Imax] to [0, 1].
    
    Args:
        iImage: Input RGB image
    
    Returns:
        oImage: Normalized image with intensities in [0, 1]
    """
    # Convert to float to preserve precision
    iImage_float = iImage.astype(np.float32)
    
    # Get min and max values
    Imin = np.min(iImage_float)
    Imax = np.max(iImage_float)
    
    # Normalize to [0, 1]
    if Imax - Imin == 0:
        oImage = np.zeros_like(iImage_float)
    else:
        oImage = (iImage_float - Imin) / (Imax - Imin)
    
    return oImage


img_normalized = normalizeImage(I)
plt.figure()
plt.imshow(img_normalized)
plt.axis('on')
plt.title('Normalized Image')
plt.show()




def rgb2hsv(iRGB):

    R = iRGB[:, :, 0]
    G = iRGB[:, :, 1]
    B = iRGB[:, :, 2]

    Cmax = np.max(iRGB, axis=2)
    Cmin = np.min(iRGB, axis=2)

    C = Cmax - Cmin
    V = np.max(iRGB, axis=2)

    L = V - (C / 2)

    H = np.zeros_like(V)
    
    # Only compute H where C != 0 to avoid division by zero
    valid_c = C != 0
    H[valid_c & (V == R)] = 60 * (G - B)[valid_c & (V == R)] / C[valid_c & (V == R)]
    H[valid_c & (V == G)] = 60 * (2 + (B - R)[valid_c & (V == G)] / C[valid_c & (V == G)])
    H[valid_c & (V == B)] = 60 * (4 + (R - G)[valid_c & (V == B)] / C[valid_c & (V == B)])

    S = np.zeros_like(V)
    valid_mask = (V != 0)
    S[valid_mask] = (C / V)[valid_mask]
    oHSV = np.stack((H, S, V), axis=2)

    return oHSV

img_hsv = rgb2hsv(img_normalized)
# Normalize HSV for display
img_hsv_display = normalizeImage(img_hsv)
plt.figure()
plt.imshow(img_hsv_display)
plt.axis('on')
plt.title('HSV Image')
plt.show()

# Extract H channel from HSV image
h_slice = img_hsv[:, :, 0]

# Divide values less than 100 by 2
h_slice[h_slice < 100] = h_slice[h_slice < 100] / 2

# Create new HSV image with modified H channel
img_hsv_transformed = img_hsv.copy()
img_hsv_transformed[:, :, 0] = h_slice

# Normalize transformed HSV for display
img_hsv_transformed_display = normalizeImage(img_hsv_transformed)

# Display transformed HSV image
plt.figure()
plt.imshow(img_hsv_transformed_display)
plt.axis('on')
plt.title('Transformed HSV Image')
plt.show()

def hsv2rgb(iHSV):
    """
    Convert HSV image to RGB.
    
    Args:
        iHSV: Input HSV image with H, S, V channels
    
    Returns:
        oRGB: Output RGB image
    """
    H = iHSV[:, :, 0]
    S = iHSV[:, :, 1]
    V = iHSV[:, :, 2]
    
    C = V * S
    H_prime = H / 60
    X = C * (1 - np.abs(np.mod(H_prime, 2) - 1))
    
    R_prime = np.zeros_like(V)
    G_prime = np.zeros_like(V)
    B_prime = np.zeros_like(V)
    
    mask0 = (H_prime >= 0) & (H_prime < 1)
    mask1 = (H_prime >= 1) & (H_prime < 2)
    mask2 = (H_prime >= 2) & (H_prime < 3)
    mask3 = (H_prime >= 3) & (H_prime < 4)
    mask4 = (H_prime >= 4) & (H_prime < 5)
    mask5 = (H_prime >= 5) & (H_prime < 6)
    
    R_prime[mask0] = C[mask0]
    G_prime[mask0] = X[mask0]
    
    R_prime[mask1] = X[mask1]
    G_prime[mask1] = C[mask1]
    
    G_prime[mask2] = C[mask2]
    B_prime[mask2] = X[mask2]
    
    G_prime[mask3] = X[mask3]
    B_prime[mask3] = C[mask3]
    
    R_prime[mask4] = X[mask4]
    B_prime[mask4] = C[mask4]
    
    R_prime[mask5] = C[mask5]
    B_prime[mask5] = X[mask5]
    
    m = V - C
    
    R = R_prime + m
    G = G_prime + m
    B = B_prime + m
    
    oRGB = np.stack((R, G, B), axis=2)
    
    return oRGB


# Test on original HSV image
rgb_from_hsv = hsv2rgb(img_hsv)
plt.figure()
plt.imshow(rgb_from_hsv)
plt.axis('on')
plt.title('RGB from HSV')
plt.show()

# Test on transformed HSV image
rgb_from_hsv_transformed = hsv2rgb(img_hsv_transformed)
plt.figure()
plt.imshow(rgb_from_hsv_transformed)
plt.axis('on')
plt.title('RGB from Transformed HSV')
plt.show()

# Compare original and transformed result side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(I)
axes[0].set_title('Original Image')
axes[0].axis('on')

axes[1].imshow(rgb_from_hsv_transformed)
axes[1].set_title('Transformed Image')
axes[1].axis('on')

plt.tight_layout()
plt.show()