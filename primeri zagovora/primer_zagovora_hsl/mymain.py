import numpy as np
from matplotlib import pyplot as plt
from OSV_lib import display_image

#%%
if __name__ == "__main__":
    path = r"primeri zagovora\primer_zagovora_hsl\planina-uint8.jpeg"
    # need to load the image and display it, i dont know the size of it
    img = plt.imread(path)
    display_image(img, "Original Image")

    print("Image shape:", img.shape)

## naloga 1
def standardize_image(iImage):

    oImage = np.zeros_like(iImage)

    mean = np.mean(iImage)
    std = np.std(iImage)

    oImage = (iImage - mean) / std

    return oImage

if __name__ == "__main__":

    img_std = standardize_image(img)
    # i need to normalize the standardized image to the range [0, 1] for display purposes
    img_std_norm = (img_std - np.min(img_std)) / (np.max(img_std) - np.min(img_std))
    display_image(img_std_norm, "Standardized Image Normalized to [0, 1]")

#%%


## naloga 2

def rgb2hsl(iRGB):
   
    V = np.max(iRGB, axis=2)

    R = iRGB[:, :, 0]
    G = iRGB[:, :, 1]
    B = iRGB[:, :, 2]

    C = V - np.min(iRGB, axis=2)

    L = V - C / 2

    H = np.zeros_like(V)
    H[V == R] = (((G - B) / C))[V == R]
    H[V == G] = ((2 + (B - R) / C))[V == G]
    H[V == B] = ((4 + (R - G) / C))[V == B]
    H[C == 0] = 0
    H = H * 60
    
    S = np.zeros_like(V)
    # Avoid division by zero when lightness is 0 or 1
    valid_mask = (L != 0) & (L != 1)
    S[valid_mask] = (C / (1 - np.abs(2 * L - 1)))[valid_mask]

    oHSL = np.stack((H, S, L), axis=2)

    return oHSL


if __name__ == "__main__":
    img_hsl = rgb2hsl(img_std_norm)
    display_image(img_hsl, "HSL Image")

## naloga 4
def hsl2rgb(iHSL):

    H = iHSL[:, :, 0]
    S = iHSL[:, :, 1]
    L = iHSL[:, :, 2]
    C = (1 - np.abs(2 * L - 1)) * S

    # Initialize RGB channels
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    H_hat = H / 60

    X = C * (1 - np.abs((H_hat % 2) - 1))

    m = L - (C / 2)

    oRGB = np.zeros_like(iHSL)

    # conditions for H_hat ranges

    maska1 = (0 <= H_hat) & (H_hat < 1)
    maska2 = (1 <= H_hat) & (H_hat < 2)
    maska3 = (2 <= H_hat) & (H_hat < 3)
    maska4 = (3 <= H_hat) & (H_hat < 4)
    maska5 = (4 <= H_hat) & (H_hat < 5)
    maska6 = (5 <= H_hat) & (H_hat < 6)

    R[maska1] = C[maska1]; G[maska1] = X[maska1]; B[maska1] = 0
    R[maska2] = X[maska2]; G[maska2] = C[maska2]; B[maska2] = 0
    R[maska3] = 0; G[maska3] = C[maska3]; B[maska3] = X[maska3]
    R[maska4] = 0; G[maska4] = X[maska4]; B[maska4] = C[maska4]
    R[maska5] = X[maska5]; G[maska5] = 0; B[maska5] = C[maska5]
    R[maska6] = C[maska6]; G[maska6] = 0; B[maska6] = X[maska6]

    oRGB[:, :, 0] = R + m
    oRGB[:, :, 1] = G + m
    oRGB[:, :, 2] = B + m

    
    return oRGB

if __name__ == "__main__":
    img_rgb_converted = hsl2rgb(img_hsl)
    img_rgb_norm = (img_rgb_converted - np.min(img_rgb_converted)) / (
        np.max(img_rgb_converted) - np.min(img_rgb_converted)
    )
    display_image(img_rgb_norm, "Converted back to RGB Image")

## naloga 3


if __name__ == "__main__":
    # H slice transform: shrink hues in [10, 200) by factor 10
    h_slice = img_hsl[:, :, 0].copy()
    mask = (h_slice >= 10) & (h_slice < 200)
    h_slice[mask] = h_slice[mask] / 10.0

    img_hsl_transformed = np.stack((h_slice, img_hsl[:, :, 1], img_hsl[:, :, 2]), axis=2)
    hsl_trans_display = np.stack(
        (
            img_hsl_transformed[:, :, 0] / 360.0,
            img_hsl_transformed[:, :, 1],
            img_hsl_transformed[:, :, 2],
        ),
        axis=2,
    )
    display_image(hsl_trans_display, "Transformed HSL (H scaled to 0-1)")

    # the filtered hsl convert back to rgb and display
    img_rgb_transformed = hsl2rgb(img_hsl_transformed)
    img_rgb_transformed_norm = (img_rgb_transformed - np.min(img_rgb_transformed)) / (
        np.max(img_rgb_transformed) - np.min(img_rgb_transformed)
    )
    display_image(img_rgb_transformed_norm, "Transformed RGB Image after HSL Modification")
    



