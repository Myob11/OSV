import numpy as np
from matplotlib import pyplot as plt
from OSV_lib import display_image, load_image, transformImage

def color2grayscale(iImage):
    # Convert input image to numpy array with float type for processing
    img = np.array(iImage, dtype=float)
    
    # If image has more than 3 channels (e.g., RGBA), keep only RGB channels
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    
    # Normalize image by subtracting minimum value to shift range to start at 0
    img -= img.min()
    
    # Scale pixel values to [0, 1] range by dividing by maximum value
    # Check to avoid division by zero
    if img.max() != 0:
        img = img / img.max()
    
    # Scale to [0, 255] range for standard uint8 representation
    img *= 255.0
    
    # Convert RGB to grayscale by averaging the 3 color channels for each pixel
    # Use floor to round down, then convert to uint8
    gray = np.floor(img.mean(axis=2)).astype(np.uint8)
  

    
    return gray

if __name__ == "__main__":
    path = r"primeri zagovora\Primer_zagovora2\data\paris_map-807-421.png"
    try:
        # first try custom loader (expects shape Y,X,C) and uint8
        rgb = load_image(path, size=[421, 807], type=np.uint8)
    except Exception:
        # fallback to matplotlib for PNG
        rgb = plt.imread(path)

    gray = color2grayscale(rgb)
    display_image(gray, "Grayscale Paris map", cmap="gray")
    
    # Get coordinates from user measurements
    print("Please click on two points in the image:")
    print("  First click: Point A (upper-left corner of the square)")
    print("  Second click: Point B (upper-right corner of the square)")
    pts = plt.ginput(2, timeout=-1)
    plt.close()
    
    A = (int(pts[0][0]), int(pts[0][1]))  # upper-left corner of the square
    B = (int(pts[1][0]), int(pts[1][1]))  # upper-right corner of the square
    print(f"Selected coordinates: A={A}, B={B}")

    vec = np.array(B) - np.array(A)
    phi = np.arctan2(vec[1], vec[0])      # angle of AB vs x-axis
    angle = -phi                          # rotate so AB becomes horizontal

    rotated = transformImage(
        "rotation_center",
        gray,
        (1, 1),           # pixel dimensions (dx, dy) - each pixel is 1x1 unit
        (angle, A),       # rotate about point A
        iBackground=0,
        iInterp=1         # bilinear
    )

    display_image(rotated, "Rotated grayscale Paris map", cmap="gray")
    plt.show()