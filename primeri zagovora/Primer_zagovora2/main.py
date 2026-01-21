import numpy as np
from matplotlib import pyplot as plt
from OSV_lib import display_image, load_image

def color2grayscale(iImage):
    # Convert to grayscale by averaging RGB channels
    oImage = np.mean(iImage, axis=2).astype(np.uint8)
    return oImage

display_image(I, "Original Image")
gray_image = color2grayscale(I)
display_image(gray_image, "Grayscale Image")

def get_rotation_angle(point_A, point_B):
    """
    Calculate rotation angle between vector AB and x-axis.
    
    Args:
        point_A: [x, y] - top-left corner of square
        point_B: [x, y] - top-right corner of square
    
    Returns:
        angle in degrees
    """
    # Vector from A to B
    vector_a = np.array([point_B[0] - point_A[0], point_B[1] - point_A[1]], dtype=float)
    # Unit vector along x-axis
    vector_b = np.array([1, 0], dtype=float)
    
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
    plt.show()

 # TODO: set these from measured coordinates (x, y)
    A = (100, 80)  # upper-left corner of the square
    B = (300, 90)  # upper-right corner of the square

    vec = np.array(B) - np.array(A)
    phi = np.arctan2(vec[1], vec[0])      # angle of AB vs x-axis
    angle = -phi                          # rotate so AB becomes horizontal

    rotated = transformImage(
        "rotation_center",
        gray,
        gray.shape,
        (angle, A),       # rotate about point A
        iBackground=0,
        iInterp=1         # bilinear
    )

    display_image(rotated, "Rotated grayscale Paris map", cmap="gray")
    plt.show()