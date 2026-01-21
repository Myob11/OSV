import numpy as np
from matplotlib import pyplot as plt
from OSV_lib import display_image, load_image
from scipy import ndimage

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

def get_coordinates_interactively(image):
    """
    Interactive coordinate picker for selecting points A and B on the image.
    Click on two points: first point A (upper-left corner), then point B (upper-right corner).
    Press Enter after selecting both points to continue.
    """
    coords = []
    
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            coords.append((x, y))
            
            # Plot the clicked point
            ax.plot(x, y, 'rx', markersize=10, markeredgewidth=2)
            
            # Label the point
            if len(coords) == 1:
                ax.text(x, y, f'  A({x},{y})', color='red', fontsize=10)
            elif len(coords) == 2:
                ax.text(x, y, f'  B({x},{y})', color='red', fontsize=10)
                
            fig.canvas.draw()
            
            # Print to console
            print(f"Point {'A' if len(coords) == 1 else 'B'} selected: ({x}, {y})")
            
            # Close after selecting both points
            if len(coords) == 2:
                print("Both points selected. Close the window or press Enter to continue...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title("Click to select coordinates:\n1. Point A (upper-left corner)\n2. Point B (upper-right corner)")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    
    # Connect the click event
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.show()
    
    # Return the coordinates or default values if not enough points selected
    if len(coords) >= 2:
        return coords[0], coords[1]
    else:
        print("Warning: Not enough points selected. Using default coordinates.")
        return (100, 80), (300, 90)

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

    # Interactive coordinate selection - click on points A and B on the image
    print("\nSelect coordinates by clicking on the image:")
    print("1. Click on point A (upper-left corner of the square)")
    print("2. Click on point B (upper-right corner of the square)")
    A, B = get_coordinates_interactively(gray)

    vec = np.array(B) - np.array(A)
    phi = np.arctan2(vec[1], vec[0])      # angle of AB vs x-axis
    angle_deg = -phi * 180 / np.pi        # rotate so AB becomes horizontal (convert to degrees)

    # Rotate the image using scipy's ndimage.rotate
    # Note: scipy rotates around the center by default
    rotated = ndimage.rotate(gray, angle_deg, reshape=False, order=1, cval=0)

    display_image(rotated, "Rotated grayscale Paris map", cmap="gray")
    plt.show()