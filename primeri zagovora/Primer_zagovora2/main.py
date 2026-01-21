import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from OSV_lib import houghTransform2D2P

I = plt.imread(r"primeri zagovora\Primer_zagovora2\data\paris_map-807-421.png")

# Convert to [0, 255] range if it's in [0, 1]
if I.max() <= 1.0:
    I = (I * 255).astype(np.uint8)

def display_image(iImage, title="Image"):
    plt.figure()
    # Use cmap='gray' for grayscale images
    if len(iImage.shape) == 2:  # Grayscale
        plt.imshow(iImage, cmap='gray')
    else:  # RGB
        plt.imshow(iImage)
    plt.title(title)
    plt.axis('on')
    plt.show()

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
    
    # Calculate angle using atan2 for proper quadrant handling
    angle_rad = np.arctan2(vector_a[1], vector_a[0])
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def rotate_image_around_point(iImage, rotation_center, angle, bg_value=0):
    """
    Rotate image around a specified point using linear interpolation.
    
    Args:
        iImage: Input image
        rotation_center: [x, y] center of rotation
        angle: Rotation angle in degrees (counter-clockwise)
        bg_value: Background value (default 0)
    
    Returns:
        Rotated image
    """
    height, width = iImage.shape[:2]
    oImage = np.full_like(iImage, bg_value, dtype=np.float32)
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    cx, cy = rotation_center
    
    # Iterate through output image coordinates
    for y in range(height):
        for x in range(width):
            # Translate to rotation center
            dx = x - cx
            dy = y - cy
            
            # Rotate backwards to find source pixel
            src_x = cos_a * dx + sin_a * dy + cx
            src_y = -sin_a * dx + cos_a * dy + cy
            
            # Bilinear interpolation
            if 0 <= src_x < width - 1 and 0 <= src_y < height - 1:
                x0, y0 = int(src_x), int(src_y)
                x1, y1 = x0 + 1, y0 + 1
                
                fx = src_x - x0
                fy = src_y - y0
                
                # Get the four neighboring pixels
                if len(iImage.shape) == 2:  # Grayscale
                    v00 = iImage[y0, x0].astype(float)
                    v10 = iImage[y0, x1].astype(float)
                    v01 = iImage[y1, x0].astype(float)
                    v11 = iImage[y1, x1].astype(float)
                else:  # Color
                    v00 = iImage[y0, x0].astype(float)
                    v10 = iImage[y0, x1].astype(float)
                    v01 = iImage[y1, x0].astype(float)
                    v11 = iImage[y1, x1].astype(float)
                
                # Bilinear interpolation
                v0 = v00 * (1 - fx) + v10 * fx
                v1 = v01 * (1 - fx) + v11 * fx
                interpolated = v0 * (1 - fy) + v1 * fy
                
                oImage[y, x] = interpolated
    
    return oImage.astype(iImage.dtype)

def detect_edges_sobel(iImage, threshold=20):
    """
    Detect edges using Sobel operator with provided kernels.
    
    Args:
        iImage: Input grayscale image
        threshold: Threshold value for edge detection [0, 255]
    
    Returns:
        edges: Binary edge image
        gradient_magnitude: Gradient magnitude normalized to [0, 255]
    """
    # Sobel kernels
    Gx = np.array([[-1, 0, +1],
                   [-2, 0, +2],
                   [-1, 0, +1]], dtype=np.float32)
    
    Gy = np.array([[+1, +2, +1],
                   [0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)
    
    # Apply convolution
    gradient_x = ndimage.convolve(iImage.astype(float), Gx)
    gradient_y = ndimage.convolve(iImage.astype(float), Gy)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize to [0, 255]
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    
    # Apply threshold
    edges = (gradient_magnitude > threshold).astype(np.uint8) * 255
    
    return edges, gradient_magnitude

def getCenterPoint(iAcc, iThreshold=0.5):
    """
    Get center point from Hough accumulator with maximum value.
    
    Args:
        iAcc: Hough accumulator array
        iThreshold: Threshold as fraction of max value (not used, kept for compatibility)
    
    Returns:
        [x, y] coordinates of center point with max accumulation
    """
    max_val = np.max(iAcc)
    max_pos = np.where(iAcc == max_val)
    
    y = max_pos[0][0]
    x = max_pos[1][0]
    
    return np.array([x, y]), max_val

def getSquareCenterPoint(iImage, iLength):
    """
    Detect square center using Hough transform for lines.
    
    Args:
        iImage: Binary edge image
        iLength: Side length of square to detect
    
    Returns:
        oCenter: [x, y] coordinates of square center
        oAcc: Hough accumulator array
    """
    # Apply Hough transform to detect lines
    oAcc, rangeR, rangeFi = houghTransform2D2P(iImage, stepR=1, stepF=1)
    
    # Find peaks in accumulator (lines with most votes)
    # The center of the square should be where 4 lines intersect
    # For simplicity, find the point with maximum accumulation
    oCenter, _ = getCenterPoint(oAcc)
    
    return oCenter, oAcc

def draw_square(iImage, center, side_length):
    """
    Draw square on image.
    
    Args:
        iImage: Input image to draw on
        center: [x, y] center coordinates
        side_length: Side length of square
    
    Returns:
        Image with drawn square
    """
    oImage = iImage.copy()
    if len(oImage.shape) == 2:
        oImage = np.stack([oImage]*3, axis=-1)
    
    half_len = side_length / 2
    cx, cy = int(center[0]), int(center[1])
    
    # Calculate corner coordinates
    x1, y1 = int(cx - half_len), int(cy - half_len)  # top-left
    x2, y2 = int(cx + half_len), int(cy - half_len)  # top-right
    x3, y3 = int(cx + half_len), int(cy + half_len)  # bottom-right
    x4, y4 = int(cx - half_len), int(cy + half_len)  # bottom-left
    
    # Draw square edges in red
    oImage[y1:y2+1, x1, :] = [255, 0, 0]  # left
    oImage[y1:y2+1, x2, :] = [255, 0, 0]  # right
    oImage[y1, x1:x3+1, :] = [255, 0, 0]  # top
    oImage[y3, x1:x3+1, :] = [255, 0, 0]  # bottom
    
    # Draw center point
    oImage[cy-2:cy+3, cx-2:cx+3, :] = [255, 0, 0]
    
    return oImage

# Task 2: Align the image with interactive point selection
points = []

def onclick(event):
    """Handle mouse click events on the image"""
    if event.xdata is not None and event.ydata is not None:
        points.append([event.xdata, event.ydata])
        print(f"Point {len(points)}: ({event.xdata:.1f}, {event.ydata:.1f})")
        
        if len(points) == 2:
            print("Both points selected. Close the window to continue.")
            plt.close()

fig, ax = plt.subplots()
ax.imshow(gray_image, cmap='gray')
ax.set_title("Click on point A (top-left), then point B (top-right)")
ax.axis('on')

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

if len(points) >= 2:
    point_A = points[0]
    point_B = points[1]
    
    print(f"Point A: {point_A}")
    print(f"Point B: {point_B}")
    
    angle = get_rotation_angle(point_A, point_B)
    print(f"Rotation angle: {angle} degrees")
    
    aligned_image = rotate_image_around_point(gray_image, point_A, -angle, bg_value=0)
    display_image(aligned_image, "Aligned Image")
    
    threshold = 100
    # Task 3: Edge detection
    edges, gradient = detect_edges_sobel(aligned_image, threshold=threshold)
    
    display_image(gradient, "Gradient Magnitude")
    display_image(edges, f"Detected Edges (threshold={threshold})")
    
    # Task 4: Hough transform for square detection
    square_length = int(np.linalg.norm(np.array(point_B) - np.array(point_A)))
    print(f"Estimated square side length: {square_length}")
    
    center, accumulator = getSquareCenterPoint(edges, square_length)
    
    print(f"Square center: {center}")
    
    # Display accumulator
    display_image(accumulator, "Hough Accumulator")
    
    # Draw square on aligned image
    result_image = draw_square(aligned_image, center, square_length)
    display_image(result_image, "Konční rezultat")
    
else:
    print("Not enough points selected.")


