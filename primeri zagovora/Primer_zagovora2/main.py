import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy import signal

sys.path.append(r"c:\Users\marti\Dropbox\Faks\mag\1.letnik\1.semester\Obdelava slik in videa\laboratorijske vaje")
from OSV_lib import transformImage

def color2grayscale(iImage):
    """
    Convert RGB color image to grayscale.
    
    Args:
        iImage: RGB color image (3D array)
    
    Returns:
        oImage: 2D grayscale image
    """
    # First, ensure the image intensities are in range [0, 255]
    # If image is in [0, 1] range (float), scale to [0, 255]
    if iImage.max() <= 1.0:
        iImage = iImage * 255
    
    # Calculate average of R, G, B components
    # Average across the color channel axis (axis=2)
    oImage = np.mean(iImage[:, :, :3], axis=2)
    
    # Round down to integer (floor)
    oImage = np.floor(oImage).astype(np.uint8)
    
    return oImage


# Load the input image
I = plt.imread(r"primeri zagovora\Primer_zagovora2\data\paris_map-807-421.png")

# Convert to grayscale
I_gray = color2grayscale(I)

# Display original color image
plt.figure()
plt.imshow(I)
plt.title('Original Color Image')
plt.axis('off')
plt.show()

# Display grayscale image
plt.figure()
plt.imshow(I_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

print(f"Original image shape: {I.shape}")
print(f"Grayscale image shape: {I_gray.shape}")
print(f"Grayscale value range: [{I_gray.min()}, {I_gray.max()}]")


# ========== Task 2: Align the square ==========

# --- Step 1: Interactively select points A and B ---

print("\nPlease select two points on the grayscale image:")
print("1. Click on the TOP-LEFT corner of the square (Point A)")
print("2. Click on the TOP-RIGHT corner of the square (Point B)")

plt.figure()
plt.imshow(I_gray, cmap='gray')
plt.title('Select Point A (top-left) and Point B (top-right)')
plt.axis('on') # Show axes to help with coordinate selection
points = plt.ginput(2, timeout=0) # Select 2 points, no timeout
plt.close()

if len(points) < 2:
    print("\nError: You did not select two points. Please run the script again.")
    # Using default points as a fallback
    A = np.array([160.0, 120.0])
    B = np.array([320.0, 180.0])
    print("Using default fallback points.")
else:
    A = np.array(points[0])
    B = np.array(points[1])

print(f"\nSelected Point A (x, y): ({A[0]:.2f}, {A[1]:.2f})")
print(f"Selected Point B (x, y): ({B[0]:.2f}, {B[1]:.2f})")

# Display the selected points on the image for verification
plt.figure()
plt.imshow(I_gray, cmap='gray')
plt.plot(A[0], A[1], 'r+', markersize=15, markeredgewidth=2, label='Point A')
plt.plot(B[0], B[1], 'g+', markersize=15, markeredgewidth=2, label='Point B')
plt.title('Verification of Selected Points')
plt.legend()
plt.show()


# --- Step 2: Calculate rotation angle ---

# Vector from point A to point B
vec_a = B - A
# Reference vector (unit vector along the x-axis)
vec_b = np.array([1, 0])

# Calculate the angle between vec_a and vec_b using the dot product
# a · b = |a| |b| cos(phi)  
dot_product = np.dot(vec_a, vec_b)
norm_a = np.linalg.norm(vec_a)
norm_b = np.linalg.norm(vec_b) # This is 1, but we include it for clarity

cos_phi = dot_product / (norm_a * norm_b)
phi = np.arccos(np.clip(cos_phi, -1.0, 1.0)) # Use clip for numerical stability

# The angle phi is the angle of the square's top edge relative to the x-axis.
# To align it with the x-axis, we need to rotate by -phi.
# We also need to determine the sign of the angle. The cross product can help.
# A 2D cross product (z-component) tells us the direction.
cross_product_z = vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]
if cross_product_z > 0:
    rotation_angle = -phi
else:
    rotation_angle = phi

print(f"\nAngle of the square's top edge: {np.degrees(phi):.2f} degrees")
print(f"Required rotation angle to align: {np.degrees(rotation_angle):.2f} degrees")


# --- Step 3: Rotate the image ---

# We will rotate the image around point A.
# The transformImage function from OSV_lib can be used for this.
# It requires the angle in radians, and the center of rotation.
rotation_center = A

# The 'iP' parameter for 'rotation_center' should be a tuple: (angle, center)
params = (rotation_angle, rotation_center)

# Perform the rotation. We'll use nearest neighbor for now (iInterp=0).
# Note: The provided transformImage function needs to be checked for linear interpolation (iInterp=1)
I_aligned = transformImage(
    iType='rotation_center',
    iImage=I_gray,
    iDim=(1, 1), # Assuming pixel dimensions are 1x1
    iP=params,
    iBackground=0, # Black background for areas with no data
    iInterp=0 # Using Nearest Neighbor interpolation
)


# --- Step 4: Display the aligned image ---

plt.figure()
plt.imshow(I_aligned, cmap='gray')
plt.title('Aligned Image')
plt.axis('on')
plt.show()


# ========== Task 3: Edge Detection ==========

# --- Step 1: Apply Sobel operator ---

# Define Sobel kernels
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Apply convolution to get gradients
grad_x = signal.convolve2d(I_aligned, sobel_x, mode='same', boundary='symm')
grad_y = signal.convolve2d(I_aligned, sobel_y, mode='same', boundary='symm')

# Calculate gradient magnitude
grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Normalize gradient magnitude to the range [0, 255]
grad_magnitude_normalized = (grad_magnitude / np.max(grad_magnitude)) * 255
grad_magnitude_normalized = grad_magnitude_normalized.astype(np.uint8)

# Display the gradient magnitude image
plt.figure()
plt.imshow(grad_magnitude_normalized, cmap='gray')
plt.title('Gradient Magnitude (Sobel Edges)')
plt.axis('off')
plt.show()


# --- Step 2: Threshold the edge image ---

# Choose a threshold value. This is often done by trial and error.
# A value around 50-100 is often a good starting point for images like this.
threshold = 75

# Apply thresholding
# Pixels with magnitude above the threshold are set to 255 (white), others to 0 (black).
I_edges = (grad_magnitude_normalized > threshold) * 255
I_edges = I_edges.astype(np.uint8)

# Display the final thresholded edge image
plt.figure()
plt.imshow(I_edges, cmap='gray')
plt.title(f'Thresholded Edges (Threshold = {threshold})')
plt.axis('off')
plt.show()


# ========== Task 4: Hough Transform for Square Detection ==========

def getSquareCenterPoint(iImage, iLength):
    """
    Finds the center of a square of a given side length using a Hough transform.

    Args:
        iImage (np.array): The binary edge image.
        iLength (float): The side length of the square to detect.

    Returns:
        tuple: A tuple containing:
            - oCenter (tuple): The (x, y) coordinates of the detected center.
            - oAcc (np.array): The Hough accumulator array.
    """
    height, width = iImage.shape
    oAcc = np.zeros((height, width), dtype=np.uint32)
    L_half = iLength / 2

    # Get coordinates of all edge pixels
    edge_pixels = np.argwhere(iImage > 0)

    # For each edge pixel, cast votes for possible center points
    for y, x in edge_pixels:
        # Hypothesis 1: The edge pixel is on the top or bottom side
        # Possible center x-coordinates
        x0_start = int(round(x - L_half))
        x0_end = int(round(x + L_half))
        
        # Vote for centers assuming pixel is on the top edge
        y0_top = int(round(y + L_half))
        if 0 <= y0_top < height:
            for x0 in range(max(0, x0_start), min(width, x0_end + 1)):
                oAcc[y0_top, x0] += 1
        
        # Vote for centers assuming pixel is on the bottom edge
        y0_bottom = int(round(y - L_half))
        if 0 <= y0_bottom < height:
            for x0 in range(max(0, x0_start), min(width, x0_end + 1)):
                oAcc[y0_bottom, x0] += 1

        # Hypothesis 2: The edge pixel is on the left or right side
        # Possible center y-coordinates
        y0_start = int(round(y - L_half))
        y0_end = int(round(y + L_half))

        # Vote for centers assuming pixel is on the left edge
        x0_left = int(round(x + L_half))
        if 0 <= x0_left < width:
            for y0 in range(max(0, y0_start), min(height, y0_end + 1)):
                oAcc[y0, x0_left] += 1
        
        # Vote for centers assuming pixel is on the right edge
        x0_right = int(round(x - L_half))
        if 0 <= x0_right < width:
            for y0 in range(max(0, y0_start), min(height, y0_end + 1)):
                oAcc[y0, x0_right] += 1

    # Find the maximum in the accumulator, which corresponds to the center
    max_vote_y, max_vote_x = np.unravel_index(np.argmax(oAcc), oAcc.shape)
    oCenter = (max_vote_x, max_vote_y)
    
    return oCenter, oAcc


# --- Step 1: Run the Hough transform ---

# The side length of the square is the distance between points A and B
square_side_length = norm_a
print(f"\nDetected square side length: {square_side_length:.2f} pixels")

# Find the center using the edge image and the calculated side length
center_coords, hough_accumulator = getSquareCenterPoint(I_edges, square_side_length)

print(f"Detected square center (x, y): ({center_coords[0]}, {center_coords[1]})")


# --- Step 2: Display the results ---

# Display the Hough accumulator
plt.figure()
plt.imshow(hough_accumulator, cmap='hot', aspect='auto')
plt.title('Hough Accumulator Space')
plt.colorbar(label='Votes')
plt.plot(center_coords[0], center_coords[1], 'c+', markersize=15, label='Detected Center')
plt.legend()
plt.show()

# Display the final result: square drawn on the aligned image
plt.figure()
plt.imshow(I_aligned, cmap='gray')
plt.title('Detected Square on Aligned Image')

# Draw the center point
plt.plot(center_coords[0], center_coords[1], 'r+', markersize=15, markeredgewidth=2, label='Detected Center')

# Calculate the corners of the square from the center and side length
L_half = square_side_length / 2
x0, y0 = center_coords
top_left = (x0 - L_half, y0 - L_half)
top_right = (x0 + L_half, y0 - L_half)
bottom_left = (x0 - L_half, y0 + L_half)
bottom_right = (x0 + L_half, y0 + L_half)

# Draw the sides of the square
plt.plot([top_left[0], top_right[0]], [top_left[1], top_right[1]], 'b-') # Top
plt.plot([bottom_left[0], bottom_right[0]], [bottom_left[1], bottom_right[1]], 'b-') # Bottom
plt.plot([top_left[0], bottom_left[0]], [top_left[1], bottom_left[1]], 'b-') # Left
plt.plot([top_right[0], bottom_right[0]], [top_right[1], bottom_right[1]], 'b-') # Right

plt.legend()
plt.axis('on')
plt.show()

