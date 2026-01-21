import numpy as np
from matplotlib import pyplot as plt
from OSV_lib import display_image, load_image

path = r"primeri zagovora\Primer_zagovora1\data\rose-366-366-08bit.raw"
I = load_image(path, [366, 366], np.uint8)
display_image(I, "Original Image")


def getBoundaryIndices(iImage , iAxis):

    # naredi mi for loop ki gre cez sliko in najde meje, ozadje je bele barve (255)
    if iAxis == 1:  # x-axis (columns) - išči levi in desni stolpec
        for j in range(iImage.shape[1]):
            col = iImage[:, j]
            if np.any(col < 255):
                oIdx1 = j
                break
        
        for j in range(iImage.shape[1] - 1, -1, -1):
            col = iImage[:, j]
            if np.any(col < 255):
                oIdx2 = j
                break
                
    elif iAxis == 2:  # y-axis (rows) - išči zgornjo in spodnjo vrstico
        for i in range(iImage.shape[0]):
            row = iImage[i, :]
            if np.any(row < 255):
                oIdx1 = i
                break
        
        for i in range(iImage.shape[0] - 1, -1, -1):
            row = iImage[i, :]
            if np.any(row < 255):
                oIdx2 = i
                break

    return oIdx1 , oIdx2


def expandImage(iImage):
    """
    Expand the spatial domain of the image.
    Input image of size X_c × Y_c is placed in the center of upper part of output image (2Y_c × 2Y_c)
    """
    Yc, Xc = iImage.shape
    
    # Create output image of size 2*Yc × 2*Yc filled with white (255)
    oImage = np.ones((2*Yc, 2*Yc), dtype=np.uint8) * 255
    
    # Place input image in the center of the upper part
    # Center horizontally and at the top
    start_col = (2*Yc - Xc) // 2
    start_row = 0
    
    oImage[start_row:start_row+Yc, start_col:start_col+Xc] = iImage
    
    return oImage


def createRotatedPattern(iImage, iAngle):
    """
    Create a circular pattern by rotating and repeating the image around its center.
    """
    from scipy import ndimage
    
    # Calculate number of repetitions
    num_repeats = int(np.round(360 / iAngle))
    actual_angle = 360 / num_repeats
    
    print(f"Angle threshold: {iAngle:.2f}°")
    print(f"Number of repetitions: {num_repeats}")
    print(f"Actual angle between repetitions: {actual_angle:.2f}°")
    
    # Initialize output image as float to handle accumulation
    oImage = iImage.astype(np.float32).copy()
    
    # Create rotated copies and add them
    for i in range(1, num_repeats):
        rotation_angle = i * actual_angle
        rotated = ndimage.rotate(iImage.astype(np.float32), rotation_angle, reshape=False, order=1, cval=255)
        oImage += rotated
    
    # Normalize to [0, 255]
    # Divide by number of repetitions to get average
    oImage = oImage / num_repeats
    
    # Clip to valid range
    oImage = np.clip(oImage, 0, 255).astype(np.uint8)
    
    return oImage


def getPointsAndAngle(iImage):
    """
    Find points S, L, D and angle φ in the image.
    """
    # Find S: point at the middle of stem in the last row
    last_row = iImage.shape[0] - 1
    non_bg_indices_last = np.where(iImage[last_row, :] < 255)[0]
    if len(non_bg_indices_last) > 0:
        S = (last_row, int((non_bg_indices_last[0] + non_bg_indices_last[-1]) // 2))
    
    # Find L and D: points at Y_c/4 height
    quarter_row = iImage.shape[0] // 4
    non_bg_indices_quarter = np.where(iImage[quarter_row, :] < 255)[0]
    
    if len(non_bg_indices_quarter) > 0:
        L = (quarter_row, int(non_bg_indices_quarter[0]))  # left point
        D = (quarter_row, int(non_bg_indices_quarter[-1]))  # right point
        
        # Calculate angle φ
        # Vectors from S to L and S to D
        a = np.array([L[1] - S[1], L[0] - S[0]])  # (x, y) format
        b = np.array([D[1] - S[1], D[0] - S[0]])
        
        # Calculate angle using dot product
        cos_phi = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        phi = np.arccos(np.clip(cos_phi, -1, 1)) * 180 / np.pi
        
        return S, L, D, phi
    
    return None, None, None, None


if __name__ == "__main__":
    # Step 1: Find boundaries
    col_start, col_end = getBoundaryIndices(I, 1)
    row_start, row_end = getBoundaryIndices(I, 2)
    
    print("Row boundaries:", row_start, row_end)
    print("Column boundaries:", col_start, col_end)
    
    # Step 1: Crop the image
    cI = I[row_start:row_end+1, col_start:col_end+1]
    display_image(cI, "Cropped Image (cI)")
    
    # Step 2: Find points S, L, D and angle φ
    S, L, D, phi = getPointsAndAngle(cI)
    print(f"S (stem center): {S}")
    print(f"L (left point): {L}")
    print(f"D (right point): {D}")
    print(f"Angle φ: {phi:.2f}°")
    
    # Step 3: Expand the spatial domain
    sI = expandImage(cI)
    display_image(sI, "Expanded Image (sI)")
    
    # Step 4: Create circular pattern
    angle_threshold = phi  # Use the calculated angle φ
    final_image = createRotatedPattern(sI, angle_threshold)
    display_image(final_image, "Circular Pattern of Roses")