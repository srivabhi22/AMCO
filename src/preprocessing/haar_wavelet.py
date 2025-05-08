import cv2
import numpy as np
import time
import sys

def get_haar_wavelet(src):
    """
    Perform Haar Wavelet Transform on the input image
    
    Args:
        src (numpy.ndarray): Input image (float32)
    
    Returns:
        numpy.ndarray: Transformed image
    """
    height, width = src.shape
    dst = np.zeros((height, width), dtype=np.float32)

    # Horizontal transform
    horizontal = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width // 2):
            mean_pixel = (src[i, 2*j] + src[i, 2*j+1]) / 2
            horizontal[i, j] = mean_pixel
            horizontal[i, j + width//2] = src[i, 2*j] - mean_pixel

    # Vertical transform
    for i in range(height // 2):
        for j in range(width):
            mean_pixel = (horizontal[2*i, j] + horizontal[2*i+1, j]) / 2
            dst[i, j] = mean_pixel
            dst[i + height//2, j] = horizontal[2*i, j] - mean_pixel

    return dst

def get_emax(src, scale):
    """
    Compute max values in scaled regions
    
    Args:
        src (numpy.ndarray): Input image
        scale (int): Scaling factor
    
    Returns:
        numpy.ndarray: Max values in scaled regions
    """
    height, width = src.shape
    h_scaled = height // scale
    w_scaled = width // scale
    dst = np.zeros((h_scaled, w_scaled), dtype=np.float32)

    for i in range(h_scaled):
        for j in range(w_scaled):
            region = src[scale*i:scale*(i+1), scale*j:scale*(j+1)]
            dst[i, j] = np.max(region)

    return dst

def detect_blur(image_path, threshold=35, min_zero=0.05):
    """
    Detect blur in an image using Haar Wavelet Transform
    
    Args:
        image_path (str): Path to the input image
        threshold (float): Edge detection threshold
        min_zero (float): Minimum zero threshold for clear image detection
    
    Returns:
        dict: Blur detection results
    """
    # Start time
    start_time = time.time()

    # Read image
    img0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height0, width0 = img0.shape
    img0 = img0.astype(np.float32)

    # Pad image to multiple of 16
    height = int(np.ceil(height0 / 16) * 16)
    width = int(np.ceil(width0 / 16) * 16)
    img = np.zeros((height, width), dtype=np.float32)
    img[:height0, :width0] = img0

    # Haar Wavelet Transform
    level1 = get_haar_wavelet(img)
    level2 = get_haar_wavelet(level1[:height//2, :width//2])
    level3 = get_haar_wavelet(level2[:height//4, :width//4])

    # Compute energy maps
    HL1 = level1[height//2:, :width//2] ** 2
    LH1 = level1[:height//2, width//2:] ** 2
    HH1 = level1[height//2:, width//2:] ** 2
    Emap1 = np.sqrt(HL1 + LH1 + HH1)

    HL2 = level2[height//4:, :width//4] ** 2
    LH2 = level2[:height//4, width//4:] ** 2
    HH2 = level2[height//4:, width//4:] ** 2
    Emap2 = np.sqrt(HL2 + LH2 + HH2)

    HL3 = level3[height//8:, :width//8] ** 2
    LH3 = level3[:height//8, width//8:] ** 2
    HH3 = level3[height//8:, width//8:] ** 2
    Emap3 = np.sqrt(HL3 + LH3 + HH3)

    # Get max energy maps
    Emax1 = get_emax(Emap1, 8)
    Emax2 = get_emax(Emap2, 4)
    Emax3 = get_emax(Emap3, 2)

    # Detect edges
    m, n = Emax1.shape
    Nedge = 0
    Eedge = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            if (Emax1[i,j] > threshold or 
                Emax2[i,j] > threshold or 
                Emax3[i,j] > threshold):
                Nedge += 1
                Eedge[i,j] = 1.0

    # Detect Dirac and Astep
    Nda = 0
    for i in range(m):
        for j in range(n):
            if (Eedge[i,j] > 0.1 and 
                Emax1[i,j] > Emax2[i,j] and 
                Emax2[i,j] > Emax3[i,j]):
                Nda += 1

    # Detect Roof and Gstep
    Nrg = 0
    Eedge_Gstep_Roof = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            if (Eedge[i,j] > 0.1 and 
                ((Emax1[i,j] < Emax2[i,j] and Emax2[i,j] < Emax3[i,j]) or 
                 (Emax2[i,j] > Emax1[i,j] and Emax2[i,j] > Emax3[i,j]))):
                Nrg += 1
                Eedge_Gstep_Roof[i,j] = 1.0

    # Detect blurred roof and gstep
    Nbrg = 0
    for i in range(m):
        for j in range(n):
            if (Eedge_Gstep_Roof[i,j] > 0.1 and Emax1[i,j] < threshold):
                Nbrg += 1

    # Compute blur extent
    Per = Nda / Nedge if Nedge > 0 else 0
    unblured = 1 if Per > min_zero else 0
    BlurExtent = Nbrg / Nrg if Nrg > 0 else 0

    # End time
    end_time = time.time()
    cost_time = end_time - start_time

    # Return results
    return {
        'image_path': image_path,
        'height': height0,
        'width': width0,
        'is_clear': unblured == 1,
        'num_edge_points': Nedge,
        'num_dirac_astep': Nda,
        'num_roof_gstep': Nrg,
        'num_blurred_roof_gstep': Nbrg,
        'blur_extent': BlurExtent,
        'time_cost': cost_time
    }

def get_blur_score(image_path):

    # Use image path from command line if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    # Detect blur
    results = detect_blur(image_path)

    # # Print results
    # print("\nImage:", results['image_path'])
    # print("Height:", results['height'])
    # print("Width:", results['width'])
    # print("Image is clear:" if results['is_clear'] else "Image is blurred")
    # print("Num of edge points:", results['num_edge_points'])
    # print("Num of Dirac and Astep:", results['num_dirac_astep'])
    # print("Num of Roof and Gstep:", results['num_roof_gstep'])
    # print("Num of Roof and Gstep lost sharp:", results['num_blurred_roof_gstep'])
    # print("BlurExtent:", results['blur_extent'])
    # print("Time cost: {:.4f} s".format(results['time_cost']))

    return results['blur_extent']

# if __name__ == '__main__':
#     image_path = '/home/srivabhi22/Desktop/MRL/creek_00116.png'
#     print(get_blur_score(image_path))