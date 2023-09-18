import cv2
import numpy as np
import os

# Directory containing images to distort
input_dir = 'C:\\Users\\riand\\Documents\\Masters\\Project\\Rendered_Data\\Undistorted\\'

# Directory to save distorted images
output_dir = 'C:\\Users\\riand\\Documents\\Masters\\Project\\Rendered_Data\\Distorted\\'

# Get the list of all files in the input directory
input_files = os.listdir(input_dir)

# Allowed image extensions
image_extensions = ['.jpg', '.png', '.bmp', '.jpeg']

# Camera intrinsics and distortion coefficients
image_size = [1280,720]
fx = 500
fy = 500
cx  =image_size[0]/2
cy = image_size[1]/2
k1, k2, p1, p2, k3 = 0.1, 0.1, 0.01, 0.01, 0.1

# Define the camera matrix K
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# Define distortion coefficients
d = np.array([k1, k2, p1, p2, k3]) 

for file in input_files:
    # Check the file extension
    _, ext = os.path.splitext(file)
    if ext.lower() not in image_extensions:
        continue
    
    # Form the full input file path
    file_path = os.path.join(input_dir, file)
    
    # Load the image
    image = cv2.imread(file_path)

    # Apply distortion
    distorted_image = cv2.undistort(image, K, d)
    
    # Form the full output file path
    output_file_path = os.path.join(output_dir, file)

    # Save the distorted image
    cv2.imwrite(output_file_path, distorted_image)
