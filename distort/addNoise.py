import os
import cv2
import numpy as np

# Path to the source directory
src_dir = "C:\\Users\\riand\\Documents\\Masters\\Project\\Rendered_Data\\Auto_generated\\Undistorted\\"

# Path to the destination directory
dst_dir = "C:\\Users\\riand\\Documents\\Masters\\Project\\Rendered_Data\\Auto_generated\\withNoise\\"

# Gaussian blur parameters
kernel_size = (1, 1)

# Gaussian noise parameters
mean = 0
stddev = 5

# Loop over sub-directories in the source directory
for subdir in os.listdir(src_dir):
    subdir_path = os.path.join(src_dir, subdir)

    # Only proceed if the path is indeed a directory
    if os.path.isdir(subdir_path):
        # Create corresponding sub-directory in the destination directory
        dst_subdir = os.path.join(dst_dir, subdir)
        os.makedirs(dst_subdir, exist_ok=True)

        # List of image files in the source sub-directory
        images = os.listdir(subdir_path)

        # Process each image
        for img_name in images:
            # Load the image
            img = cv2.imread(os.path.join(subdir_path, img_name))

            # Add Gaussian blur
            blurred_img = cv2.GaussianBlur(img, kernel_size, 0)

            # Generate gaussian noise
            #https://pythonexamples.org/python-opencv-add-noise-to-image/

            noise = np.random.normal(mean, stddev, img.shape)
            noise = (noise).astype(int)
            print(noise)




            # Add the Gaussian noise to the image
            noisy_img = blurred_img + noise

            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

            # Write the modified image to the destination directory
            cv2.imwrite(os.path.join(dst_subdir, 'noisy_'+img_name), noisy_img)
