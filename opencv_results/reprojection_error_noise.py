import os
import cv2
import numpy as np
import random
from sklearn.model_selection import ParameterGrid
import pickle

src_dir = "C:\\Users\\riand\\Documents\\Masters\\Project\\Rendered_Data\\Auto_generated\\Undistorted\\"

# Define blur kernel sizes and noise levels
blur_kernel_sizes = [(i, i) for i in range(1, 8, 2)]
noise_stddevs = [i for i in range(1, 255, 25)]

# Define the grid of parameters
param_grid = list(ParameterGrid({"blur": blur_kernel_sizes, "noise": noise_stddevs}))

num_samples = 15  # Number of images to sample per directory
num_repeats = 1   # Number of times to repeat the experiment

# Termination criteria for corner sub-pix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points like (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
objp = np.zeros((15*10,3), np.float32)
objp[:,:2] = np.mgrid[0:15,0:10].T.reshape(-1,2)

def calibrate_camera(images):
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (15,10), None)

        if ret == True:
            objpoints.append(objp)

            # Refine the corners
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    return mean_error / len(objpoints), mtx

results = {}


print("Number of parameter combinations:", len(param_grid))
for subdir in os.listdir(src_dir):
    subdir_path = os.path.join(src_dir, subdir)

    if os.path.isdir(subdir_path):
        images = os.listdir(subdir_path)
        sampled_images = random.sample(images, num_samples)

        for j, params in enumerate(param_grid):
            print(j)
            print(params)
            for i in range(num_repeats):
                noisy_images = []
                for img_name in sampled_images:
                    img = cv2.imread(os.path.join(subdir_path, img_name))

                    # Apply Gaussian blur
                    blurred_img = cv2.GaussianBlur(img, params["blur"], 0)

                    # Generate and add Gaussian noise
                    noise = np.random.normal(0, params["noise"], img.shape)
                    noisy_img = np.clip(blurred_img + noise, 0, 255).astype(np.uint8)
                    noisy_images.append(noisy_img)
                
                error, matrix = calibrate_camera(noisy_images)
                # Store the results
                results[(subdir, str(params), i)] = (error, matrix)
                print(error)

print(results)
# Save the dictionary containing the errors
with open('noise_results.pkl', 'wb') as f:
    pickle.dump(results, f)