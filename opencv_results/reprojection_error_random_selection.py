import cv2
import glob
import numpy as np
import os
import random
import pickle
# Termination criteria for corner sub-pix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points like (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
objp = np.zeros((15*10,3), np.float32)
objp[:,:2] = np.mgrid[0:15,0:10].T.reshape(-1,2)

def calibrate_camera(images):
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    for fname in images:
        img = cv2.imread(fname)
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

    return mean_error / len(objpoints)

# Base directory
base_dir = "C:\\Users\\riand\\Documents\\Masters\\Project\\Rendered_Data\\Auto_generated\\Undistorted\\"

# Dictionary to hold errors
errors = {}
min_images = 10
max_images = 25
experiment_repeats = 5
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)

    # Ensure the path is a directory
    if os.path.isdir(subdir_path):
        # Create array to hold errors for this folder
        errors[subdir] = np.zeros((max_images + 1 - min_images, experiment_repeats))

        image_files = glob.glob(os.path.join(subdir_path, '*.png'))  
        print(subdir)

        for n_images in range(min_images, max_images + 1):  # Loop over numbers from 10 to 50
            print(n_images)
            for i in range(experiment_repeats):  # Repeat each experiment 

                # Randomly sample images
                sampled_images = random.sample(image_files, n_images)

                # Run camera calibration
                error = calibrate_camera(sampled_images)

                # Save error
                errors[subdir][n_images-min_images, i] = error


print(errors)
# Save the dictionary containing the errors
with open('errors.pkl', 'wb') as f:
    pickle.dump(errors, f)