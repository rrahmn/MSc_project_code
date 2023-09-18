import numpy as np
import cv2 as cv
import glob
import os


def main():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((15*10,3), np.float32)
    objp[:,:2] = np.mgrid[0:15,0:10].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    #Make sure to be in script directory before running code
    # Get the current working directory
    script_directory = os.getcwd()
    # Get the the parent directory
    parent_directory = os.path.dirname(os.path.dirname(script_directory))
    print(parent_directory)
    #Get the data directory
    data_directory = parent_directory + '\Data'


    images = glob.glob(data_directory + '\Frames_Left' + '\*.png')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (15,10), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (15,10), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(0)
    cv.destroyAllWindows()


    #calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
if __name__=="__main__":
    main()