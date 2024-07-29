'''
Sample Usage:-
python calibration.py --dir calibration_checkerboard/ --visualize
'''

import numpy as np
import cv2
import os
import argparse

def calibrate(dirpath, square_size, width, height, visualize):
    """ Apply camera calibration operation for images in the given directory path. """

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    images = os.listdir(dirpath)

    for fname in images:
        img = cv2.imread(os.path.join(dirpath, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

            if visualize:
                cv2.imshow('img', img)
                cv2.waitKey(0)

    if visualize:
        cv2.destroyAllWindows()

    # Perform camera calibration
    if len(objpoints) < 1 or len(imgpoints) < 1:
        raise ValueError("Not enough points to calibrate the camera. Check images.")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
 
        print( "Total reprojection error (smaller is better): {}".format(mean_error/len(objpoints)) )    

    return ret, mtx, dist, rvecs, tvecs

def save_calibration_results(mtx, dist):
    """ Save the calibration results to files. """
    np.save("calibration_matrix", mtx)
    np.save("distortion_coefficients", dist)
    print("Calibration results saved.")

def parse_arguments():
    """ Parse command line arguments. """
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Path to folder containing checkerboard images for calibration")
    ap.add_argument("-w", "--width", type=int, default=7, help="Number of internal corners along the width of the board(default=7)")
    ap.add_argument("-t", "--height", type=int, default=5, help="Number of internal corners along the height of the board (default=5)")
    ap.add_argument("-s", "--square_size", type=float, default=0.031, help="Length of one whole square in meters (default=0.031)")
    ap.add_argument("-v", "--visualize", action='store_true', help="To visualize each checkerboard image")
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    dirpath = args.dir
    square_size = args.square_size
    width = args.width
    height = args.height
    visualize = args.visualize

    ret, mtx, dist, rvecs, tvecs = calibrate(dirpath, square_size, width, height, visualize)

    print("Camera matrix:")
    print(mtx)
    print("Distortion coefficients:")
    print(dist)

    save_calibration_results(mtx, dist)