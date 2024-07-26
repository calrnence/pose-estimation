import numpy as np
import cv2
import os
import argparse

def calibrate(dirpath, square_size, width, height, visualize=False):
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

    images = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
    if not images:
        raise ValueError(f"No images found in directory {dirpath}")

    for fname in images:
        img_path = os.path.join(dirpath, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping...")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
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
    ap.add_argument("-w", "--width", type=int, default=8, help="Width of checkerboard (default=9)")
    ap.add_argument("-t", "--height", type=int, default=6, help="Height of checkerboard (default=6)")
    ap.add_argument("-s", "--square_size", type=float, default=0.031, help="Length of one edge (in meters)")
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