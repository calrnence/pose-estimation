'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''

import numpy as np
import cv2
import sys
from utils import ARUCO_DICT, displayid, save_data, setup_hdf5
import argparse
import time
from datetime import datetime
import h5py


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, length, filename, timestamp):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    the estimated pose is the pose of the object with respect to the camera coordinate axis
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(cv2.aruco_dict, parameters)

    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    # show marker id in top left corner
    frame = displayid(corners, ids, rejected_img_points, frame)

        # If markers are detected
    if corners:
        ids = ids.flatten()
        objPoints = np.array([[-length/2, length/2,0],
                         [length/2, length/2, 0],
                         [length/2, -length/2, 0],
                         [-length/2, -length/2, 0]])
        with h5py.File(filename,'a') as file:
            
            for i in range(0, len(ids)):
                if f'marker_{ids[i]}' not in file:
                    setup_hdf5(file, ids[i])
                group = file[f'marker_{ids[i]}']
                # Estimate pose of each marker and return the temporary values for rvec_ and tvec_---(different from those of camera coefficients)
                ret, rvec, tvec = cv2.solvePnP(objPoints, corners[i], matrix_coefficients, distortion_coefficients, flags=cv2.SOLVEPNP_ITERATIVE)
                if ret:
                    rvec, tvec, = cv2.solvePnPRefineLM(objPoints, corners[i], matrix_coefficients, distortion_coefficients, rvec, tvec) # pose refinement step, optional
                    
                    # if a group for the marker doesnt exist, create new group
                    save_data(group, timestamp, rvec, tvec)
                    # Draw a square around the markers
                    cv2.aruco.drawDetectedMarkers(frame, corners) 
                    # Draw axis on center of marker
                    cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    ap.add_argument("-l", "--length", type=float, default=0.015, help="Length of detected marker in meters")
    args = vars(ap.parse_args())
    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    marker_length = args["length"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)
    # initialize stopwatch new csv file to store data 
    filename = datetime.now().strftime('transformations_%Y-%m-%d_%H-%M-%S.hdf5')
    start = time.perf_counter()

    while True:
        ret, frame = video.read()
        if not ret:
            break
        # stop stopwatch
        timestamp = time.perf_counter() - start
        output = pose_estimation(frame, aruco_dict_type, k, d, marker_length, filename, timestamp)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()