'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''

import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import csv
from datetime import datetime

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, length):
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
    
    rvec, tvec = None, None

        # If markers are detected
    if corners:
        objPoints = np.array([[-length/2, length/2,0],
                         [length/2, length/2, 0],
                         [length/2, -length/2, 0],
                         [-length/2, -length/2, 0]])
        for corner in corners:
            # Estimate pose of each marker and return the temporary values for rvec_ and tvec_---(different from those of camera coefficients)
            ret, rvec_, tvec_ = cv2.solvePnP(objPoints, corner, matrix_coefficients, distortion_coefficients, flags=cv2.SOLVEPNP_ITERATIVE)
            if ret: # check if solvePnP was successful
                rvec_, tvec_, = cv2.solvePnPRefineLM(objPoints, corner, matrix_coefficients, distortion_coefficients, rvec_, tvec_) # pose refinement step, optional
                if ret:
                    rvec, tvec = rvec_, tvec_
                    # Draw a square around the markers
                    cv2.aruco.drawDetectedMarkers(frame, corners) 

                    # Draw axis on center of marker
                    cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame, rvec, tvec  

# from utils.py 
def displayid(corners, ids, rejected, image):
    if corners:        
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return image

# appends timestamp, rotation and translation to pre-existing csv file
def save(filename, timestamp, rvec, tvec):
    with open(filename, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp,'', rvec[0], rvec[1], rvec[2],'', tvec[0], tvec[1], tvec[2]])

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
    # initialize new csv file to store data
    filename = datetime.now().strftime('transformations_%Y-%m-%d_%H-%M-%S.csv')
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['','', 'rotation', '', '', '', 'translation'])
        writer.writerow(['timestamp','', 'yaw', 'pitch', 'roll', '', 'x', 'y', 'z'])
    # start stopwatch 
    start = time.perf_counter()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        output, rvec, tvec = pose_estimation(frame, aruco_dict_type, k, d, marker_length)

        # saves data only if solvePnP was successful
        if rvec is not None and tvec is not None:
            timestamp = time.perf_counter() - start
            save(filename, timestamp, rvec, tvec)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()