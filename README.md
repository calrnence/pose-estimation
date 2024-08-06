Marker generation codes were taken from https://github.com/ddelago/Aruco-Marker-Calibration-and-Pose-Estimation
Calibration and pose estimation codes were taken from https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python

# Overview
This repository contains all the code necessary to generate, detect, identify and estimate the pose of ArUco tags.

## Libraries used:
Python 3.12.4\ 
Pandas 2.2.2\ 
OpenCV-Python 4.10.0.8\ 
Numpy 2.0.1\ 

Ensure that you are in the correct working directory to run all the code.

i.e., cd /d D:\my_name\pose_estimation

## A pipeline to estimate the pose of an ArUco tag is as follows:

###    1. Acquire images of calibration ChArUco board for calibration via generate_ChArUco.py
        a. Images should be in a designated folder
        b. Images should be taken with the camera you are using for pose estimation
        c. The calibration board should have different orientations in the Images
            i. More images --> Better calibration
        d. Parameters of the chessboard can be adjusted when running the command

###   2. Acquire camera and distortion matrices via calibration.py
        a. The matrices will be stored as .npy files
        b. A reprojection error (RMSE) is displayed; a smaller error is better

###    3. Run pose_estimation.py to acquire the pose of the tag relative to the camera
        a. ArUco tags can be generated using generate_arucoTags.py or generate_arucoGrid.py
        b. The pose of the marker returned by solvePnP is the transformation from the marker coordinate frame to the camera coordinate frame
        c. All pose information and the corresponding timestamps are stored in a .csv file


---------------------------------------------------------------------------------------------------

# generate_arucoTags.py

Used to generate a single jpg image of an ArUco tag. The image is saved after running.

### COMMAND: 
python generate_arucoTags.py --id {INTEGER} --type {STRING} --size {INTEGER}

### EXAMPLE: 
python generate_arucoTags.py --id 24 --type DICT_5X5_100 -o tags/

### PARAMETERS:
-i, --id | (REQUIRED) id of ArUco tag to generate\
-t, --type | type of ArUco tag to generate, see utils.py for full dictionary (default='DICT_ARUCO_ORIGINAL')\
-s, --size | size of ArUco tag to generate (default=200)\

### OUTPUT: 
jpg of a single ArUco tag


---------------------------------------------------------------------------------------------------

# generate_arucoGrid.py

Used to generate a grid of ArUco tags. The image is saved after running.

### COMMAND: 
python generate_arucoGrid.py --type {STRING} --width {INTEGER} --height {INTEGER} --length {FLOAT} --seperation {FLOAT}

### EXAMPLE: 
python generate_arucoGrid.py --type DICT_5X5_50

### PARAMETERS:
-t, --type | type of ArUco tag to generate, see utils.py for full dictionary (default='DICT_ARUCO_ORIGINAL')\ 
-w, --width | number of ArUco tags along width of the grid (default=3)\
-y, --height | number of ArUco tags along height of the grid (default=4)\
-l, --length | length of ArUco tags in meters (default=0.1)\
-s, --separation | separation between ArUco tags in meters (default=0.01)\

### OUTPUT: 
jpg of ArUco grid


---------------------------------------------------------------------------------------------------

# generate_ChArUco.py

Used to generate a chessboard with ArUco tags for calibration. The image is saved after running.

### COMMAND: 
python generate_ChArUco.py --type {STRING} --width {INTEGER} --height {INTEGER} --square {FLOAT} --marker {FLOAT}

### EXAMPLE:
python generate_ChArUco.py --type DICT_5X5_50 --width 3 --height 4

### PARAMETERS:
-t, --type | type of ArUco tag to generate, see utils.py for full dictionary (default='DICT_ARUCO_ORIGINAL')\
-w, --width | number of squares along width of the chessboard (default=6)\
-y, --height | number of squares along height of the chessboard (default=8)\
-q, --square | length of chessboard square in meters (default=0.04)\
-m, --marker | length of ArUco tag in meters (default=0.02)\

### OUTPUT:
 jpg of ChArUco board


---------------------------------------------------------------------------------------------------

# detect_aruco_images.py

Used to detect and identify ArUco markers in an image. 

### COMMAND: 
python detect_aruco_images.py  --image {PATH} --type {STRING}

### EXAMPLE: 
python detect_aruco_images.py --image Images/test_image_1.png --type DICT_5X5_100

### PARAMETERS:
-i, --image | (REQUIRED) path to input image\
-t, --type | type of ArUco tag to generate, see utils.py for full dictionary (default='DICT_ARUCO_ORIGINAL')\

### OUTPUT: 
image of detected markers


---------------------------------------------------------------------------------------------------

# detect_aruco_video.py

Used to detect and identify ArUco markers in a recording or live video feed.

### COMMAND: 
python detect_aruco_video.py --type {STRING} --camera {BOOL} --video {PATH}

### EXAMPLE: 
python detect_aruco_video.py --type DICT_5X5_100 --camera False --video test_video.mp4

### PARAMETERS:
-i, --camera | (REQUIRED) boolean to check for live camera feed, true if live camera feed\
-v, --video, | path to video file if applicable\
-t, --type | type of ArUco tag to detect, see utils.py for full dictionary (default='DICT_ARUCO_ORIGINAL')\

## OUTPUT: 
video of detected markers


---------------------------------------------------------------------------------------------------

# calibration.py

Used to calibrate a camera from a set of calibration images. Returns two .npy files containing the camera and distortion matrices.

### COMMAND: 
python calibration.py --dir {PATH} --width {INTEGER} --height {INTEGER} --square_size {FLOAT} --visualize

### EXAMPLE: 
python calibration.py --dir calibration_checkerboard/ --visualize

### PARAMETERS:
-d, --dir | (REQUIRED) path to folder with calibration images\
-w, --width | number of internal corners along width of checkerboard (default=7)\
-t, --height | number of internal corners along height of checkerboard (default=5)\
-s, square_size | length of one whole checkerboard square (meteres) (default=0.031)\
-v, --visualize | visualize calibration for each image, enter --visualize for true, remove for false\

### OUTPUT:
camera coefficient matrix and distortion matrix stored as numpy files


---------------------------------------------------------------------------------------------------

# pose_estimation.py

Used to display and collect pose information of the markers. Returns a .csv file containing the pose information relative to the camera coordinate frame. 

### COMMAND: 
python pose_estimation.py --K_Matrix {PATH} --D_Coeff {PATH} --type {STRING} --length {FLOAT}

### EXAMPLE: 
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100

### PARAMETERS:
-k, --K_Matrix | (REQUIRED) path to intrinsic camera matrix numpy file\
-d, --D_Coeff | (REQUIRED) path to distortion matrix numpy file\
-t, --type | type of ArUco tag to detect, see utils.py for full dictionary (default='DICT_ARUCO_ORIGINAL')\
-l, --length | length of one ArUco tag (default=0.015)\

### OUTPUT: 
transformations from object coordinate frame to camera coordinate frame, csv file with all transformations and corresponding timestamps

---------------------------------------------------------------------------------------------------
