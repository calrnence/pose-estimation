import cv2
import argparse
from utils import ARUCO_DICT

# Create ChArUco board, which is a set of Aruco markers in a chessboard setting
# meant for calibration
# the following call gets a ChArUco board of tiles 5 wide X 7 tall

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tags to generate")
ap.add_argument("-w", "--width", type=int, default=6, help="Number of squares along the width of the chessboard")
ap.add_argument("-y", "--height", type=int, default=8, help="Number of squares along the height of the chessboard")
ap.add_argument("-q", "--square", type=float, default=0.04, help="Length of chessboard square in meters")
ap.add_argument("-m", "--marker", type=float, default=0.02, help="Length of ArUco tag in meters")

args = vars(ap.parse_args())

gridboard = cv2.aruco.CharucoBoard(
        size=(args["width"],args["height"]), 
        squareLength=args["square"], 
        markerLength=args["marker"], 
        dictionary=cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]]))

# Create an image from the gridboard
img = gridboard.generateImage(outSize=(988, 1400))
cv2.imwrite("charuco.jpg", img)

# Display the image to us
cv2.imshow('Gridboard', img)
# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()