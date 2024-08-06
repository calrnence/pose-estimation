import cv2
import argparse
from utils import ARUCO_DICT
# Create gridboard, which is a set of Aruco markers
# the following call gets a board of markers 5 wide X 7 tall

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tags to generate")
ap.add_argument("-w", "--width", type=int, default=3, help="Number of ArUco tags along width of the grid")
ap.add_argument("-y", "--height", type=int, default=4, help="Number of ArUco tags along height of the grid")
ap.add_argument("-l", "--length", type=float, default=0.1, help="Length of ArUCo tag in meters")
ap.add_argument("-s", "--seperation", type=float, default=0.01, help="Seperation between ArUco tags")

args = vars(ap.parse_args())

gridboard = cv2.aruco.GridBoard(
        size=(args["width"],args["height"]), 
        markerLength=args["length"], 
        markerSeparation=args["seperation"], 
        dictionary=cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]]))

# Create an image from the gridboard
img = gridboard.generateImage(outSize=(988, 1400))
cv2.imwrite("gridboard.jpg", img)

# Display the image to us
cv2.imshow('Gridboard', img)
# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()