from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2 as cv
import numpy as np
from random import randrange
import math

# PLAN
# 1. Run facial landmarks detection script on input 2D face, to get detect 68 landmarks on the face
# 2. Match up the biological regions in the thickness paper, to the landmarks within those regions.
# 3. Find the inverse function that takes a vertex in the 3DMM model and outputs a coordinate in the
# 2D image, so that you can find its thickness based on the nearest landmark.
# 4. Rinse and repeat for every vertex in the 3D model.

# NOTES
# 1. We can even potentially add more landmarks than 68, if we define it as relative to an already present landmark.
# 2. To go from 3D vector to 2D coordinate, just take the x, y point from the portrait of a 3DMM model

# PROBLEMS:
# 1. The thickness values around the nostrils aren't accounted for in the facial landmarks
#     a) Add more of your own thickness values to the map
# 2. Why is the current skin thickness value d, 0.128...? Where is this value coming from and what are its units?
# 3. If the face is turned to the right for example, then the right side of the face is hidden below the front of the face in the 3dmm model. As it wraps around, this causes it to map to the incorrect features in the feature landmark image, such as the nose, etc. This results in incorrect thicknesses for those portions of the face. This only happens for highly tilted faces.

# TODO:
# 1. Use the maximal distanced (in the x-axis) points in both the vertices map and the original landmarked image (should be around the cheeks), and use these as reference points to scale and transform one or both images until they are aligned.
# 2. Once they are aligned, drop the z-axis of the vertex, and apply the thickness at a vertex location to the vertex's corresponding angle in the compute_strength function. 

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

# Construct a data structure that maps each landmark to a thickness, and each coordinate of a landmark to its landmark index?
# Eventually you want to find a direct relationship between a coordinate in the 3d place, to a coordinate in the 2d plane, to a nearby landmark, to a thickness


def contruct_mapping():
    # Doesn't include landmarks 61 - 68 (inside of lips)
    epi_depth = {1:1.64, 2:1.37, 3:1.37, 4:1.7, 5:1.7, 6:1.5, 7:1.3, 8:1.3, 9:1.54, 10:1.3, 11:1.3, 12:1.5, 13:1.7, 14:1.7, 15:1.37, 16:1.37, 17:1.64, 18:1.64, 19:1.54, 20:1.54, 21:1.54, 22:1.77, 23:1.77, 24:1.54, 25:1.54, 26:1.54, 27:1.64, 28:1.94, 29:1.94, 30:1.58, 31:1.70, 32:1.89, 33:1.58, 34:1.58, 35:1.58, 36:1.89, 37:1.62, 38:1.43, 39:1.0, 40:1.11, 41:1.55, 42:1.14, 43:1.11, 44:1.0, 45:1.43, 46:1.62, 47:1.14, 48:1.14, 49:1.69, 50:1.89, 51:1.58, 52:1.58, 53:1.58, 54:1.89, 55:1.69, 56:1.3, 57:1.4, 58:1.54, 59:1.4, 60:1.3}
    return epi_depth
    

def main():
    vectors = np.array([ [randrange(0, 500), randrange(0, 500), randrange(0, 500)] for i in range(60000)])
    vectors = np.array([ [50,50,100], [250, 250, 213], [400, 10, 0], [200, 400, 0]])
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
	            help="path to facial landmark predictor")
    ap.add_argument("-i", "--image", required=True,
	            help="path to input image")
    args = vars(ap.parse_args())
    
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    #predictor = dlib.cnn_face_detection_model_v1(args["shape_predictor"])

    # load the input image, resize it, and convert it to grayscale
    image = cv.imread(args["image"])
    image = imutils.resize(image, width=500)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = rect_to_bb(rect)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        cv.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for j, (x, y) in enumerate(shape):
            #cv.circle(image, (x, y), 1, (0, 0, 255), -1)
            cv.putText(image, f"{j+1}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        # show the output image with the face detections + facial landmarks
        #cv.imshow("Output", image)
        #cv.waitKey(0)
    
    epi_depth = contruct_mapping()
    skin_depth = []
    for vect in vectors:
        cv.circle(image, (vect[0], vect[1]), 5, (0, 255), -1)
        closest = [-1, float('inf')]
        sec_closest = [-1, float('inf')]
        for j, (x, y) in enumerate(shape):
            if j >= 60:
                break
            euc_dist = math.dist([x,y], vect[:2])
            min_closest = min(closest[1], euc_dist)

            if min_closest != closest[1]:
                sec_closest = closest.copy()
                closest = [j+1, min_closest]
            else:
                min_closest = min(sec_closest[1], euc_dist)
                sec_closest = [j+1, min_closest]
        tot_clos = closest[1]+sec_closest[1]
        # NOTE: Arbitrary weightage of closest and second closest thickness, based on distance of other.
        # Eg. if closest is 1 away and second closest is 4 away, the weight of the closer thickness is 4/5 (and the other weight is 1/5).
        weight_avg = (epi_depth[closest[0]]*sec_closest[1]/tot_clos) + (epi_depth[sec_closest[0]]*closest[1]/tot_clos)
        skin_depth.append([weight_avg, closest[0], sec_closest[0], vect])
    print(f"landmarks:{shape}")
    print(f"skin_depths:{skin_depth}")

    cv.imshow("Output", image)
    cv.waitKey(0)

if __name__ == "__main__":
    main()
