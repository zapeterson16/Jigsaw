import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_img(path):
    img_color = cv2.imread(path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img_color)       # get b,g,r
    img_color = cv2.merge([r, g, b])/255.0
    return img_color, img_gray


def splitPieces():
    pieces_color, pieces_gray = read_img('../data/Fairies_pieces.png')

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()

    # Detect blobs.
    keypoints = detector.detect(pieces_gray)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(
        pieces_gray, keypoints, None, color=(0, 255, 0), flags=0)

    blurred = cv2.GaussianBlur(pieces_gray, (11, 11), 0)
    edged = cv2.Canny(blurred, 50, 100)
    dilated = cv2.dilate(edged, None, iterations=1)
    eroded = cv2.erode(dilated, None, iterations=1)

    # find contours in the thresholded image and initialize the shape detector
    contours = cv2.findContours(
        eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1]

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(pieces_color, [approx], -1, (0, 255, 0), 2)

    # Show keypoints
    plt.imsave('./out.png', pieces_color)


splitPieces()
