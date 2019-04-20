import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math


def read_img(path):
    img_color = cv2.imread(path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img_color)       # get b,g,r
    img_color = cv2.merge([r, g, b])/255.0
    return img_color, img_gray

def getDistForHP(p, H):
    pi = np.matrix([p[0], p[1], 1])
    p1f = np.matmul(H, pi.transpose())
    p1f = p1f.transpose().tolist()[0]
    p1f = [p1f[0]/p1f[2], p1f[1]/p1f[2]]
    distance = math.sqrt((p1f[0]-p[2])**2 + (p1f[1]-p[3])**2)
    return distance


def pointPairToPoint(p1, p2):
    return [p1[0], p1[1], p2[0], p2[1]]


def match_piece(piece, solved_kp, solved_desc, finished_gray):
        # piece is grayscale sub_image of piece
        # solved_kp are the keypoints in the grayscale solved image
        # solved_desc are the descriptors for the keypoints in the solved image
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(piece, None)

    # Match features.
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des, solved_desc, k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    


    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)
    
    verified_matches = []
    
    for i, match in enumerate(good):
        points1[i, :] = kp[match.queryIdx].pt
        points2[i, :] = solved_kp[match.trainIdx].pt
        verified_matches.append((match.queryIdx, match.trainIdx))
    
    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    if H == None:
        
        print(len(good))
        raise Exception("ERROR")

    inlier_matches = []

    num_inliers = 0

    for match in verified_matches:
        p = pointPairToPoint(kp[match[0]].pt, solved_kp[match[1]].pt)
        dist = getDistForHP(p, H)
        if dist < 1:
            inlier_matches.append(cv2.DMatch(match[0], match[1], 0))
            num_inliers += 1

    img3 = cv2.drawMatches(piece, kp, finished_gray,
                           solved_kp, inlier_matches, None, flags=2)

    # Find coordinates on finished puzzle image
    input_cord = np.array([piece.shape[0]/2, piece.shape[1]/2, 1])
    output_cord = np.matmul(H, input_cord.transpose())

    output_xy = (int(output_cord[0]/output_cord[2]),
                 int(output_cord[1]/output_cord[2]))
    # cv2.rectangle(finished_gray, (output_xy[0]-10, output_xy[1]-10),\
    #               (output_xy[0]+10, output_xy[1]+10), (0, 255, 0), 2)
    # plt.imshow(finished_gray)
    # plt.show()
    return output_xy


def splitPieces():
    pieces_color, pieces_gray = read_img('../data/Fairies.png')
    finished_color, finished_gray = read_img('../data/Fairies_complete.png')

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()

    # Detect blobs.
    keypoints = detector.detect(pieces_gray)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(
        pieces_gray, keypoints, None, color=(0, 255, 0), flags=0)

    blurred = cv2.GaussianBlur(pieces_gray, (5, 5), 1)
    blurred = cv2.medianBlur(blurred, 7)
    ret, thresh1 = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    # plt.imsave('./thresh.png', thresh1, cmap='gray')


    # edged = cv2.Canny(blurred, 0, 150)
    # dilated = cv2.dilate(edged, None, iterations=1)
    # eroded = cv2.erode(dilated, None, iterations=1)

    # ret, thresh1 = cv2.threshold(dilated, 30, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh1)
    # plt.show()
    # plt.imshow(thresh1)
    # plt.show()
    # exit()

    # find contours in the thresholded image and initialize the shape detector
    contours = cv2.findContours(
        thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1]

    sift = cv2.xfeatures2d.SIFT_create()
    (kp_finished, desc_finished) = sift.detectAndCompute(finished_gray, None)

    piece_matches = []

    for index, contour in enumerate(contours):
        if cv2.contourArea(contour) < 3000:
            continue
        print('yeet')
        x, y, w, h = cv2.boundingRect(contour)
        crop_img = pieces_gray[y:y+h, x:x+w]
        # plt.imsave('./sub/' + str(index) + '.png', crop_img, cmap='gray')
        cv2.rectangle(pieces_color, (x, y), (x+w, y+h), (0, 1, 0), 2)

        try:
            finished_point = match_piece(
                crop_img, kp_finished, desc_finished, finished_gray)
            pieces_point = (x+w/2, y+h/2)
            piece_matches.append((pieces_point, finished_point))
        except:
            print("in exception, match_piece failed")
        # epsilon = 0.05 * cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, epsilon, True)
        # cv2.drawContours(pieces_color, [approx], -1, (0, 255, 0), 2)
    print(len(piece_matches))
    np.save('piece_matches.npy', piece_matches)
    # Show keypoints
    plt.imsave('./out.png', pieces_color)
    print(piece_matches)


splitPieces()
