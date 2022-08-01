import math
import matplotlib.pyplot as plt
import cv2
import numpy as np

# ------------------------------------------------------------------------------------------------------------------
# This is Q1 in Ex 2 for Computer Vision Course
# the Program in this file demostrates an algorithm for detecting epipolar lines given 2 images, first we chose a point
# set in the images that correspond to each other containing 10 points, we computed the fundemental matrix using these points
# then computed the average SED, then we chose a different set of points also containing 10 points, which we computed the
# epipolar lines from them, then using the previous fundemental matrix, computed the average SED.
# **********
# Students:-
# Giries Abu Ayob
# Abla Dawood
# **********
# ------------------------------------------------------------------------------------------------------------------

# functions for loading the given images
def InputOxford():
    images_array = []

    sourceLeft = cv2.resize(cv2.imread(r'Q1\location_1_frame_001.jpg'), (500, 500)
                         , interpolation=cv2.INTER_AREA)
    sourceRight = cv2.resize(cv2.imread(r'Q1\location_1_frame_002.jpg'), (500, 500)
                         , interpolation=cv2.INTER_AREA)

    images_array.append(sourceLeft)
    images_array.append(sourceRight)

    return images_array

def InputCastle():
    images_array = []

    sourceLeft = cv2.resize(cv2.imread(r'Q1\location_2_frame_001.jpg'), (500, 500)
                         , interpolation=cv2.INTER_AREA)
    sourceRight = cv2.resize(cv2.imread(r'Q1\location_2_frame_002.jpg'), (500, 500)
                         , interpolation=cv2.INTER_AREA)

    images_array.append(sourceLeft)
    images_array.append(sourceRight)

    return images_array

# function to show the images with the computed SED as a suptitle
def ShowImages(img1, img2, avg):
    plt.figure()
    plt.suptitle("SED = " + str(avg))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.title("Left")
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.title("Right")
    plt.show()

# function to draw the computed epipolar lines
def drawlines(img1, img2, lines, pts1, pts2):
    r, c , f = img1.shape
    colors = []
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2) ,5 , color, -1)
        colors.append(color)
    return img1, img2, colors

def drawlines2(img1, img2, lines, pts1, pts2, colors):
    r, c , f = img1.shape
    for r, pt1, pt2, color in zip(lines, pts1, pts2, colors):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2) ,5 , color, -1)
    return img1, img2

# function to compute the epipoalr lines given 2 images and 2 point sets
def EpipolarLines(img1, img2, pts1, pts2):
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6, colors = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines2(img2, img1, lines2, pts2, pts1, colors)
    return img5, img3, F

# function to compute the SED given a fundemental matrix and 2 point sets
def SED(pts1, pts2, F):
    diff = []
    for i in range(10):
        xtag = pts2[i].reshape(3, 1)
        xtag = np.transpose(xtag)
        x = pts1[i].reshape(3, 1)
        L = np.matmul(F, x)
        XF = np.matmul(xtag, F)
        XFx = np.matmul(XF, x)
        d = XFx/(math.sqrt(pow(L[0], 2) + pow(L[1], 2)))
        d = abs(d)
        diff = np.append(diff, d)
    avg = np.average(diff)
    return diff, avg

if __name__ == '__main__':

    Oxford = InputOxford()
    Castle = InputCastle()

#----------------Oxford----------------------------
    img1 = Oxford[0]
    img2 = Oxford[1]

    #---------------First Point Set----------------
    pts1 = np.asarray([(147, 85), (100, 262), (88, 251), (55, 172), (433, 230), (286, 317), (223, 442),
                      (317, 206), (22, 460), (287, 316)])
    pts2 = np.asarray([(153, 90), (108, 256), (98, 246), (65, 174), (417, 224), (286, 311), (224, 425),
                      (315, 203), (40, 440), (285, 311)])

    pts1_3d = np.asarray([(147, 85, 1), (100, 262, 1), (88, 251, 1), (55, 172, 1), (433, 230, 1), (286, 317, 1), (223, 442, 1),
                       (317, 206, 1), (22, 460, 1), (287, 316, 1)])
    pts2_3d = np.asarray([(153, 90, 1), (108, 256, 1), (98, 246, 1), (65, 174, 1), (417, 224, 1), (286, 311, 1), (224, 425, 1),
                       (315, 203, 1), (40, 440, 1), (285, 311, 1)])

    #-----Compute epipolar lines-------------------
    drawnimg1, drawnimg2, F = EpipolarLines(img1, img2, pts1, pts2)

    #------compute SED avg and showing the Images---------
    diff, avg = SED(pts1_3d, pts2_3d, F)

    ShowImages(drawnimg1, drawnimg2, avg)

    print("SED for the first set in oxford:", avg)



    #----------------- Second Set of points----------------------
    Oxford = InputOxford()
    img1 = Oxford[0]
    img2 = Oxford[1]

    pts1 = np.asarray(
        [(284, 347), (55, 248), (87, 172), (11, 177), (241, 394), (419, 157), (456, 303),
         (246, 334), (90, 468), (176, 410)])
    pts2 = np.asarray(
        [(282, 340), (66, 247), (96, 176), (24, 180), (240, 383), (405, 158), (436, 290),
         (246, 327), (104, 450), (179, 398)])

    # ---- computing epipolar lines for second point set ------
    # the fundemental matrix (exces) here is redundant, we don't use it

    drawnimg1, drawnimg2, exces = EpipolarLines(img1, img2, pts1, pts2)

    # SED for tester set
    pts1_3d = np.asarray(
        [(284, 347, 1), (55, 248, 1), (87, 172, 1), (11, 177, 1), (241, 394, 1), (419,157 , 1), (456, 303, 1),
         (246, 334, 1), (90, 468, 1), (176, 410, 1)])
    pts2_3d = np.asarray(
        [(282, 340, 1), (66, 247, 1), (96, 176, 1), (24, 180, 1), (240, 383, 1), (405, 158, 1), (436, 290, 1),
         (246, 327, 1), (104, 450, 1), (179, 398, 1)])

    diff, avg = SED(pts1_3d, pts2_3d, F)

    ShowImages(drawnimg1, drawnimg2, avg)

    print("SED for the second set in oxford:", avg)

    # ------------------Castle-------------------
    img1 = Castle[0]
    img2 = Castle[1]

    pts1 = np.asarray([(271, 160), (20, 278), (260, 357), (422, 15), (134, 118), (383, 212), (184, 180),
                       (82, 342), (460, 265), (487, 339)])
    pts2 = np.asarray([(253, 140), (53, 241), (261, 310), (399, 21), (140, 95), (370, 184), (185, 156),
                       (104,301), (434, 224), (456, 281)])

    pts1_3d = np.asarray([(271, 160, 1), (20, 278, 1), (260, 357, 1), (422, 15, 1), (134, 118, 1), (383, 212, 1), (184, 180, 1),
                       (82, 342, 1), (460, 265, 1), (487, 339, 1)])
    pts2_3d = np.asarray([(253, 140, 1), (53, 241, 1), (261, 310, 1), (399, 21, 1), (140, 95, 1), (370, 184, 1), (185, 156, 1),
                       (104,301, 1), (434, 224, 1), (456, 281, 1)])

    #--------------compute Epipolar Lines---------------------
    drawnimg1, drawnimg2, F = EpipolarLines(img1, img2, pts1, pts2)

    #-------SED average with the fundemental matrix and showing images -------
    diff, avg = SED(pts1_3d, pts2_3d, F)

    ShowImages(drawnimg1, drawnimg2, avg)

    print("SED for the first set in castle:", avg)

    # ---------------Second Set---------------------------
    Castle = InputCastle()
    img1 = Castle[0]
    img2 = Castle[1]

    pts1 = np.asarray(
        [(126, 61), (9, 210), (176, 209), (299, 138), (418, 100), (332, 326), (467, 340),
         (281, 188), (99, 282), (439, 206)])
    pts2 = np.asarray(
        [(134, 45), (41, 177), (177, 182), (282, 120), (394, 90), (317, 283), (440, 285),
         (265, 164), (117, 246), (417, 176)])

    #---------------new epipolar lines with the tester set------------
    drawnimg1, drawnimg2, exces = EpipolarLines(img1, img2, pts1, pts2)

    # SED for a tester set
    pts1_3d = np.asarray(
        [(126, 61, 1), (9, 210, 1), (176, 209, 1), (299, 138, 1), (418, 100, 1), (332, 326, 1), (467, 340, 1),
         (281, 188, 1), (99, 282, 1), (439, 206, 1)])
    pts2_3d = np.asarray(
        [(134, 45, 1), (41, 177, 1), (177, 182, 1), (282, 120, 1), (394, 90, 1), (317, 283, 1), (440, 285, 1),
         (265, 164, 1), (117, 246, 1), (417, 176, 1)])

    #------SED with the previous matrix----------
    diff, avg = SED(pts1_3d, pts2_3d, F)

    ShowImages(drawnimg1, drawnimg2, avg)

    print("SED for the second set in castle:", avg)