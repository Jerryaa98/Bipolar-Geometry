import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation as R

# ------------------------------------------------------------------------------------------------------------------
# This is Q2 in Ex 2 for Computer Vision Course
# the Program in this file demostrates an algorithm for metric reconstruction lines given 2 images as input, camera matrices,
# and matched points, we read the data via the given text files then drew the border of the house using a function we implemented,
# then used triangulation to get the 3D coordinates of the points, reduced the 3D coordinates to 2D then projected the
# model on an XY cartesian plane, then we rotated the projection multiple times and constructed a gif.
# **********
# Students:-
# Giries Abu Ayob
# Abla Dawood
# **********
# ------------------------------------------------------------------------------------------------------------------

# Reading the data from the camera matrices text files
def cameraMatrices():
    points1 = open("cameraMatrix1.txt", "r+")
    points = points1.readlines()
    my_points = []
    for point in points:
        new_point = point.split(",")
        if new_point[-1][-1] == '\n':
            new_point[-1] = new_point[-1][:-1]
        my_points.append(new_point)
    matrix1 = np.array(my_points, dtype=float)
    points2 = open("cameraMatrix2.txt", "r+")
    points2 = points2.readlines()
    my_points2 = []
    for point in points2:
        new_point = point.split(",")
        if new_point[-1][-1] == '\n':
            new_point[-1] = new_point[-1][:-1]
        my_points2.append(new_point)
    matrix2 = np.array(my_points2, dtype=float)

    return matrix1, matrix2

# Reading the data the matched points text files
def getData():
    points1 = open("matchedPoints1.txt", "r+", encoding='utf-8')
    points = points1.readlines()
    my_points1 = []
    for point in points:
        new_list = []
        new_point = point.split(",")
        if new_point[-1][-1] == '\n':
            new_point[-1] = new_point[-1][:-1]
        new_list.append(float(new_point[0]))
        new_list.append(float(new_point[1]))
        my_points1.append(new_list)

    points2 = open("matchedPoints2.txt", "r+", encoding='utf-8')
    points2 = points2.readlines()
    my_points2 = []
    for point in points2:
        new_list = []
        new_point = point.split(",")
        new_point[-1] = new_point[-1].replace('\n', '')
        new_list.append(float(new_point[0]))
        new_list.append(float(new_point[1]))
        my_points2.append(new_list)

    return my_points1, my_points2

# drawing the points on the house border
def DrawPoints(img1, img2):
    colors = []
    for pt1, pt2 in zip(pts1, pts2):
        color = np.random.randint(0, 255, 3).tolist()
        colors.append(color)
    return img1, img2, colors

# taking the images as input
def InputHouse():
    images_array = []
    #this is the path for the two images in my computer, must be changed to a relevant path of yours
    sourceLeft = cv2.imread('/Users/ablosh/Downloads/r2ya2/house_1.png')
    sourceRight = cv2.imread('/Users/ablosh/Downloads/r2ya2/house_2.png')

    images_array.append(sourceLeft)
    images_array.append(sourceRight)

    return images_array

# function to show the images
def ShowImages(img1, img2):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.title("img1")
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.title("img2")
    plt.show()

# function to draw the lines on the border
def DrawPerimeter(img1, img2, pts1, pts2, colors):
    for i in range(0, 21):
        pt1 = (int(pts1[i][0]), int(pts1[i][1]))
        pt2 = (int(pts2[i][0]), int(pts2[i][1]))
        PT1 = (int(pts1[i + 1][0]), int(pts1[i + 1][1]))
        PT2 = (int(pts2[i + 1][0]), int(pts2[i + 1][1]))
        img1 = cv2.line(img1, pt1, PT1, colors[i], 4)
        img2 = cv2.line(img2, pt2, PT2, colors[i], 4)
    return img1, img2

# compute the triangulation
def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    print('Triangulated point: ')
    print(Vh[3, 0:3] / Vh[3, 3])
    return Vh[3, 0:3] / Vh[3, 3]

# calculate the average of the coordinates
def calculateAverage(p3dsMatrix):
    sum_x = 0
    sum_y = 0
    sum_z = 0
    for coordinate in range(len(p3dsMatrix)):
        sum_x += p3dsMatrix[coordinate][0]
        sum_y += p3dsMatrix[coordinate][1]
        sum_z += p3dsMatrix[coordinate][2]
    avg_x = sum_x / 22
    avg_y = sum_y / 22
    avg_z = sum_z / 22
    return avg_x, avg_y, avg_z


def minus(average_x, average_y, average_z, matrix):
    for coordinate in matrix:
        coordinate[0] = coordinate[0] - average_x
        coordinate[1] = coordinate[1] - average_y
        coordinate[2] = coordinate[2] - average_z
    return matrix


# --------------Make gif --------------
def rotateX(number, matrix):
    r = R.from_euler('xyz', (number, 0, 0), degrees=True)
    rotatedX = r.apply(matrix)  # Rotated points
    return rotatedX


def rotateY(number, matrix):
    r = R.from_euler('xyz', (0, number, 0), degrees=True)
    rotatedY = r.apply(matrix)  # Rotated points
    return rotatedY


if __name__ == '__main__':
    house = InputHouse()

    img1 = house[0]
    img2 = house[1]

    pts1, pts2 = getData()

    # -------- PART A -------
    img1, img2, colors = DrawPoints(img1, img2)
    img1, img2 = DrawPerimeter(img1, img2, pts1, pts2, colors)

    # -------- PART B -------

    P1, P2 = cameraMatrices()
    p3ds = []
    uv1, uv2 = getData()
    uv1 = np.array(uv1, dtype=float)
    uv2 = np.array(uv2, dtype=float)
    P1 = np.array(P1, dtype=float)
    P2 = np.array(P2, dtype=float)

    # triangulation
    for j in range(len(uv1)):
        new_point = DLT(P1, P2, uv1[j], uv2[j])
        p3ds.append(new_point)
    p3ds = np.array(p3ds)

    #calculate average and reduce coordinates
    avg_x, avg_y, avg_z = calculateAverage(p3ds)
    new_matrix = minus(avg_x, avg_y, avg_z, p3ds)
    my_graph = plt.figure()
    graph1 = my_graph.add_subplot(111)
    for i in range(len(p3ds) - 1):
        firstPoint = new_matrix[i]
        secondPoint = new_matrix[i + 1]
        yValue = [(firstPoint[1]), (secondPoint[1])]
        xValue = [(firstPoint[0]), (secondPoint[0])]
        graph1.plot(xValue, yValue)

    plt.show()

    ShowImages(img1, img2)
    #returns tuple with 3 random values as the center of rotation
    number = tuple(np.random.randint(0,360,3))
    r = R.from_euler('xyz', number, degrees=True)
    rotatedXYZ = r.apply(new_matrix)  # Rotated points
    '''A simple example: 3D rotation of points (reference frame: [0,0,0],[x,y,z])'''
    num = -10
    for i in range(37):
        num = num + 10
        new = rotateX(num, rotatedXYZ)
        my_graph = plt.figure()
        graph1 = my_graph.add_subplot(111)
        graph1.axis([-5, 5, -5, 5])

        for j in range(len(new) - 1):
            firstPoint = new[j]
            secondPoint = new[j + 1]
            yValue = [(firstPoint[1]), (secondPoint[1])]
            xValue = [(firstPoint[0]), (secondPoint[0])]
            graph1.plot(xValue, yValue)
        plt.savefig("folder/rotation_x" + str(i) + ".png")

    num =-10
    for z in range(37):
        num = num + 10
        new = rotateY(num, rotatedXYZ)
        my_graph = plt.figure()
        graph1 = my_graph.add_subplot(111)
        graph1.axis([-5, 5, -5, 5])
        for f in range(len(new) - 1):
            firstPoint = new[f]
            secondPoint = new[f + 1]
            yValue = [(firstPoint[1]), (secondPoint[1])]
            xValue = [(firstPoint[0]), (secondPoint[0])]
            graph1.plot(xValue, yValue)
        plt.savefig("folder/rotation_y" + str(z) + ".png")

