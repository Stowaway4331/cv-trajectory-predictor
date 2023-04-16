import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# Initialize the Video
cap = cv2.VideoCapture("Videos/vid (4).mp4")

# Create the ColorFinder Object
colorFinder = ColorFinder(False)
hsvVals = {"hmin": 6, "smin": 130, "vmin": 0, "hmax": 16, "smax": 255, "vmax": 255}

# Variables
posListX = []
posListY = []
xList = [item for item in range(0, 1300)]

showPrediction = 0

while True:
    # Grab the image
    success, img = cap.read()
    # img = cv2.imread("Videos/Ball.png")
    img = img[0:900, :]

    # Find the Color of the ball
    imgColor, mask = colorFinder.update(img, hsvVals)

    # Find location of the ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=200)

    if contours:
        posListX.append(contours[0]["center"][0])
        posListY.append(contours[0]["center"][1])

    if posListX:
        # Polynomial Regression y = Ax^2 + Bx + C
        # Find the coefficients
        A, B, C = np.polyfit(posListX, posListY, 2)

        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)
            cv2.circle(imgContours, pos, 10, (0, 255, 0), cv2.FILLED)
            if i == 0:
                cv2.line(imgContours, pos, pos, (0, 255, 0), 2)
            else:
                cv2.line(
                    imgContours, pos, (posListX[i - 1], posListY[i - 1]), (0, 255, 0), 5
                )
        for x in xList:
            y = int(A * x**2 + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

        if len(posListX) == 10:
            # Prediction
            # X = 330 t0 430, Y = 590

            a = A
            b = B
            c = C - 590

            x = (-b - math.sqrt(b**2 - (4 * a * c))) / (2 * a)

            if 320 < x < 430:
                showPrediction = 1
            else:
                showPrediction = -1

    # Display
    # img = cv2.resize(img, (0, 0), None, 0.7, 0.7)
    # imgColor = cv2.resize(imgColor, (0, 0), None, 0.7, 0.7)
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    # cv2.imshow("ImageColor", imgColor)
    if showPrediction == 1:
        cvzone.putTextRect(
            imgContours,
            "Basket",
            (50, 200),
            scale=4,
            thickness=3,
            colorR=(0, 200, 0),
            offset=20,
        )
    elif showPrediction == -1:
        cvzone.putTextRect(
            imgContours,
            "No Basket",
            (50, 200),
            scale=4,
            thickness=3,
            colorR=(0, 0, 200),
            offset=20,
        )
    cv2.imshow("Image Contours", imgContours)
    # cv2.imshow("Image", img)
    cv2.waitKey(100)
