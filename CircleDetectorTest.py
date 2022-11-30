import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt


def detectcircles(filename:str):

    img = cv2.imread(filename)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hMin = 0
    sMin = 40
    vMin = 185
    hMax = 179
    sMax = 255
    vMax = 255
    hsvMin = np.array([hMin, sMin, vMin])
    hsvMax = np.array([hMax, sMax, vMax])
    mask = cv2.inRange(hsv, hsvMin, hsvMax)
    blur = cv2.GaussianBlur(mask,(9,9),3)
    median0 = cv2.medianBlur(mask,7)
    median1 = cv2.medianBlur(mask,11)
    median2 = cv2.medianBlur(mask,11)
    edges = cv2.Canny(blur,70, 150, apertureSize = 3)
    edges0 = cv2.Canny(median0,70, 150, apertureSize = 5)
    edges1 = cv2.Canny(median1,70, 150, apertureSize = 5)


    kernel = np.ones((3,3),np.uint8)
    openedMask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closedMedian = cv2.morphologyEx(median1, cv2.MORPH_CLOSE, kernel)
    closedMedian1 = cv2.morphologyEx(closedMedian, cv2.MORPH_CLOSE, kernel)
    closedMedian2 = cv2.morphologyEx(closedMedian1, cv2.MORPH_CLOSE, kernel)

    edges2 = cv2.Canny(closedMedian,70, 150, apertureSize = 5)

    # cv2.imshow('original image',img)
    # cv2.imshow('HSV mask',mask)
    # cv2.imshow('blurred mask',blur)
    # cv2.imshow('Edge detected',edges)
    #
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
    f = plt.figure()
    f.set_figwidth(16)
    f.set_figheight(5)
    # plt.subplot(151),plt.imshow(img,cmap = 'gray')
    # plt.title('Original image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(152),plt.imshow(mask,cmap = 'gray')
    # plt.title('HSV mask'), plt.xticks([]), plt.yticks([])
    # plt.subplot(153),plt.imshow(blur,cmap = 'gray')
    # plt.title('Gaussian blurred mask'), plt.xticks([]), plt.yticks([])
    # plt.subplot(154),plt.imshow(median,cmap = 'gray')
    # plt.title('Median blurred mask'), plt.xticks([]), plt.yticks([])
    # plt.subplot(155),plt.imshow(edges,cmap = 'gray')
    # plt.title('Median edge image'), plt.xticks([]), plt.yticks([])


    # plt.subplot(241),plt.imshow(mask,cmap = 'gray')
    # plt.title('HSV mask'), plt.xticks([]), plt.yticks([])
    # plt.subplot(242),plt.imshow(median0,cmap = 'gray')
    # plt.title('Median blurred mask 0'), plt.xticks([]), plt.yticks([])
    # plt.subplot(243),plt.imshow(median1,cmap = 'gray')
    # plt.title('Median blurred maskc1'), plt.xticks([]), plt.yticks([])
    # plt.subplot(244),plt.imshow(median2,cmap = 'gray')
    # plt.title('Median blurred mask 2'), plt.xticks([]), plt.yticks([])
    # plt.subplot(245),plt.imshow(closedMedian,cmap = 'gray')
    # plt.title('Closed'), plt.xticks([]), plt.yticks([])
    # plt.subplot(246),plt.imshow(closedMedian1,cmap = 'gray')
    # plt.title('Closed'), plt.xticks([]), plt.yticks([])
    # plt.subplot(247),plt.imshow(closedMedian2,cmap = 'gray')
    # plt.title('Closed'), plt.xticks([]), plt.yticks([])
    # plt.subplot(248),plt.imshow(edges2,cmap = 'gray')
    # plt.title('Median edge image 2'), plt.xticks([]), plt.yticks([])
    #
    # plt.show()

    circleTarget = edges2
    circles = cv2.HoughCircles(circleTarget,cv2.HOUGH_GRADIENT,1,100,param1=60,param2=20,minRadius=20,maxRadius=300)
    #circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,250,param1=60,param2=32,minRadius=0,maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(255,255,255),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,str(i[2]),(i[0],i[1]), font, 1, (0, 0, 0), 2,cv2.LINE_AA)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detectcircles("ColorBlobs.jpg")
