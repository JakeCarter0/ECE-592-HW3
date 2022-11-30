"""
Created on Wed June 25 2022

@author: Jake Carter
PiE Homework 3
Includes detectcircles and setectstopsign functions
This code with work on both blob images but it will not detect multiple stop signs

"""
import cv2
import numpy as np
import sys
import math
from matplotlib import pyplot as plt


def angle(i): #Used to sort centerpoints of circles by their angle with respect to the average position
    return i[0]

def detectcircles(filename:str):
    """
    detectcircles(filename:str)
    Takes the filename of an image file containing an image of colored circles on pavement.
    This function detects the locations and sizes of each circle and creates a polygon with vertices at each of the centerpoints of the circles
    """
    try:
        img = cv2.imread(filename)
    except:
        print("Error, invalid input file")
        return
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hMin = 0
        sMin = 40
        vMin = 185
        hMax = 179
        sMax = 255
        vMax = 255
        hsvMin = np.array([hMin, sMin, vMin])
        hsvMax = np.array([hMax, sMax, vMax])
        mask = cv2.inRange(hsv, hsvMin, hsvMax) # Creates a mask based on the hsv values above to isolate teh colorful circles
        median = cv2.medianBlur(mask,11) # Median blurs the image to get rid of some of the noise

        kernel = np.ones((3,3),np.uint8)
        closedMedian = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel) # Closes the holes in the circles by dialating then eroding

        edges = cv2.Canny(closedMedian,70, 150, apertureSize = 5) # Uses canny algorithm to detect edges


        circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,100,param1=60,param2=20,minRadius=20,maxRadius=300) # Uses Hough algorithm to detect circles

        circles = np.uint16(np.around(circles))
        vertices = list()
        for i in circles[0,:]:
            cv2.circle(img,(i[0],i[1]),i[2],(0,0,255),2) # Draws circles
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,str(i[2]),(i[0],i[1]), font, 1, (0, 0, 0), 2,cv2.LINE_AA) # Writes circle radius
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),4) # Draws circle center point
            vertices.append(i[0:2]) # Appends to list of center points

        xAvg = 0
        yAvg = 0
        for i in vertices:
            xAvg += i[0]
            yAvg += i[1]
        xAvg = xAvg / len(vertices)
        yAvg = yAvg / len(vertices)
        unorderedVerticies = list()
        for i in vertices:
            unorderedVerticies.append((np.arctan2(i[1] - yAvg,i[0] - xAvg), i)) # Appends the angle with respect to the average position and the position of each circle center point

        orderedVerticies = sorted(unorderedVerticies, key = angle) # This sorts the points by their polar coordinate angle originating at the average position of the coordinates

        for i in range(0,len(orderedVerticies) - 1):
            cv2.line(img, orderedVerticies[i][1], orderedVerticies[i + 1][1], (255,0,0), 2) # Draws edges connecting each vertex in order
        cv2.line(img, orderedVerticies[0][1], orderedVerticies[-1][1], (255,0,0), 2) # Draws final edge

        logoHeight = 56
        logoWidth = 45
        logo = cv2.resize(cv2.imread("opencvlogo.png"), (logoWidth, logoHeight))

        roi = img[0:logoHeight, 0:logoWidth] # Creates ROI where the logo will go
        mask = cv2.threshold(cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1] # Creates mask of logo
        maskedImg = cv2.bitwise_and(roi,roi,mask = cv2.bitwise_not(mask)) # Uses mask to cut out shape of logo from roi

        img[0:logoHeight, 0:logoWidth] = cv2.add(maskedImg,logo)


        cv2.imshow('detected circles',img)
        keyPress = cv2.waitKey(0)
        if keyPress == ord("s") or keyPress == ord("S"):
            filename = input("Save image as: ")
            cv2.imwrite(filename,img)
        cv2.destroyAllWindows()
    except:
        print("No circles detected")




def detectstopsign(filename:str):
    """
    detectstopsign(filename:str)
    Takes the filename of an image file containing a stop sign
    This function detects the locations and sizes of each stop sign and displays a mask of the stop sign along with the original image with a box around the stop sign.
    Above the stop sign the side length of the stop sign is printed
    """
    try:
        img = cv2.imread(filename)
    except:
        print("Error, invalid input file")
        return
    try:
        RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


        testMask = cv2.inRange(hsv,np.array([0, 180, 90]), np.array([179, 255, 255]))
        contours,hierarchy = cv2.findContours(testMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            equHsv = hsv.copy()
        else:
            isoSat = hsv[:, :, 1] #If the image is desaturated, use a histogram equalizer to increase saturation
            equSat = cv2.equalizeHist(isoSat)
            res = np.hstack((isoSat,equSat))
            equHsv = hsv.copy()
            equHsv[:,:,1] = equSat #Replace saturation with equalized satuaration


        hMin0 = 0
        hMin1 = 159
        sMin = 130
        vMin = 100
        hMax0 = 6
        hMax1 = 179
        sMax = 255
        vMax = 255


        hsvMin0 = np.array([hMin0, sMin, vMin])
        hsvMax0 = np.array([hMax0, sMax, vMax])
        hsvMin1 = np.array([hMin1, sMin, vMin])
        hsvMax1 = np.array([hMax1, sMax, vMax])
        mask0 = cv2.inRange(equHsv, hsvMin0, hsvMax0) #HSV filter to find lower H value reds(0 - 6)
        mask1 = cv2.inRange(equHsv, hsvMin1, hsvMax1) #HSV filter to find higher H value reds(159 - 179)
        mask = cv2.bitwise_or(mask0,mask1)


        kernel = np.ones((2,2),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) #Remove some excess noise from mask
        # kernel = np.ones((2,2),np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Run findContours to find stop sign
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt) #Get bounding recatngle size and location
        # print(x)
        # print(y)
        # print(x+w)
        # print(y+h)
        RGBimg = cv2.rectangle(RGBimg,(x,y),(x+w,y+h),(255, 0,0),1) #Draw red box around stop sign
        size = ((w + h) / 2) / (1 + np.sqrt(2))

        kernel = np.ones((int(2),int(2)),np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) #Closes holes in mask, used for calculating perimeter of stop sign
        contours,hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        perimeter = cv2.arcLength(cnt,True)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(RGBimg,str(math.ceil(perimeter/8)),(x,y - h // 10), font, math.ceil(img.shape[1]/500), (255,0,0), math.ceil(img.shape[0]/200),cv2.LINE_AA) #prints stop sign sidelength
        output = np.concatenate((cv2.cvtColor(RGBimg, cv2.COLOR_RGB2BGR), cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)), axis=1) #Combines mask with original image
        width = 1200 #Width of window
        height = int((width / output.shape[1]) * output.shape[0]) #Scaled heigh of window
        output = cv2.resize(output, (width, height)) #resizes output


        cv2.imshow('Detected Stopsigns',output)


        keyPress = cv2.waitKey(0)
        if keyPress == ord("s") or keyPress == ord("S"):
            filename = input("Save image as: ")
            cv2.imwrite(filename,img)
        cv2.destroyAllWindows()
    except:
        print("No stop sign detected")
        return

# Uncomment this block to run all tests:
# detectcircles("ColorBlobs.jpg")
# detectcircles("ColorBlobs5.jpg")
# detectstopsign("Stop1.png")
# detectstopsign("Stop3.jpg")
# detectstopsign("Stop4.jpg")
# detectstopsign("Stop5.jpg")
