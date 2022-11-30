
# Python program to compute and visualize the
# histogram of image for all three channels


# importing libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# reading the input image
img = cv2.imread('Stop1.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# define colors to plot the histograms


isoSat = hsv[:, :, 1]
equSat = cv2.equalizeHist(isoSat)
res = np.hstack((isoSat,equSat))
equHsv = hsv.copy()
# equHsv[:,:,1] = equSat


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

mask0 = cv2.inRange(equHsv, hsvMin0, hsvMax0)
mask1 = cv2.inRange(equHsv, hsvMin1, hsvMax1)

mask = cv2.bitwise_or(mask0,mask1)
kernel = np.ones((2,2),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
print(cnt)
x,y,w,h = cv2.boundingRect(cnt)
print(x)
print(y)
print(x+w)
print(y+h)
output = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
cv2.drawContours(img,contours,-1,(0,255,0),1)
cv2.imshow("Keypoints", output)

cv2.waitKey(0)


# cv2.imshow('detected circles',cv2.cvtColor(equHsv, cv2.COLOR_HSV2BGR))
# keyPress = cv2.waitKey(0)
# cv2.destroyAllWindows()


# hist = cv2.calcHist([hsv],[2],None,[256],[0,256])
# # histEq = cv2.calcHist([equHsv],[1],None,[256],[0,256])
# plt.plot(hist)
# # plt.plot(histEq)
# plt.title('Image Histogram GFG')
# plt.show()
