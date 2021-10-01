import os
import cv2
import numpy as np

img = cv2.imread("images/test.jpg", -1)

cv2.imshow("Test", img)
kernel = np.ones((5,5),np.uint8)

cv2.waitKey(0)

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, img_gray = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

# Implement median filter
img_blur = cv2.medianBlur(img_gray, ksize=5)

img_edge = cv2.Laplacian(img_blur,cv2.CV_64F)

img_edge = np.uint8(img_edge)

img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_GRADIENT, kernel)

cv2.imshow("Check" , img_edge)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=img_edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

print(type(img_edge))

h, w = img_edge.shape

maxLabelSize = (h/4.0) * (w/6.0)
minLabelSize = ((h/40.0) * (w/60.0))

goodContours = []
for i in range(len(contours)):
    size = cv2.contourArea(contours[i])
    if maxLabelSize > size > minLabelSize:
        goodContours.append(contours[i])

# draw contours on the original image
image_copy = img_blur.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

cv2.imshow( "Contours", image_copy)
cv2.waitKey(0)


