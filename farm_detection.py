import os
import cv2

img = cv2.imread("images/test.jpg", -1)

cv2.imshow("Test", img)

cv2.waitKey(0)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)