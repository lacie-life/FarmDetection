# import dependencies
import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image using cv's imread(nameoffile)
img = cv2.imread("images/test.jpg")

# split the image into blue, green, and red channels
b, g, r = cv2.split(img)
# here we 'amplify' the color green to stand out, without red/blue
gscale = 2 * g - r - b  # we are going to refer to this as our grayscale img
# Canny edge detection
gscale = cv2.Canny(gscale, 280, 290, apertureSize=3)
# checking the results (good practice)
plt.figure()
plt.plot(), plt.imshow(gscale)
plt.title('Canny Edge-Detection Results')
plt.xticks([]), plt.yticks([])
plt.show()

size = np.size(gscale)  # returns the product of the array dimensions
skel = np.zeros(gscale.shape, np.uint8)  # array of zeros
ret, gscale = cv2.threshold(gscale, 128, 255, 0)  # thresholding the image
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False
while (not done):
    eroded = cv2.erode(gscale, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(gscale, temp)
    skel = cv2.bitwise_or(skel, temp)
    gscale = eroded.copy()
zeros = size - cv2.countNonZero(gscale)
if zeros == size:
    done = True

lines = cv2.HoughLines(skel, 1, np.pi / 180, 130)
a, b, c = lines.shape
for i in range(a):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
# showing the results:
plt.subplot(121)
# OpenCV reads images as BGR, this corrects so it is displayed as RGB
plt.plot(), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Row Detection'), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.plot(), plt.imshow(skel, cmap='gray')
plt.title('Skeletal Image'), plt.xticks([]), plt.yticks([])
plt.show()
