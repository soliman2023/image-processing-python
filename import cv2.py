import cv2
import NumPy as np

img_path = r"Screenshot_824.jpeg"
img = cv2.imread(img_path)
img = cv2.resize(img, (1280, 720))
cv2.imshow("Color Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img.shape
b, g, r = cv2.split(img)
zeros_ch = np.zeros(img.shape[0:2], dtype="uint8")
blue_img = cv2.merge([b, zeros_ch, zeros_ch])
cv2.imshow("Blue Image", blue_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
green_img = cv2.merge([zeros_ch, g, zeros_ch])
cv2.imshow("Green Image", green_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
red_img = cv2.merge([zeros_ch, zeros_ch, r])
cv2.imshow("Red Image", red_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# importing library for plotting
from matplotlib import pyplot as plt

# find frequency of pixels in range 0-255
histr = cv2.calcHist([gray_img],[0],None,[256],[0,256])
  
# show the plotting graph of an image
plt.plot(histr)
plt.show()
equ = cv2.equalizeHist(gray_img)
plt.plot(equ)
plt.show()
#Add salt and pepper noise to the image.
import random 
def add_noise(image):
 
    # Getting the dimensions of the image
    row , col , ch = image.shape
     
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to white
        image[y_coord][x_coord] = 255
         
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to black
        image[y_coord][x_coord] = 0
         
    return image

add_noise(img)
cv2.imshow("Salt_pepper_noise", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('Screenshot_824.jpeg')
blur = cv.blur(img,(5,5))
cv2.imshow("Original", img)
cv2.imshow("Blurred", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
from matplotlib import pyplot as plt

# loading image
img0 = cv2.imread('Screenshot_824.jpeg',)

# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(gray,(3,3),0)

# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()