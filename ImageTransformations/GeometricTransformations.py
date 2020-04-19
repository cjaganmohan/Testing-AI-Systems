import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib import image as mpimg
import csv
import os

img = cv2.imread('1479425751884901474.jpg')
#img = cv2.imread('1479425818246842936.jpg')

#print (img.shape)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Brightness and Contrast  -- new_img = alpha*old_img + beta
alpha = 1.5 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)
# adjusted_brightness = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
# adjusted_contrast = cv2.convertScaleAbs(img, alpha=1.0, beta=0)
# adjusted_brightness_contrast = cv2.convertScaleAbs(img, alpha=1.0, beta=100)
# cv2.imshow("Brightness + Contrast", np.hstack((adjusted_brightness, adjusted_contrast,adjusted_brightness_contrast)))
# cv2.waitKey(0)  # waits until a key is pressed
# cv2.destroyAllWindows()  # destroys the window showing image
# adjusted_brightness_contrast2 = cv2.convertScaleAbs(img, alpha=0.1, beta=0)
# adjusted_brightness_contrast2a = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
# adjusted_brightness_contrast3 = cv2.convertScaleAbs(img, alpha=2.0, beta=0)
# cv2.imshow("Brightness + Contrast + Extreme", np.hstack((adjusted_brightness_contrast2, adjusted_brightness_contrast2a,adjusted_contrast,adjusted_brightness_contrast3)))
# cv2.waitKey(0)  # waits until a key is pressed
# cv2.destroyAllWindows()  # destroys the window showing image

# Contrast
for contrast_value in np.arange(0,15,0.5):
    alpha = contrast_value  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    print(alpha, beta)
    adjusted_contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imshow("Contrast", adjusted_contrast )
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image

# Brightness
for brightness_value in np.arange(0,160,10):
    alpha = 1  # Contrast control (1.0-3.0)
    beta = brightness_value # Brightness control (0-100)
    print(alpha, beta)
    adjusted_brightness = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imshow("Brightness", adjusted_brightness )
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image

# Blur - Gaussian blur
for blur_value in np.arange(3,33,2):
    print((blur_value, blur_value))
    gaussian_blur_image = cv2.GaussianBlur(img, (blur_value, blur_value), cv2.BORDER_DEFAULT)
    blur_image = cv2.blur(img,(blur_value, blur_value) )
    median_blur = cv2.medianBlur(img,blur_value)
    #bilateral_blur = cv2.bilateralFilter(img, 9, 150, 150)
    cv2.imshow("Gaussian blur", np.hstack((blur_image, gaussian_blur_image,median_blur)))
    #cv2.imshow("bilateralFilter", bilateral_blur)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image

#Rotation -- https://www.tutorialkart.com/opencv/python/opencv-python-rotate-image/#
height_original_image = int(img.shape[0])
width_original_image = int(img.shape[1])
center_position_of_the_image = (width_original_image / 2, height_original_image / 2 )
angle60 = -5
angle90 = 180
scale = 1.0 # rotate a given image WHILE retaining the image size
for rotating_angle in np.arange(-50, 60, 10):
    print(rotating_angle)
    M = cv2.getRotationMatrix2D(center_position_of_the_image, rotating_angle, scale)
    rotated_image = cv2.warpAffine(img, M, (width_original_image, height_original_image))
    cv2.imshow('Rotated_image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


scale_percent = 5
#Scaling -- resizing -- https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
for scale_percent in np.arange(50,160, 10):
    print(scale_percent)
    width = int(img.shape[1] * scale_percent / 100)  # img.shape --> height * width
    height = int(img.shape[0] * scale_percent / 100)
    scaled_image_dimension = (width, height)
    # resizing the image
    resized_image = cv2.resize(img, scaled_image_dimension, interpolation=cv2.INTER_AREA)
    cv2.imshow('Scaled_image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Shearing
for shear in np.arange(-1.0, 1.1, 0.1):
    M2 = np.float32([[1, shear, 0], [0, 1, 0]])
    #M3 = np.float32([[1, 0, 0], [shear, 1, 0]])
    print(shear)
    shear_image = cv2.warpAffine(img, M2, (width_original_image, height_original_image))  # 3rd argument -- size of the output image...
    cv2.imshow('Horizontal_Shear_image', shear_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Shearing
for shear in np.arange(-1.0, 1.1, 0.1):
    #M2 = np.float32([[1, shear, 0], [0, 1, 0]])
    M3 = np.float32([[1, 0, 0], [shear, 1, 0]])
    print(shear)
    shear_image = cv2.warpAffine(img, M3, (width_original_image, height_original_image))  # 3rd argument -- size of the output image...
    cv2.imshow('Vertical_Shear_image', shear_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Translation
for translation in np.arange(-100, 110, 10):
    print(translation)
    M1 = np.float32([[1, 0, translation], [0, 1, translation]])
    translated_image = cv2.warpAffine(img, M1, (width_original_image, height_original_image))  # 3rd argument -- size of the output image...
    cv2.imshow('translated_image', translated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Contrast
for alpha in np.arange(0, 10.2, 0.2):
    contrast_adjusted_image = cv2.multiply(img, np.array([alpha]))
    print(alpha)
    cv2.imshow('contrast_adjusted_image', np.hstack((img,contrast_adjusted_image)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Brightness  -- new_img = alpha*old_img + beta [alpha --> contrast; beta --> brightness
#for beta in np.arange(0, 10.0, 0.2):

