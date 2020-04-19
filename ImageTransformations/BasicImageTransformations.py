'''
Sairam
Basic image transformations : Blur, Brightness, Contrast

'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
import os
#img = cv2.imread('img1.png', cv2.IMREAD_COLOR) #https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_basic_image_operations_pixel_access_image_load.php


#ImageTransformations/1479425774739119337.jpg

img = cv2.imread('1479425774739119337.jpg')

print (img.shape)
#print (img[440,350])
rows = 480
cols = 640

def image_blur(img,matrix):
    blur_modified_image = []
    '''
    if matrix == 3:
        blur_modified_image = cv2.blur(img, (3,3))
    if matrix == 4:
        blur_modified_image = cv2.blur(img, (4,4))
    if matrix == 5:
        blur_modified_image = cv2.blur(img, (5,5))
    if matrix == 6:
        blur_modified_image = cv2.blur(img, (6,6))
    '''
    blur_modified_image = cv2.blur(img, (3,3))
    return blur_modified_image

def image_brightness(img,beta):
    brightness_modified_image = []
    if beta == 10:
        b, g, r = cv2.split(img)
        b = cv2.add(b, 20)
        g = cv2.add(g, 40)
        r = cv2.add(r, 30)
        brightness_modified_image = cv2.merge((b, g, r))
    if beta == 50:
        b, g, r = cv2.split(img)
        b = cv2.add(b, 50)
        g = cv2.add(g, 60)
        r = cv2.add(r, 90)
        brightness_modified_image = cv2.merge((b, g, r))
    if beta == 100:
        b, g, r = cv2.split(img)
        b = cv2.add(b, 120)
        g = cv2.add(g, 145)
        r = cv2.add(r, 210)
        brightness_modified_image = cv2.merge((b, g, r))
    '''
    b, g, r = cv2.split(img)
    b = cv2.add(b, 40)
    g = cv2.add(g, 50)
    r = cv2.add(r, 25)
    brightness_modified_image = cv2.merge((b,g,r)) 
    '''
    return brightness_modified_image
'''
def image_brightness(img):
    b, g, r = cv2.split(img)
    b = cv2.add(b, 40)
    g = cv2.add(g, 50)
    r = cv2.add(r, 25)
    brightness_modified_image = cv2.merge((b,g,r))
    return brightness_modified_image
'''

def image_contrast(img, alpha):
    contrast_modified_image = cv2.multiply(img, alpha)
    return contrast_modified_image

#def image_translation(img,):


blurred_image = image_blur(img,1)
contrast_adjusted_image = image_contrast(img, 1.2) # adjust 2nd parameter to modify constart level
brightness_adjusted_image = image_brightness(img,30)

with open('BasicImageTranslation-output.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    path = './outputImages'
    counter = 1

    for row in readCSV:
        #print(row)
        #print(row[0])
        print('Blur  '+ row[0],'Brightness   '+ row[1],'Contrast   '+row[2],)
        #print(type(row[2]) )
        #print(type(row[1]))
        #print(row[2])
        #print(row[1])
        #print (row[0][0])
        blurred_image = image_blur(img, row[0][0])
        contrast_adjusted_image = image_contrast(blurred_image, float(row[2]))  # adjust 2nd parameter to modify constrast level
        brightness_adjusted_image = image_brightness(contrast_adjusted_image, int(row[1]))

        cv2.imshow('img', brightness_adjusted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #writing image to a folder
        fileExtension='.jpg'
        sourceFileName= '1479425774739119337_'
        fileName = sourceFileName + str(counter) + '.jpg'
        outputFileDestination = './outputImages/'+fileName
        print(outputFileDestination)
        cv2.imwrite(outputFileDestination,brightness_adjusted_image)
        counter = counter+1

'''
plt.subplot(161),plt.imshow(img),plt.title('Original')
plt.subplot(162),plt.imshow(blurred_image),plt.title('Blurred - 12 x 12')
plt.subplot(163),plt.imshow(brightness_adjusted_image),plt.title('Brightness')
plt.subplot(164),plt.imshow(image_blur(brightness_adjusted_image)),plt.title('Brightness + Blur')
plt.subplot(165), plt.imshow(contrast_adjusted_image), plt.title('Contrast')
plt.subplot(166), plt.imshow(image_blur(contrast_adjusted_image)), plt.title('Contrast + Blur')
#plt.subplot(266), plt.imshow(contrast_adjusted_image), plt.title('Contrast')
plt.show()

# Translation -
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Rotation -

# cols-1 and rows-1 are the coordinate limits.
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(cols,rows))
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Scale
'''
#Blur
blur_3x3 = cv2.blur(img,(3,3))
blur_4x4 = cv2.blur(img,(4,4))
blur_5x5 = cv2.blur(img,(5,5))
blur_6x6 = cv2.blur(img,(8,8))

#Gaussian_blur
Gaussian_blur_3x3 = cv2.GaussianBlur(img,(3,3),0)
Gaussian_blur_5x5 = cv2.GaussianBlur(img,(5,5),0)
Gaussian_blur_7x7 = cv2.GaussianBlur(img,(7,7),0)
Gaussian_blur_6x6 = cv2.blur(img,(6,6),0)

#Brightness

b,g,r =  cv2.split(img)
b = cv2.add(b, 40)
g = cv2.add(g, 50)
r = cv2.add(r, 25)
brightness_adjusted_img = cv2.merge((b, g, r))


b,g,r = cv2.split(blur_5x5)
b = cv2.add(b, 40)
g = cv2.add(g, 50)
r = cv2.add(r, 25)
blur_and_brightness = cv2.merge((b, g, r))
#blur_and_brightness = cv2.add(blur_5x5, np.array([20]))

#Contrast
contrast_adjusted_image = cv2.multiply(img, np.array([2.4]))    # https://github.com/abidrahmank/OpenCV2-Python/blob/master/Official_Tutorial_Python_Codes/2_core/BasicLinearTransforms.py

blur_brightness_and_contrast = cv2.multiply(blur_and_brightness, np.array([2.4]))

plt.subplot(161),plt.imshow(img),plt.title('Original')
plt.subplot(162),plt.imshow(blur_5x5),plt.title('Blurred - 5 x 5')
plt.subplot(163),plt.imshow(brightness_adjusted_img),plt.title('Brightness')
plt.subplot(165),plt.imshow(contrast_adjusted_image),plt.title('Contrast')
#plt.subplot(165),plt.imshow(blur_6x6),plt.title('Blurred - 6 x 6')
plt.subplot(164), plt.imshow(blur_and_brightness), plt.title('Blur and Brightness')
plt.subplot(166), plt.imshow(contrast_adjusted_image), plt.title('Contrast')
'''
'''
plt.subplot(251),plt.imshow(img),plt.title('Original')
plt.subplot(252),plt.imshow(Gaussian_blur_3x3),plt.title('Gaussian_blur - 3 x 3')
plt.subplot(253),plt.imshow(Gaussian_blur_5x5),plt.title('Gaussian_blur - 4 x 4')
plt.subplot(254),plt.imshow(Gaussian_blur_7x7),plt.title('Gaussian_blur - 5 x 5')
plt.subplot(255),plt.imshow(Gaussian_blur_6x6),plt.title('Gaussian_blur - 6 x 6')
'''
'''
plt.show()

'''