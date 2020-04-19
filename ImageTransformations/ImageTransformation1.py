## Example provided in URL -- https://github.com/vxy10/ImageAugmentation
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

import numpy as np
#matplotlib inline
import matplotlib.image as mpimg

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness


    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img


def shear_image(img):
    height_original_image = int(img.shape[0])
    width_original_image = int(img.shape[1])
    for shear in np.arange(-1.0, 1.0, 0.1):
        M2 = np.float32([[1, shear, 0], [0, 1, 0]])
        # M3 = np.float32([[1, 0, 0], [shear, 1, 0]])
        print(shear)
        shear_image = cv2.warpAffine(img, M2, (width_original_image, height_original_image))  # 3rd argument -- size of the output image...
        # cv2.imshow('Horizontal_Shear_image', shear_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return shear_image


image = mpimg.imread('1479425818246842936.jpg')
plt.imshow(image);
plt.axis('off');

gs1 = gridspec.GridSpec(10, 10)
gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
plt.figure(figsize=(10,10))
for i in range(100):
    ax1 = plt.subplot(gs1[i])
    ax1.set_xticklabels([i])
    ax1.set_yticklabels([1])
    ax1.set_aspect('equal')
    #img = transform_image(image,20,10,5,brightness=1)
    img = shear_image(image)
    plt.subplot(10,10,i+1)
    plt.imshow(img)
    plt.axis('off')

plt.show()