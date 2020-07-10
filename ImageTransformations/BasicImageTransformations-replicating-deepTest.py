'''
Sairam


'''
import csv
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

#img = cv2.imread('img1.png', cv2.IMREAD_COLOR) #https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_basic_image_operations_pixel_access_image_load.php

# 1479425660620933516.jpg ,  -- [-0.9, <-0.8] - Group 2
# 1479425534498905778.jpg, -0.77587604692 -- [-0.8, <-0.7] - Group 3
# 1479425619063583545.jpg, -0.6527531147 -- [-0.7, <-0.6] - Group 4
# 1479425660020827157.jpg, -0.512664185019 -- [-0.6, <-0.5] - Group 5
# 1479425535099058605.jpg, -0.471813351582 -- [-0.5, <-0.4] - Group 6
# 1479425496442340584.jpg, -0.34557518363 -- [-0.4, <-0.3] - Group 7
# 1479425537999541591.jpg, -0.272028333527 -- [-0.3, <-0.2] - Group 8
# 1479425719031130839.jpg, -0.130899697542 -- [-0.2, <-0.1] - Group 9
# 1479425712029955155.jpg, -0.0572511968599 -- [-0.1, <0] -- Group 10
# 1479425706078866287.jpg, 0.00872664619237 -- [0, <0.1] -- Group 11
# 1479425527947728896.jpg, 0.190240889788 -- [0.1, <0.2] -- Group 12

# 1479425468287290727.jpg, 0.269597201956 -- [0.2, <0.3] -- Group 13
# 1479425470287629689.jpg, 0.383685899418 -- [0.3, <0.4] -- Group 14
# 1479425499292775434.jpg, 0.423279553138 -- [0.4, <0.5] -- Group 15
# 1479425488540828515.jpg, 0.518024973883 -- [0.5, <0.6] -- Group 16
# 1479425652219428572.jpg, 0.661356496233 -- [0.6, <0.7] -- Group 17
# 1479425654520220380.jpg, 0.738851710946 -- [0.7, <0.8] -- Group 18
# 1479425654069742649.jpg, 0.877266222344 -- [0.8, <0.9] -- Group 19
# 1479425653569688917.jpg, 0.939273793505 -- [0.9, <1.0] -- Group 20


#ImageTransformations/1479425751884901474.jpg
image_list={} #
image_list={2:'1479425660620933516.jpg',
            3: '1479425534498905778.jpg',  #-0.77587604692 - - [-0.8, < -0.7] - Group3
            4: '1479425619063583545.jpg',  #-0.6527531147 -- [-0.7, <-0.6] - Group 4
            5:'1479425660020827157.jpg',  #-0.512664185019 -- [-0.6, <-0.5] - Group 5
            6:'1479425535099058605.jpg',  #-0.471813351582 -- [-0.5, <-0.4] - Group 6
            7:'1479425496442340584.jpg',  #-0.34557518363 -- [-0.4, <-0.3] - Group 7
            8:'1479425537999541591.jpg',  #-0.272028333527 -- [-0.3, <-0.2] - Group 8
            9:'1479425719031130839.jpg',  #-0.130899697542 -- [-0.2, <-0.1] - Group 9
            10:'1479425712029955155.jpg',  #-0.0572511968599 -- [-0.1, <0] -- Group 10
            11:'1479425706078866287.jpg',  #0.00872664619237 -- [0, <0.1] -- Group 11
            12:'1479425527947728896.jpg',  #0.190240889788 -- [0.1, <0.2] -- Group 12

            13: '1479425468287290727.jpg',  # 0.269597201956 -- [0.2, <0.3] -- Group 13
            14: '1479425470287629689.jpg',  # 0.383685899418 -- [0.3, <0.4] -- Group 14
            15: '1479425499292775434.jpg',  # 0.423279553138 -- [0.4, <0.5] -- Group 15
            16: '1479425488540828515.jpg',  # 0.518024973883 -- [0.5, <0.6] -- Group 16
            17: '1479425652219428572.jpg',  # 0.661356496233 -- [0.6, <0.7] -- Group 17
            18: '1479425654520220380.jpg',  # 0.738851710946 -- [0.7, <0.8] -- Group 18
            19: '1479425654069742649.jpg',  # 0.877266222344 -- [0.8, <0.9] -- Group 19
            20: '1479425653569688917.jpg',  # 0.939273793505 -- [0.9, <1.0] -- Group 20
            }
threshold = 0.3
group = 20
print(image_list[group])
#input_file_name = '1479425712029955155.jpg'
input_file_name = image_list[group]
input_file_actual_steering_value = 'dummyValue'
img = cv2.imread(input_file_name)

print (img.shape)
#print (img[440,350])
#rows = 480
#cols = 640


def image_blur(img,blur):
    blur_type = blur[0]
    blur_value = int(blur[1])
    if (blur_type == 'B0'):  # Do Not apply blur
        return img
    if(blur_type == 'B1'):  #Averaging blur
        averaging_blur_image = cv2.blur(img, (blur_value, blur_value))
        #print('average')
        return averaging_blur_image
    if (blur_type == 'B2'):  #Gaussian blur
        gaussian_blur_image = cv2.GaussianBlur(img, (blur_value, blur_value), 0)
        #print('gaussian')
        return gaussian_blur_image
    if (blur_type == 'B3'):  #Median blur
        median_blur = cv2.medianBlur(img, blur_value)
        #print('median')
        return median_blur
    if (blur_type == 'B4'):  #Bilateral blur
        bilateral_blur = cv2.bilateralFilter(img, 9, 75, 75)
        #print('Bilateral-blur')
        return bilateral_blur


# def image_brightness_and_contrast(img, beta, alpha):
#     adjusted_brightness_and_contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
#     return adjusted_brightness_and_contrast

def image_brightness(img, param):
    if(param != 0):
        new_img = cv2.add(img, param)  # new_img = img*alpha + beta
        return new_img
    if(param == 0):
        return img


def image_contrast(img, params):
    if(params != 0.0):
        alpha = params
        new_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha
        return new_img
    if(params == 0.0):
        return img


# def image_rotation(img, rotating_angle):
#     #print(img.shape)
#     height_original_image = int(img.shape[0])
#     width_original_image = int(img.shape[1])
#     #print(rotating_angle)
#     #rotating_angle = 0
#     #print(rotating_angle)
#     center_position_of_the_image = (width_original_image / 2, height_original_image / 2)
#     scale = 1.0  # rotate a given image WHILE retaining the image size
#     M = cv2.getRotationMatrix2D(center_position_of_the_image, rotating_angle, scale)
#     rotated_image = cv2.warpAffine(img, M, (width_original_image, height_original_image))
#     return rotated_image

def image_rotation(img, params):
    if (params != 0):
        rows, cols, ch = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
        new_img = cv2.warpAffine(img, M, (cols, rows))
        return new_img
    if (params == 0):
        return img


# def image_scale(img, scale_percent):
#     #print(img.shape, scale_percent)
#     width = int(img.shape[1] * scale_percent / 100)  # img.shape --> height * width
#     height = int(img.shape[0] * scale_percent / 100)
#     scaled_image_dimension = (width, height)
#     # resizing the image
#     resized_image = cv2.resize(img, scaled_image_dimension, interpolation=cv2.INTER_AREA)
#     #print(type(resized_image))
#     return resized_image

def image_scale(img, scale_percent):
    if(scale_percent != 0.0):
        new_img = cv2.resize(img, None, fx=scale_percent, fy=scale_percent, interpolation=cv2.INTER_CUBIC)
        return new_img
    if(scale_percent == 0.0):
        return img

# def image_shear(img, shearing_value):
#     shear_type = shearing_value[0]
#     shear = float(shearing_value[1])
#     height_original_image = int(img.shape[0])
#     width_original_image = int(img.shape[1])
#     if(shear_type == 'HS'):
#         M2 = np.float32([[1, shear, 0], [0, 1, 0]])
#         #print(shear)
#         h_shear_image = cv2.warpAffine(img, M2, (
#         width_original_image, height_original_image))  # 3rd argument -- size of the output image...
#         return h_shear_image
#     # if (shear_type == 'VS'):
#     #     M2 = np.float32([[1, shear, 0], [0, 1, 0]])
#     #     #print(shear)
#     #     v_shear_image = cv2.warpAffine(img, M2, (
#     #         width_original_image, height_original_image))  # 3rd argument -- size of the output image...
#     #     return v_shear_image

def image_shear(img, params):
    if (params[1] != 0.0):
        rows, cols, ch = img.shape
        #factor = float(params[1])*(-1.0)
        factor = float(params[1])
        M = np.float32([[1, factor, 0], [0, 1, 0]])
        new_img = cv2.warpAffine(img, M, (cols, rows))
        return new_img
    if (params[1] == 0.0):
        return img


# def image_translation(img, translation):
#     height_original_image = int(img.shape[0])
#     width_original_image = int(img.shape[1])
#     M1 = np.float32([[1, 0, translation], [0, 1, translation]])
#     translated_image = cv2.warpAffine(img, M1, (
#     width_original_image, height_original_image))  # 3rd argument -- size of the output image...
#     return translated_image

def image_translation(img, params):
    if(params != 0):
        rows, cols, ch = img.shape
        # M1 = np.float32([[1, 0, translation], [0, 1, translation]])
        M = np.float32([[1, 0, params], [0, 1, params]])
        new_img = cv2.warpAffine(img, M, (cols, rows))
        return new_img
    if(params == 0):
        return img


input_test_file = '/home/jagan/Desktop/self-driving-car-project/t-way_testFiles/testFiles/Threshold_' + str(
    threshold) + '/2-way/Udacity-Self-Driving-Rambo_Threshold_' + str(threshold) + '_Grp' + str(group)+'.csv'
#input_test_file = '/Users/Jagan/Desktop/Udacity-Self-Driving-Rambo_Threshold_0.1_Grp'+str(group)+'.csv'
output_file = '/home/jagan/Desktop/Rambo_2-way_Grp' + str(group) +'.csv'
print(input_test_file)
with open(input_test_file) as input_csv_file, open(output_file, 'wb') as output_csv_file:
#with open(input_test_file) as input_csv_file:
    readCSV = csv.reader(input_csv_file, delimiter=',')
    writeCSV = csv.writer(output_csv_file, delimiter=',' , quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writeCSV.writerow(['frame_id', 'steering_angle'])  # header information for the output.csv file
    #path = '/Users/Jagan/Desktop/2-way/Grp'+ str(group)+'/'
    counter = 1
    ''' to skip first seven rows in ACTS file, calling next() for seven times'''
    next(readCSV) # skip a row in input CSV file
    next(readCSV) # skip a row in input CSV file
    next(readCSV) # skip a row in input CSV file
    next(readCSV) # skip a row in input CSV file
    next(readCSV) # skip a row in input CSV file
    next(readCSV) # skip a row in input CSV file
    next(readCSV)
    for row in readCSV:
        #print(row)
        #print(row[0])
        tc_info = 'Blur  '+ row[0],'Brightness   '+ row[1],'Contrast   '+row[2], 'Rotation  '+ row[3],'Scaling   '+ row[4],'Shearing   '+row[5],'Translation  '+ row[6]
        #print(tc_info)
        blur = str(row[0]).split('-')  # [Blur_type, Blur_value]
        beta =  int(row [1]) # Brightness control (0-100)
        alpha = float(row [2]) # Contrast control (1.0-3.0)
        rotation_angle = int(row [3])
        scaling_percent_size = float(row[4])  # MODIFIED BY JAGAN TO ACCOMODATE DEEP-TEST BASED PARAMETER VALUES
        shearing_k_value = str(row[5]).split(':-')  # [shearing_type, shearing_value]
        #print(shearing_k_value)
        translation = int(row[6])

        blur_applied_image = image_blur(img,blur)
        brightness_applied_image = image_brightness(blur_applied_image, beta )
        contrast_applied_image = image_contrast(brightness_applied_image, alpha)
        rotation_applied_image = image_rotation(contrast_applied_image, rotation_angle)
        scaling_applied_image = image_scale(rotation_applied_image, scaling_percent_size)
        shearing_applied_image = image_shear (scaling_applied_image,shearing_k_value)
        translation_applied_image = image_translation (shearing_applied_image, translation)


        # cv2.imshow('img', translation_applied_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #writing image to a folder
        fileExtension='.jpg'
        imageNumber= input_file_name.split('.')
        #fileName_withExtension = imageNumber[0]+ '_' + str(counter) + '.jpg'
        #fileName = imageNumber[0]+ '_' + str(counter) + '_Grp'+str(group) +'_2way'
        fileName = imageNumber[0] + '_' + str(counter) + '_Grp' + str(group) + '_Thres-'+ str(threshold)+'_2way'
        fileName_withExtension = fileName + fileExtension
        outputFileDestination = '/home/jagan/Desktop/Rambo/prediction-in-batches/t-way_test_inputs/T-' + str(
            threshold) + '/2-way/Grp' + str(group) + '/' + fileName_withExtension
        #outputFileDestination = '/Users/Jagan/Desktop/trial/' + fileName_withExtension
        print(imageNumber[0]+ '_' + str(counter))
        cv2.imwrite(outputFileDestination,translation_applied_image)
        counter = counter+1
        #str(0.661356496233)
        writeCSV.writerow([fileName, input_file_actual_steering_value])
        #np.savetxt(output_file, (tc_info,outputFileDestinationjxc7750ERB513


    output_csv_file.close()
