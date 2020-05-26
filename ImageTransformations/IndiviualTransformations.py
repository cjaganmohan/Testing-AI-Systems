'''
File to create Synthetic images -  Indiviual transformations


'''
import argparse
import csv
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from natsort import natsorted, ns


#img = cv2.imread('img1.png', cv2.IMREAD_COLOR) #https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_basic_image_operations_pixel_access_image_load.php

# 1479425660620933516.jpg -- [-0.9, <-0.8] - Group 2
# 1479425534498905778.jpg -- [-0.8, <-0.7] - Group 3
# 1479425619063583545.jpg -- [-0.7, <-0.6] - Group 4
# 1479425660020827157.jpg -- [-0.6, <-0.5] - Group 5
# 1479425535099058605.jpg -- [-0.5, <-0.4] - Group 6
# 1479425496442340584.jpg -- [-0.4, <-0.3] - Group 7
# 1479425537999541591.jpg -- [-0.3, <-0.2] - Group 8
# 1479425719031130839.jpg -- [-0.2, <-0.1] - Group 9
# 1479425712029955155.jpg -- [-0.1, <0] -- Group 10
# 1479425706078866287.jpg -- [0, <0.1] -- Group 11
# 1479425527947728896.jpg -- [0.1, <0.2] -- Group 12

# 1479425468287290727.jpg -- [0.2, <0.3]
# 1479425470287629689.jpg -- [0.3, <0.4]
# 1479425499292775434.jpg -- [0.4, <0.5]
# 1479425488540828515.jpg -- [0.5, <0.6]
# 1479425652219428572.jpg -- [0.6, <0.7]
# 1479425654520220380.jpg -- [0.7, <0.8]
# 1479425654069742649.jpg -- [0.8, <0.9]
# 1479425653569688917.jpg -- [0.9, <1.0]

#ImageTransformations/1479425751884901474.jpg
# input_file_name = '1479425499442845124.jpg'
# input_file_actual_steering_value = '0.480893488245'
# img = cv2.imread(input_file_name)
#
# print (img.shape)
# #print (img[440,350])
# rows = 480
# cols = 640
#
#
# def image_blur(img,blur):
#     blur_type = blur[0]
#     blur_value = int(blur[1])
#     if(blur_type == 'B1'):  #Averaging blur
#         averaging_blur_image = cv2.blur(img, (blur_value, blur_value))
#         #print('average')
#         return averaging_blur_image
#     if (blur_type == 'B2'):  #Gaussian blur
#         gaussian_blur_image = cv2.GaussianBlur(img, (blur_value, blur_value), cv2.BORDER_DEFAULT)
#         #print('gaussian')
#         return gaussian_blur_image
#     if (blur_type == 'B3'):  #Median blur
#         median_blur = cv2.medianBlur(img, blur_value)
#         #print('median')
#         return median_blur
#
#
# def image_brightness_and_contrast(img, beta, alpha):
#     adjusted_brightness_and_contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
#     return adjusted_brightness_and_contrast
#
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
#
# def image_scale(img, scale_percent):
#     #print(img.shape, scale_percent)
#     width = int(img.shape[1] * scale_percent / 100)  # img.shape --> height * width
#     height = int(img.shape[0] * scale_percent / 100)
#     scaled_image_dimension = (width, height)
#
#     # resizing the image
#     resized_image = cv2.resize(img, scaled_image_dimension, interpolation=cv2.INTER_AREA)
#     #print(type(resized_image))
#     return resized_image
#
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
#     if (shear_type == 'VS'):
#         #M2 = np.float32([[1, shear, 0], [0, 1, 0]])
#         M3 = np.float32([[1, 0, 0], [shear, 1, 0]])
#         #print(shear)
#         v_shear_image = cv2.warpAffine(img, M3, (
#             width_original_image, height_original_image))  # 3rd argument -- size of the output image...
#         return v_shear_image
#
#
# def image_translation(img, translation):
#     height_original_image = int(img.shape[0])
#     width_original_image = int(img.shape[1])
#     M1 = np.float32([[1, 0, translation], [0, 1, translation]])
#     translated_image = cv2.warpAffine(img, M1, (
#     width_original_image, height_original_image))  # 3rd argument -- size of the output image...
#     return translated_image
#
# input_test_file = 'Udacity-Self-Driving-2-way-revised-output-with-two-constraints.csv'
# output_file = 'final_evaluation.csv'

# with open(input_test_file) as input_csv_file, open(output_file, 'wb') as output_csv_file:
# #with open(input_test_file) as input_csv_file:
#     readCSV = csv.reader(input_csv_file, delimiter=',')
#     writeCSV = csv.writer(output_csv_file, delimiter=',' , quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     writeCSV.writerow(['frame_id', 'steering_angle'])  # header information for the output.csv file
#     path = './outputImages'
#     counter = 1
#     next(readCSV) # skip the header in input CSV file
#     for row in readCSV:
#         #print(row)
#         #print(row[0])
#         tc_info = 'Blur  '+ row[0],'Brightness   '+ row[1],'Contrast   '+row[2], 'Rotation  '+ row[3],'Scaling   '+ row[4],'Shearing   '+row[5],'Translation  '+ row[6]
#         #print(tc_info)
#         blur = str(row[0]).split('-')  # [Blur_type, Blur_value]
#         beta =  int(row [1]) # Brightness control (0-100)
#         alpha = float(row [2]) # Contrast control (1.0-3.0)
#         rotation_angle = int(row [3])
#         scaling_percent_size = int(row[4])
#         shearing_k_value = str(row[5]).split(':')  # [shearing_type, shearing_value]
#         #print(shearing_k_value)
#         translation = int(row[6])
#
#         blur_applied_image = image_blur(img,blur)
#         brightness_and_contrast_adjusted_image = image_brightness_and_contrast(blur_applied_image, beta, alpha)
#         rotation_applied_image = image_rotation(brightness_and_contrast_adjusted_image, rotation_angle)
#         scaling_applied_image = image_scale(rotation_applied_image, scaling_percent_size)
#         shearing_applied_image = image_shear (scaling_applied_image,shearing_k_value)
#         translation_applied_image = image_translation (shearing_applied_image, translation)
#
#
#         # cv2.imshow('img', translation_applied_image)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#
#         #writing image to a folder
#         fileExtension='.jpg'
#         imageNumber= input_file_name.split('.')
#         #fileName_withExtension = imageNumber[0]+ '_' + str(counter) + '.jpg'
#         fileName = imageNumber[0]+ '_' + str(counter)
#         fileName_withExtension = fileName + fileExtension
#         outputFileDestination = './outputImages/'+fileName_withExtension
#         print(imageNumber[0]+ '_' + str(counter))
#         cv2.imwrite(outputFileDestination,translation_applied_image)
#         counter = counter+1
#         #str(0.661356496233)
#         writeCSV.writerow([fileName, input_file_actual_steering_value])
#         #np.savetxt(output_file, (tc_info,outputFileDestination),delimiter=',')
#
#     output_csv_file.close()

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error in creating the directory  ' + directory)

def modify_brightness(img, param):
    new_img = cv2.add(img, param)  # new_img = img*alpha + beta
    return new_img

def modify_rotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
    new_img = cv2.warpAffine(img, M, (cols, rows))
    return new_img

def modify_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))    # mul_img = img*alpha
    return new_img

def modify_shear(img, params):
    rows, cols, ch = img.shape
    #factor = params*(-1.0) #Bug -- identified on 05/25/2020
    factor = params # Bug fix
    print(factor)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    new_img = cv2.warpAffine(img, M, (cols, rows))
    return new_img

def modify_translation(img, params):
    rows, cols, ch = img.shape
    #M1 = np.float32([[1, 0, translation], [0, 1, translation]])
    M = np.float32([[1, 0, params], [0, 1, params]])
    new_img = cv2.warpAffine(img, M, (cols, rows))
    return new_img

def modify_scale(img, params):
    new_img = cv2.resize(img, None, fx=params, fy=params, interpolation=cv2.INTER_CUBIC)
    return new_img

def modify_blur(img, params):
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.blur(img, (6, 6))
    if params == 5:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 7:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 8:
        blur = cv2.medianBlur(img, 3)
    if params == 9:
        blur = cv2.medianBlur(img, 5)
    if params == 10:
        blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def performIndiviualTransformation(input_folder):
    # candidate_image_list1 = [1479425660620933516, 1479425534498905778,
    #                         1479425619063583545, 1479425660020827157,
    #                         1479425535099058605, 1479425496442340584,
    #                         1479425537999541591, 1479425719031130839,
    #                         1479425712029955155, 1479425706078866287,
    #                         1479425527947728896, 1479425468287290727,
    #                         1479425470287629689, 1479425499292775434,
    #                         1479425488540828515, 1479425652219428572,
    #                         1479425654520220380, 1479425654069742649,
    #                         1479425653569688917]

    # writing image to a folder
    fileExtension = '.jpg'
    #img = cv2.imread(input_file_name)

    candidate_image_list=[]

    for file in natsorted(os.listdir(input_folder)):
        if not file.startswith(".") and file.endswith(".jpg"):
            candidate_image = file
            #print(candidate_image)
            candidate_image_list.append(candidate_image)



    #Brightness
    for x in xrange(10,110,10):
        #outputDirectory = './TransformedImages_Brightness_'+ str(x) + '_Group_'+groupNumber + '/center/'
        outputDirectory = '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations/TransformedImages_Brightness_' + str(x)+'/'
        createFolder(outputDirectory)
        for image in candidate_image_list:
            input_file = input_folder+'/'+image
            img = cv2.imread(input_file)
            transformed_image = modify_brightness(img, x)  # method call to image transformation
            fileName_withExtension = str(image)
            outputFileDestination =  outputDirectory+ fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)

    #Rotation
    for x in xrange(3,33,3):
        #outputDirectory = './TransformedImages_Rotation_'+ str(x) + '_Group_'+groupNumber + '/center/'
        outputDirectory = '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations/TransformedImages_Rotation_' + str(x)+'/'
        createFolder(outputDirectory)
        for image in candidate_image_list:
            input_file = input_folder + '/' + image
            img = cv2.imread(input_file)
            transformed_image = modify_rotation(img, x)  # method call to image transformation
            fileName_withExtension = str(image)
            outputFileDestination =  outputDirectory+ fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)

    #Contrast
    y1 = 1.2
    for x in xrange(12, 32, 2):   #1.2, 3.0, 0.2
        #y = 1.2
        print(y1)
        #outputDirectory = './TransformedImages_Contrast_'+ str(y1) + '_Group_'+groupNumber + '/center/'
        outputDirectory = '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations/TransformedImages_Contrast_' + str(y1)+'/'
        createFolder(outputDirectory)
        for image in candidate_image_list:
            input_file = input_folder + '/' + image
            img = cv2.imread(input_file)
            transformed_image = modify_contrast(img, y1)  # method call to image transformation
            fileName_withExtension = str(image)
            outputFileDestination = outputDirectory + fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)
        y1 = y1 + 0.2

    # Shear
    y2 = -1.0
    for x in xrange(-10, 0, 1):  #-1.0, -0.1, 0.1
        #y = -1.0
        print(y2)
        #outputDirectory = './TransformedImages_Shear_'+ str(y2) + '_Group_'+groupNumber + '/center/'
        outputDirectory = '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations/TransformedImages_Shear_' + str(y2)+'/'
        createFolder(outputDirectory)
        for image in candidate_image_list:
            input_file = input_folder + '/' + image
            img = cv2.imread(input_file)
            transformed_image = modify_shear(img, y2)  # method call to image transformation
            fileName_withExtension = str(image)
            outputFileDestination = outputDirectory + fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)
        y2 = y2 + 0.1

    # Translation
    for x in xrange(10, 110, 10):
        #outputDirectory = './TransformedImages_Translation_'+ str(x) + '_Group_'+groupNumber + '/center/'
        outputDirectory = '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations/TransformedImages_Translation_' + str(x)+'/'
        createFolder(outputDirectory)
        for image in candidate_image_list:
            input_file = input_folder + '/' + image
            img = cv2.imread(input_file)
            transformed_image = modify_translation(img, x)  # method call to image transformation
            fileName_withExtension = str(image)
            outputFileDestination = outputDirectory + fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)

    #Scale
    y3 = 1.5
    for x in xrange(15, 65, 5):  #1.5, 6, 0.5
        #y = 1.5
        print(y3)
        #outputDirectory = './TransformedImages_Scale_'+ groupNumber + str(x) + '_Group_'+groupNumber + '/center/'
        outputDirectory = '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations/TransformedImages_Scale_' + str(y3)+'/'
        createFolder(outputDirectory)

        for image in candidate_image_list:
            input_file = input_folder + '/' + image
            img = cv2.imread(input_file)
            transformed_image = modify_scale(img, y3)  # method call to image transformation
            fileName_withExtension = str(image)
            outputFileDestination = outputDirectory + fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)
        y3 = y3 + 0.5

    #Blur - Averaging
    for x in xrange(3, 7, 1):
        outputDirectory = '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations/TransformedImages_Blur_Averaging_' + str(x)+'/'
        createFolder(outputDirectory)
        params_value = 1
        for image in candidate_image_list:
            input_file = input_folder + '/' + image
            img = cv2.imread(input_file)
            transformed_image = modify_blur(img, params_value)  # method call to image transformation
            fileName_withExtension = str(image)
            outputFileDestination = outputDirectory + fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)
        params_value = params_value + 1

    #Blur - Gaussian
    for x in xrange(3, 9, 2):
        outputDirectory = '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations/TransformedImages_Blur_Gaussian_' + str(x)+'/'
        createFolder(outputDirectory)
        params_value = 5
        for image in candidate_image_list:
            input_file = input_folder + '/' + image
            img = cv2.imread(input_file)
            transformed_image = modify_blur(img, params_value)  # method call to image transformation
            fileName_withExtension = str(image)
            outputFileDestination = outputDirectory + fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)
        params_value = params_value + 1

    #Blur - Median
    for x in xrange(3, 7, 2):
        outputDirectory = '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations/TransformedImages_Blur_Median_' + str(x)+'/'
        createFolder(outputDirectory)
        params_value = 8
        for image in candidate_image_list:
            input_file = input_folder + '/' + image
            img = cv2.imread(input_file)
            transformed_image = modify_blur(img, params_value)  # method call to image transformation
            fileName_withExtension = str(image)
            outputFileDestination = outputDirectory + fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)
        params_value = params_value + 1

    #Blur - Bilateral filter
    outputDirectory = '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations/TransformedImages_Blur_Bilateral_filter/'
    createFolder(outputDirectory)
    params_value = 9
    for image in candidate_image_list:
        input_file = input_folder + '/' + image
        img = cv2.imread(input_file)
        transformed_image = modify_blur(img, params_value)  # method call to image transformation
        fileName_withExtension = str(image)
        outputFileDestination = outputDirectory + fileName_withExtension
        print(outputFileDestination)
        cv2.imwrite(outputFileDestination, transformed_image)
    #params_value = params_value + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type = str, help ='Path for input folder')
    #parser.add_argument('--group', type = str, help = 'Group number')
    args, unknown = parser.parse_known_args()
    performIndiviualTransformation(args.input)