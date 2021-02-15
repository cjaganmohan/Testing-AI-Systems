'''
File to create synthetic images - Individual Transformations
'''
import Automold as am
import Helpers as hp
import argparse
import csv
import cv2
import glob
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import pyplot as plt
from natsort import natsorted, ns


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error in creating the directory  ' + directory)


def modify_brightness(img, param):
    new_img = cv2.add(img, param)  # new_img = img*alpha + beta
    return new_img


def modify_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))    # mul_img = img*alpha
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


def performIndiviualTransformation(input_folder, groupNumber):
    # writing image to a folder
    fileExtension = '.jpg'
    # img = cv2.imread(input_file_name)
    # groupNumber = 8
    print (groupNumber)
    candidate_image_list = []

    #output_dir = '/home/jagan/Desktop/Chauffeur/Single_Transformation/Grp'
    output_dir = '/Users/Jagan/Desktop/Rambo/Single_Transformation/Grp'

    for file in natsorted(os.listdir(input_folder)):
        if not file.startswith(".") and file.endswith(".jpg"):
            candidate_image = file
            # print(candidate_image)
            candidate_image_list.append(candidate_image)

    # Brightness
    for x in xrange(10, 110, 10):
        # outputDirectory = './TransformedImages_Brightness_'+ str(x) + '_Group_'+groupNumber + '/center/'
        outputDirectory = output_dir + str(
            groupNumber) + '/TransformedImages_Brightness_' + str(x) + '/'
        createFolder(outputDirectory)
        for image in candidate_image_list:
            input_file = input_folder+'/'+image
            img = cv2.imread(input_file)
            transformed_image = modify_brightness(img, x)  # method call to image transformation
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
        outputDirectory = output_dir + str(groupNumber) + '/TransformedImages_Contrast_' + str(
            y1) + '/'
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


    #Blur - Averaging
    for x in xrange(3, 7, 1):
        outputDirectory = output_dir + str(
            groupNumber) + '/TransformedImages_Blur_Averaging_' + str(x) + '/'
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
        outputDirectory = output_dir + str(
            groupNumber) + '/TransformedImages_Blur_Gaussian_' + str(x) + '/'
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
        outputDirectory = output_dir + str(
            groupNumber) + '/TransformedImages_Blur_Median_' + str(x) + '/'
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
    outputDirectory = output_dir + str(
        groupNumber) + '/TransformedImages_Blur_Bilateral_filter/'
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


    #Snow
    snow_coeff_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for coeff in snow_coeff_values:
        outputDirectory = output_dir + str(groupNumber) + '/TransformedImages_Snow_' + str(coeff) + '/'
        createFolder(outputDirectory)
        for image in candidate_image_list:
            input_file = input_folder + '/' + image
            img = cv2.imread(input_file)
            transformed_image = am.add_snow(img, snow_coeff=coeff) # method call to snow transformation
            fileName_withExtension = str(image)
            outputFileDestination = outputDirectory + fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)


    #Fog
    fog_coeff_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for coeff in fog_coeff_values:
        outputDirectory = output_dir + str(
            groupNumber) + '/TransformedImages_Fog_' + str(coeff) + '/'
        createFolder(outputDirectory)
        for image in candidate_image_list:
            input_file = input_folder + '/' + image
            img = cv2.imread(input_file)
            transformed_image = am.add_fog(img, fog_coeff=coeff) # method call to fog transformation
            fileName_withExtension = str(image)
            outputFileDestination = outputDirectory + fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)


    #Rain
    rain_types= ['1', '2', '3','4', '5','6','7','8']
    for type in rain_types:
        outputDirectory = output_dir + str(
            groupNumber) + '/TransformedImages_Rain_' + str(type) + '/'
        createFolder(outputDirectory)
        for image in candidate_image_list:
            input_file = input_folder + '/' + image
            img = cv2.imread(input_file)
            if type == '1':
                transformed_image = am.add_rain(img, slant=0, rain_type='None')  # method call to rain transformation
            if type == '2':
                transformed_image = am.add_rain(img, slant=-20, rain_type='None')  # method call to rain transformation
            if type == '3':
                transformed_image = am.add_rain(img, slant=-10, rain_type='None')  # method call to rain transformation
            if type =='4':
                transformed_image = am.add_rain(img, slant=10, rain_type='None')  # method call to rain transformation
            if type =='5':
                transformed_image = am.add_rain(img, slant=20, rain_type='None')  # method call to rain transformation
            if type =='6':
                transformed_image = am.add_rain(img, slant=0, rain_type='drizzle')  # method call to rain transformation
            if type =='7':
                transformed_image = am.add_rain(img, slant=0, rain_type='heavy')  # method call to rain transformation
            if type =='8':
                transformed_image = am.add_rain(img, slant=0, rain_type='torrential')  # method call to rain transformation

            fileName_withExtension = str(image)
            outputFileDestination = outputDirectory + fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)


    #sun_flare
    flare_position= ['1', '2', '3', '4', '5', '6']
    for position in flare_position:
        outputDirectory = output_dir + str(
            groupNumber) + '/TransformedImages_Sun_Flare_' + str(position) + '/'
        createFolder(outputDirectory)
        for image in candidate_image_list:
            input_file = input_folder + '/' + image
            img = cv2.imread(input_file)
            if position == '1':
                transformed_image = am.add_sun_flare(img, flare_center=(25,25))  # method call to sun_flare transformation
            if position == '2':
                transformed_image = am.add_sun_flare(img, flare_center=(320,25))  # method call to sun_flare transformation
            if position == '3':
                transformed_image = am.add_sun_flare(img, flare_center=(600,25))  # method call to sun_flare transformation
            if position == '4':
                transformed_image = am.add_sun_flare(img, flare_center=(25,250))  # method call to sun_flare transformation
            if position == '5':
                transformed_image = am.add_sun_flare(img, flare_center=(320,250))  # method call to sun_flare transformation
            if position == '6':
                transformed_image = am.add_sun_flare(img, flare_center=(600,250))  # method call to sun_flare transformation
            fileName_withExtension = str(image)
            outputFileDestination = outputDirectory + fileName_withExtension
            print(outputFileDestination)
            cv2.imwrite(outputFileDestination, transformed_image)


    #Gravel
    outputDirectory = output_dir + str(
        groupNumber) + '/TransformedImages_gravel' + '/'
    createFolder(outputDirectory)
    for image in candidate_image_list:
        input_file = input_folder + '/' + image
        img = cv2.imread(input_file)
        transformed_image = am.add_gravel(img, rectangular_roi=(200,350,400,450)) # method call to add gravel
        fileName_withExtension = str(image)
        outputFileDestination = outputDirectory + fileName_withExtension
        print(outputFileDestination)
        cv2.imwrite(outputFileDestination, transformed_image)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path for input folder')
    parser.add_argument('--group', type=str, help='Group number')
    args, unknown = parser.parse_known_args()
    performIndiviualTransformation(args.input, args.group)