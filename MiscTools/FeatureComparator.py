''' Python file to load and compare features extracted by a Convolutional neural network
'''
# Reference # 1 - to save and reload numpy - https://pythonexamples.org/python-numpy-save-and-load-array-from-file/


import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from natsort import natsorted
from scipy.spatial import distance


def load_numpyFile(file):
    # file = open(file, "rb")
    # numpy_array = np.load(file)
    # print(numpy_array)
    original_image_npArray = np.load(open('/Users/Jagan/Desktop/Rambo_feature_extraction_trials/Original', "rb"))
    shear_1_npArray = np.load(open('/Users/Jagan/Desktop/Rambo_feature_extraction_trials/shear-0.1',"rb"))
    shear_2_npArray = np.load(open('/Users/Jagan/Desktop/Rambo_feature_extraction_trials/shear-0.2',"rb"))
    shear_3_npArray = np.load(open('/Users/Jagan/Desktop/Rambo_feature_extraction_trials/shear-0.3',"rb"))
    shear_4_npArray = np.load(open('/Users/Jagan/Desktop/Rambo_feature_extraction_trials/shear-0.4',"rb"))


    print(type(original_image_npArray))
    print(original_image_npArray.ndim)
    print(original_image_npArray.flatten().ndim)
    print(original_image_npArray.flatten())

    print ('------------- Eucledian distance of flatten arrays --------------')
    print(distance.euclidean(original_image_npArray.flatten(),original_image_npArray.flatten()))
    distance_between_Original_and_Shear_1 = distance.euclidean(original_image_npArray.flatten(), shear_1_npArray.flatten())
    distance_between_Original_and_Shear_2 = distance.euclidean(original_image_npArray.flatten(), shear_2_npArray.flatten())
    distance_between_Original_and_Shear_3 = distance.euclidean(original_image_npArray.flatten(), shear_3_npArray.flatten())
    distance_between_Original_and_Shear_4 = distance.euclidean(original_image_npArray.flatten(), shear_4_npArray.flatten())


    print'Euclidean distance between Original and Shear_0.1----->', distance_between_Original_and_Shear_1
    print'Euclidean distance between Original and Shear_0.2----->', distance_between_Original_and_Shear_2
    print'Euclidean distance between Original and Shear_0.3----->', distance_between_Original_and_Shear_3
    print'Euclidean distance between Original and Shear_0.4----->', distance_between_Original_and_Shear_4

    print('Euclidean distance between Shear_0.1 and Shear_0.1----->',
          distance.euclidean(shear_1_npArray.flatten(), shear_1_npArray.flatten()))
    print('Euclidean distance between Shear_0.1 and Shear_0.2----->',
          distance.euclidean(shear_1_npArray.flatten(), shear_2_npArray.flatten()))
    print('Euclidean distance between Shear_0.1 and Shear_0.3----->',
          distance.euclidean(shear_1_npArray.flatten(), shear_3_npArray.flatten()))
    print('Euclidean distance between Shear_0.1 and Shear_0.4----->',
          distance.euclidean(shear_1_npArray.flatten(), shear_4_npArray.flatten()))



    print('---------------------------')

    print('Euclidean distance between Shear_0.2 and Shear_0.3----->',
          distance.euclidean(shear_2_npArray.flatten(), shear_3_npArray.flatten()))
    print('Euclidean distance between Shear_0.3 and Shear_0.4----->',
          distance.euclidean(shear_3_npArray.flatten(), shear_4_npArray.flatten()))




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    load_numpyFile('/Users/Jagan/Desktop/Rambo_feature_extraction_trials/shear-0.3')