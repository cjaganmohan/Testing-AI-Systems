# reference: https://towardsdatascience.com/extract-features-visualize-filters-and-feature-maps-in-vgg16-and-vgg19-cnn-models-d2da6333edd0
import argparse
import csv
import numpy as np
import os
from matplotlib import pyplot
from natsort import natsorted
from numpy import expand_dims
from scandir import scandir
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

#from os import scandir


# load the model
model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=model.inputs, outputs=model.get_layer('block2_pool').output)
#model.summary()


def extract_features(img_src):
    #img_src = '1479425660620933516.jpg'
    img = image.load_img(img_src, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features

# Grp2_1479425660620933516.jpg
# Grp3_1479425534498905778.jpg
# Grp4_1479425619063583545.jpg
# Grp5_1479425660020827157.jpg
# Grp6_1479425535099058605.jpg
# Grp7_1479425496442340584.jpg
# Grp8_1479425537999541591.jpg
# Grp9_1479425719031130839.jpg
# Grp10_1479425712029955155.jpg
# Grp11_1479425706078866287.jpg
# Grp12_1479425527947728896.jpg
# Grp13_1479425468287290727.jpg
# Grp14_1479425470287629689.jpg
# Grp15_1479425499292775434.jpg
# Grp16_1479425488540828515.jpg
# Grp17_1479425652219428572.jpg
# Grp18_1479425654520220380.jpg
# Grp19_1479425654069742649.jpg
# Grp20_1479425653569688917.jpg


def calculate_cosine_similarities(filedir):

    # get all sub_directories
    sub_dir_list=[]
    for root, dirs, files in os.walk(filedir):  #https://pythonexamples.org/python-get-list-of-all-files-in-directory-and-sub-directories/
        for sub_dir in dirs:
            #print(os.path.join(root,sub_dir))
            sub_dir_list.append(os.path.join(root,sub_dir))
    sub_dir_list = natsorted(sub_dir_list)

    # feature extraction for original image
    original_image_features = extract_features('/Users/Jagan/Desktop/self-driving-car-project/Baseline-19-Images/Images/Grp2_1479425660620933516.jpg')

    # feature extraction for modified image &  cosine distance calculation
    for sub_dir in sub_dir_list:
        file_counter = 1
        for file in natsorted(os.listdir(sub_dir)):
            if not file.startswith(".") and file.endswith(".jpg"):
                fileName = sub_dir + '/' + file
                if file_counter == 3:
                    #print(fileName)
                    modified_image_features = extract_features(fileName)
                    print(sub_dir.rsplit('/',1)[1], distance.cosine(original_image_features.flatten(), modified_image_features.flatten()))
            file_counter = file_counter+1


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,help="path for input directory")
    args, unknown = parser.parse_known_args()
    print(args.input.rsplit('/',1)[1])
    #print(args.input)
    calculate_cosine_similarities(args.input)



