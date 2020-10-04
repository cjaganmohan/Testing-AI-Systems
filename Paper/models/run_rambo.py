"""
This is an example script for reproducing rambo model in predicting hmb3 dataset
and udacity autonomous car challenge2 test dataset.
"""
from __future__ import print_function

import argparse
import csv
import cv2
import numpy as np
import os
import sys
import shutil
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity
from natsort import natsorted, ns
import theano
import keras

reload(sys)
sys.setdefaultencoding('ISO-8859-1')


# Environment requirements: Run this model with Theano - 0.9.0 as backend

class Model(object):
    def __init__(self,
                 model_path,
                 X_train_mean_path):

        self.model = load_model(model_path)
        self.model.compile(optimizer="adam", loss="mse")
        self.X_mean = np.load(X_train_mean_path)
        self.mean_angle = np.array([-0.004179079])
        print(self.mean_angle)
        self.img0 = None
        self.state = deque(maxlen=2)

    def predict(self, img_path):
        # img_path = 'test.jpg'
        # misc.imsave(img_path, img)
        img1 = load_img(img_path, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)

        if self.img0 is None:
            self.img0 = img1
            return self.mean_angle

        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)  # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            return self.mean_angle

        else:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)  # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            X = np.concatenate(self.state, axis=-1)
            X = X[:, :, ::-1]
            X = np.expand_dims(X, axis=0)
            X = X.astype('float32')
            X -= self.X_mean
            X /= 255.0
            return self.model.predict(X)[0]


def calc_rmse(yhat, label):
    mse = 0.
    count = 0
    if len(yhat) != len(label):
        print("yhat and label have different lengths")
        return -1
    for i in xrange(len(yhat)):
        count += 1
        predicted_steering = yhat[i]
        steering = label[i]
        mse += (float(steering) - float(predicted_steering)) ** 2.
    return (mse / count) ** 0.5


def rambo_reproduce(dataset_path, group_number, file_type, output_path):
    # Load model weights
    model = Model("./final_model.hdf5", "./X_train_mean.npy")
    # yhat = model.predict(os.path.join(seed_inputs2, f))

    # sort the images
    images = []
    for root, sub_dirs, files in os.walk(dataset_path):
        for f in files:
            if '.jpg' in f or '.png' in f:
                images.append((root, f))
    images.sort(key=lambda x: x[1])

    # output_file information
    csv_filename = 'Rambo-' + str(file_type) + '_Group' + str(group_number) + '.csv'
    txt_filename = 'Rambo-' + str(file_type) + '_Group' + str(group_number) + '.txt'

    save_console_output = str(output_path) + '/' + txt_filename
    sys.stdout = open(save_console_output, 'w')
    output_as_csv = str(output_path) + '/' + csv_filename

    # load, predict and save the results
    with open(output_as_csv, 'ab', 0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['File_name', 'Predicted_steering_angle'])
        for image in images:
            # test_input = cv2.imread(os.path.join(image[0], image[1]))
            prediction = model.predict(os.path.join(image[0], image[1]))
            print(image[1] + ',', str(prediction)[1:-1])
            writer.writerow([image[1], str(prediction)[1:-1]])  # image_name, prediction_value
        print('---------------')
        print('Environment information: ', '  Backend - Theano: ', theano.__version__, '  Keras: ', keras.__version__)
        print('Completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--group', type=str)
    parser.add_argument('--file_type', type=str)  # {'Baseline', 'Individual_transformation','2-way'}
    parser.add_argument('--output_path', type=str)

    args, unknown = parser.parse_known_args()
    print('Calling the Rambo model now ----- ')
    print(args.dataset, args.group, args.file_type, args.output_path)
    rambo_reproduce(args.dataset, args.group, args.file_type, args.output_path)
