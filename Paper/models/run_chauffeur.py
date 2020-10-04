"""
This is an example script for reproducing chauffeur model in predicting hmb3 dataset
and udacity autonomous car challenge2 test dataset.  --- MODIFIED version to execute a specific scenario -- Apply transformation
only to the SUBJECT image and use the original version for the remaining 99 images
"""
from __future__ import print_function

import argparse
import csv
import cv2
import numpy as np
import os
# from importlib import reload
import pdb
import shutil
import sys
from collections import deque
# import rospy
from keras import backend as K
from keras.models import model_from_json
from natsort import natsorted, ns
import tensorflow as tf
import keras

reload(sys)


# sys.setdefaultencoding('utf8')
# keras 1.2.2 tf:1.2.0
class ChauffeurModel(object):
    def __init__(self,
                 cnn_json_path,
                 cnn_weights_path,
                 lstm_json_path,
                 lstm_weights_path):

        self.cnn = self.load_from_json(cnn_json_path, cnn_weights_path)
        self.encoder = self.load_encoder(cnn_json_path, cnn_weights_path)
        self.lstm = self.load_from_json(lstm_json_path, lstm_weights_path)

        self.scale = 16.
        self.timesteps = 100

        self.threshold_cnn = 0.1
        self.threshold_lstm = 0.4
        self.timestepped_x = np.empty((1, self.timesteps, 8960))

    def load_encoder(self, cnn_json_path, cnn_weights_path):
        model = self.load_from_json(cnn_json_path, cnn_weights_path)
        model.load_weights(cnn_weights_path)

        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        return model

    def load_from_json(self, json_path, weights_path):
        model = model_from_json(open(json_path, 'r').read())
        model.load_weights(weights_path)
        return model

    def make_cnn_only_predictor(self):
        def predict_fn(img):
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img[120:240, :, :]
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img = ((img - (255.0 / 2)) / 255.0)
            return self.cnn.predict_on_batch(img.reshape((1, 120, 320, 3)))[0, 0] / self.scale

        return predict_fn

    def make_stateful_predictor(self):
        steps = deque()

        def predict_fn(img):
            # preprocess image to be YUV 320x120 and equalize Y histogram
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img[120:240, :, :]
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img = ((img - (255.0 / 2)) / 255.0)
            # apply feature extractor
            img = self.encoder.predict_on_batch(img.reshape((1, 120, 320, 3)))

            # initial fill of timesteps
            if not len(steps):
                for _ in xrange(self.timesteps):
                    steps.append(img)

            # put most recent features at end
            steps.popleft()
            steps.append(img)

            timestepped_x = np.empty((1, self.timesteps, img.shape[1]))
            for i, img in enumerate(steps):
                timestepped_x[0, i] = img
            return self.lstm.predict_on_batch(timestepped_x)[0, 0] / self.scale

        return predict_fn


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


def chauffeur_reproduce(dataset_path, group_number, file_type, output_path):
    cnn_json_path = "./cnn.json"
    cnn_weights_path = "./cnn.weights"
    lstm_json_path = "./lstm.json"
    lstm_weights_path = "./lstm.weights"

    def make_predictor():
        K.set_learning_phase(0)
        model = ChauffeurModel(
            cnn_json_path,
            cnn_weights_path,
            lstm_json_path,
            lstm_weights_path)
        return model.make_stateful_predictor()

    model = make_predictor()

    # sort the images
    images = []
    for root, sub_dirs, files in os.walk(dataset_path):
        for f in files:
            if '.jpg' in f or '.png' in f:
                images.append((root, f))
    images.sort(key=lambda x: x[1])

    # output_file information
    csv_filename = 'Chauffeur-' + str(file_type) + '_Group' + str(group_number) + '.csv'
    txt_filename = 'Chauffeur-' + str(file_type) + '_Group' + str(group_number) + '.txt'

    save_console_output = str(output_path) + '/' + txt_filename
    sys.stdout = open(save_console_output, 'w')
    output_as_csv = str(output_path) + '/' + csv_filename

    # load, predict and save the results
    with open(output_as_csv, 'ab', 0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['File_name', 'Predicted_steering_angle'])
        for image in images:
            test_input = cv2.imread(os.path.join(image[0], image[1]))
            prediction = model(test_input)
            print(image[1] + ',', prediction)
            writer.writerow([image[1], prediction])  # image_name, prediction_value
        print('---------------')
        print('Environment information: ', '  Tensorflow: ', tf.__version__, 'Keras: ', keras.__version__)
        print('Completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--group', type=str)
    parser.add_argument('--file_type', type=str)  # {'Baseline', 'Individual_transformation','2-way'}
    parser.add_argument('--output_path', type=str)

    args, unknown = parser.parse_known_args()
    print('Calling the Chauffeur model now ----- ')
    print(args.dataset, args.group, args.file_type, args.output_path)
    chauffeur_reproduce(args.dataset, args.group, args.file_type, args.output_path)
