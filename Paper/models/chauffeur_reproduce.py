"""
This is an example script for reproducing chauffeur model in predicting hmb3 dataset
and udacity autonomous car challenge2 test dataset.
"""
from __future__ import print_function

import argparse
import csv
import cv2
import numpy as np
import os
# from importlib import reload
import pdb
import sys
from collections import deque
# import rospy
from keras import backend as K
from keras.models import model_from_json

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
        # pdb.set_trace()
        return model

    def load_from_json(self, json_path, weights_path):
        model = model_from_json(open(json_path, 'r').read())
        model.load_weights(weights_path)
        # pdb.set_trace()
        return model

    def make_cnn_only_predictor(self):
        def predict_fn(img):
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img[120:240, :, :]
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img = ((img - (255.0 / 2)) / 255.0)
            # pdb.set_trace()
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
            # pdb.set_trace()
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
        # print(predicted_steering)
        # print(steering)
        mse += (float(steering) - float(predicted_steering)) ** 2.
        # print("Observed Steering Angle : " + str(steering) + " Predicted Steering Angle: " + str(
        #     predicted_steering) + " Mean square error: " + str(
        #     mse))  # Jagan
    return (mse / count) ** 0.5


def chauffeur_reproduce(dataset_path):
    # seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    # seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    # seed_inputs2 = os.path.join(dataset_path, "Ch2_001/center/")
    # seed_labels2 = os.path.join(dataset_path, "Ch2_001/CH2_final_evaluation.csv")

    seed_inputs1 = os.path.join(dataset_path, "testData/")
    seed_labels1 = os.path.join(dataset_path, "testData/test_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "center/")
    seed_labels2 = os.path.join(dataset_path, "final_evaluation.csv")
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
        # pdb.set_trace() # added by Jagan
        return model.make_stateful_predictor()

    model = make_predictor()
    # print(model.summary())  # added by Jagan
    print("Prediction results from Chauffer-model")  # Jagan
    filelist1 = []
    for image_file in sorted(os.listdir(seed_inputs1)):
        if image_file.endswith(".jpg"):
            filelist1.append(image_file)
        truth = {}
    with open(seed_labels1, 'rb') as csvfile1:
        label1 = list(csv.reader(csvfile1, delimiter=',', quotechar='|'))
    label1 = label1[1:]
    for i in label1:
        truth[i[0] + ".jpg"] = i[1]

    filelist2 = []
    for image_file in sorted(os.listdir(seed_inputs2)):
        if image_file.endswith(".jpg"):
            filelist2.append(image_file)
    with open(seed_labels2, 'rb') as csvfile2:
        label2 = list(csv.reader(csvfile2, delimiter=',', quotechar='|'))
    label2 = label2[1:]

    for i in label2:
        truth[i[0] + ".jpg"] = i[1]

    yhats = []
    labels = []
    count = 0
    total = len(filelist1) + len(filelist2)

    # filename = 'Rambo-model-group_' + group_num + '.csv'
    filename = 'Chauffeur-consolidated.csv'
    # print(filename)
    with open(filename, 'ab', 0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['File_name', 'Observed_steering_angle(Ground_truth)',
                         'Predicted_steering_angle'])
        for f in filelist1:
            seed_image = cv2.imread(os.path.join(seed_inputs1, f))
            # pdb.set_trace()
            yhat = model(seed_image)
            yhats.append(yhat)
            labels.append(truth[f])
            if count % 500 == 0:
                print("processed images: " + str(count) + " total: " + str(total))
            count = count + 1

        for f in filelist2:
            seed_image = cv2.imread(os.path.join(seed_inputs2, f))
            # pdb.set_trace()
            yhat = model(seed_image)
            yhats.append(yhat)
            labels.append(truth[f])
            print("filename: " + f + " truth_value: " + truth[f] + " yhat_value: " + str(yhat))  # Jagan
            # if count % 500 == 0:
            # print ("processed images: " + str(count) + " total: " + str(total))
            count = count + 1
            writer.writerow([f, truth[f], str(yhat)])
        mse = calc_rmse(yhats, labels)

    print("rmse: " + str(mse))
    print('Print via VSCode- SSH')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    chauffeur_reproduce(args.dataset)
