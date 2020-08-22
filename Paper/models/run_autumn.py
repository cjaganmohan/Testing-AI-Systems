# import rospy
# from steering_node import SteeringNode
# modified by Jagan

import argparse
import csv
import cv2
import scipy.misc
import tensorflow as tf
import tensorflow as tf
from collections import deque
from keras import backend as K
from keras import backend as K
from keras.layers import *
from keras.layers import *
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.models import *
from math import pi
import pdb
import sys
import os
import shutil
import tensorflow
import keras

reload(sys)
sys.setdefaultencoding('ISO-8859-1')


# from rmse import calc_rmse
# from generator import gen


class AutumnModel(object):
    def __init__(self, cnn_graph_path, lstm_json_path, cnn_weights_path, lstm_weights_path):
        sess = tf.InteractiveSession()
        saver = tf.train.import_meta_graph(cnn_graph_path)
        saver.restore(sess, cnn_weights_path)  # modified by Jagan
        self.cnn = tf.get_default_graph()

        self.fc3 = self.cnn.get_tensor_by_name("fc3/mul:0")
        self.y = self.cnn.get_tensor_by_name("y:0")
        self.x = self.cnn.get_tensor_by_name("x:0")
        self.keep_prob = self.cnn.get_tensor_by_name("keep_prob:0")

        # Bug - Inconsistent prediction result -- same image, but different prediction on multiple execution
        # Root cause -- appears to be the weights loaded to the model
        # Fix -- Run the model with Keras 1.1.0 AND Tensorflow - 0.11.0rc1
        with open(lstm_json_path, 'r') as f:
            json_string = f.read()
        self.model = model_from_json(json_string)
        self.model.load_weights(lstm_weights_path)

        self.prev_image = None
        self.last = []
        self.steps = []

    def process(self, img):
        prev_image = self.prev_image if self.prev_image is not None else img
        self.prev_image = img
        prev = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
        next = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        self.last.append(flow)

        if len(self.last) > 4:
            self.last.pop(0)

        weights = [1, 1, 2, 2]
        last = list(self.last)
        for x in range(len(last)):
            last[x] = last[x] * weights[x]

        avg_flow = sum(last) / sum(weights)
        mag, ang = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])

        hsv = np.zeros_like(prev_image)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    def predict(self, img):
        # pdb.set_trace()
        img = self.process(img)
        image = scipy.misc.imresize(img[-400:], [66, 200]) / 255.0
        # pdb.set_trace()
        cnn_output = self.fc3.eval(feed_dict={self.x: [image], self.keep_prob: 1.0})
        self.steps.append(cnn_output)
        if len(self.steps) > 100:
            self.steps.pop(0)
        output = self.y.eval(feed_dict={self.x: [image], self.keep_prob: 1.0})
        return output[0][0]


def calc_rmse(yhat, label):
    mse = 0.
    count = 0
    if len(yhat) != len(label):
        print ("yhat and label have different lengths")
        return -1
    for i in xrange(len(yhat)):
        count += 1
        predicted_steering = yhat[i]
        steering = label[i]
        mse += (float(steering) - float(predicted_steering)) ** 2.

    return (mse / count) ** 0.5


def autumn_reproduce(dataset_path, group_number, file_type, output_path):
    # load the weights
    cnn_graph_path = "./autumn-cnn-model-tf.meta"
    cnn_weights_path = "./autumn-cnn-weights.ckpt"
    lstm_json_path = "./autumn-lstm-model-keras.json"
    lstm_weights_path = "./autumn-lstm-weights.hdf5"

    def make_predictor():
        model = AutumnModel(cnn_graph_path, lstm_json_path, cnn_weights_path, lstm_weights_path)
        return lambda img: model.predict(img)

    def process(predictor, img):
        return predictor(img)

    model = make_predictor()

    # sort the images
    images = []
    for root, sub_dirs, files in os.walk(dataset_path):
        for f in files:
            if '.jpg' in f or '.png' in f:
                images.append((root, f))
    images.sort(key=lambda x: x[1])

    # output_file information
    csv_filename = 'Autumn-' + str(file_type) + '_Group' + str(group_number) + '.csv'
    txt_filename = 'Autumn-' + str(file_type) + '_Group' + str(group_number) + '.txt'

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
            print image[1] + ',', prediction
            writer.writerow([image[1], prediction])  # image_name, prediction_value
        print '---------------'
        print'Environment information: ', '  Backend - Tensorflow: ', tf.__version__, '  Keras: ', keras.__version__
        print'Completed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--group', type=str)
    parser.add_argument('--file_type', type=str)  # {'Baseline', 'Individual_transformation','2-way'}
    parser.add_argument('--output_path', type=str)

    args, unknown = parser.parse_known_args()
    print'Calling the Autumn model now ----- '
    print args.dataset, args.group, args.file_type, args.output_path
    autumn_reproduce(args.dataset, args.group, args.file_type, args.output_path)
