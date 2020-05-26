#import rospy
#from steering_node import SteeringNode
#modified by Jagan

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


# from rmse import calc_rmse
# from generator import gen


class AutumnModel(object):
    def __init__(self, cnn_graph_path, lstm_json_path, cnn_weights_path, lstm_weights_path):
        sess = tf.InteractiveSession()
        saver = tf.train.import_meta_graph(cnn_graph_path)
        saver.restore(sess, cnn_weights_path) #modified by Jagan
        self.cnn = tf.get_default_graph()

        self.fc3 = self.cnn.get_tensor_by_name("fc3/mul:0")
        self.y = self.cnn.get_tensor_by_name("y:0")
        self.x = self.cnn.get_tensor_by_name("x:0")
        self.keep_prob = self.cnn.get_tensor_by_name("keep_prob:0")

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
        img = self.process(img)
        image = scipy.misc.imresize(img[-400:], [66, 200]) / 255.0
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
        #print(predicted_steering)
        #print(steering)

        mse += (float(steering) - float(predicted_steering))**2.
        print("Observed Steering Angle : " + str(steering) + " Predicted Steering Angle: " + str(
            predicted_steering) + " Mean square error: " + str(
            mse))  # Jagan
    return (mse/count) ** 0.5

def autumn_reproduce(dataset_path):
    # seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    # seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    # seed_inputs2 = os.path.join(dataset_path, "Ch2_001/center/")
    # seed_labels2 = os.path.join(dataset_path, "Ch2_001/CH2_final_evaluation.csv")

    seed_inputs1 = os.path.join(dataset_path, "testData/")
    seed_labels1 = os.path.join(dataset_path, "testData/test_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "center/")
    seed_labels2 = os.path.join(dataset_path, "final_evaluation.csv")
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

    #filename = 'Autumn-model-group_' + group_num + '.csv'
    filename = 'Autumn-baseline-tf-1.12.csv'
    # print(filename)
    with open(filename, 'ab', 0) as csvfile:
        for f in filelist1:
            seed_image = cv2.imread(os.path.join(seed_inputs1, f))
            yhat = model(seed_image)
            yhats.append(yhat)
            labels.append(truth[f])
            if count % 500 == 0:
                print ("processed images: " + str(count) + " total: " + str(total))
            count = count + 1

        for f in filelist2:
            seed_image = cv2.imread(os.path.join(seed_inputs2, f))
            yhat = model(seed_image)
            yhats.append(yhat)
            labels.append(truth[f])
            print(" f-value: " + f + " truth-value: " + truth[f] + " yhat-value: " + str(yhat)[1:-1])
            # if count % 500 == 0:
            #     print("processed images: " + str(count) + " total: " + str(total))
            # count = count + 1
            #writer.writerow([f, truth[f], str(yhat)[1:-1]])

    mse = calc_rmse(yhats, labels)
    print("mse: " + str(mse))

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Model Runner for team autumn')
    # parser.add_argument('bagfile', type=str, help='Path to ROS bag')
    # parser.add_argument('--input', '-i', action='store', dest='input_file',
    #                     default='example-final.csv', help='Input model csv file name')
    # parser.add_argument('--output', '-o', action='store', dest='output_file',
    #                     default='output-lstm.csv', help='Output csv file name')
    # parser.add_argument('--data-dir', '--data', action='store', dest='data_dir')
    # parser.add_argument('--cnn-graph', '--cnn-meta', action='store', dest='cnn_graph',
    #                     default='autumn/autumn-cnn-model-tf.meta')
    # parser.add_argument('--lstm-json', '--lstm-meta', action='store', dest='lstm_json',
    #                     default='autumn/autumn-lstm-model-keras.json')
    # parser.add_argument('--cnn-weights', action='store', dest='cnn_weights',
    #                     default='autumn/autumn-cnn-weights.ckpt')
    # parser.add_argument('--lstm-weights', action='store', dest='lstm_weights',
    #                     default='autumn/autumn-lstm-weights.hdf5')
    # #args = parser.parse_args()  # commented by Jagan
    # args, unknown = parser.parse_known_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help = 'path for test dataset')
    args = parser.parse_args()
    autumn_reproduce(args.dataset)

    # def make_predictor():
    #     model = AutumnModel(args.cnn_graph, args.lstm_json, args.cnn_weights, args.lstm_weights)
    #     return lambda img: model.predict(img)
    #
    # def process(predictor, img):
    #     return predictor(img)
    #
    # model = make_predictor()
    #
    # print calc_rmse(lambda image_pred: model(image_pred),
    #                gen(args.bagfile))
