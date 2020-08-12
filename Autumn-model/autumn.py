import argparse
import cv2
import scipy.misc
import tensorflow as tf
from keras.layers import *
from keras.models import *


class AutumnModel(object):
    def __init__(self, cnn_graph, lstm_json, cnn_weights, lstm_weights):
        sess = tf.InteractiveSession()
        saver = tf.train.import_meta_graph(cnn_graph)
        saver.restore(sess, cnn_weights)
        self.cnn = tf.get_default_graph()

        self.fc3 = self.cnn.get_tensor_by_name("fc3/mul:0")
        self.y = self.cnn.get_tensor_by_name("y:0")
        self.x = self.cnn.get_tensor_by_name("x:0")
        self.keep_prob = self.cnn.get_tensor_by_name("keep_prob:0")

        with open(lstm_json, 'r') as f:
            json_string = f.read()
        self.model = model_from_json(json_string)
        self.model.load_weights(lstm_weights)

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


def get_predictor():
    model = AutumnModel('../deepTest-master/models/autumn-cnn-model-tf.meta',
                        '../deepTest-master/models/autumn-lstm-model-keras.json',
                        '../deepTest-master/models/autumn-cnn-weights.ckpt',
                        '../deepTest-master/models/autumn-lstm-weights.hdf5')
    return lambda img: model.predict(img)
