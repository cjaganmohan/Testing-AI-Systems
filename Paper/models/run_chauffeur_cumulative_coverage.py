'''
Leverage neuron coverage to guide the generation of images from combinations of transformations.
'''
from __future__ import print_function

import argparse
import argparse
import csv
import cv2
import numpy as np
import os
import pickle
import random
import sys
from collections import defaultdict
from collections import deque
from keras import backend as K
from keras import backend as K
from keras.models import model_from_json
from ncoverage import NCoverage
from scipy.misc import imread, imresize, imsave
from natsort import natsorted, ns

reload(sys)
sys.setdefaultencoding('utf8')


# keras 1.2.2 tf:1.2.0
class ChauffeurModel(object):
    '''
    Chauffeur model with integrated neuron coverage
    '''

    def __init__(self,
                 cnn_json_path,
                 cnn_weights_path,
                 lstm_json_path,
                 lstm_weights_path, only_layer=""):

        self.cnn = self.load_from_json(cnn_json_path, cnn_weights_path)
        self.encoder = self.load_encoder(cnn_json_path, cnn_weights_path)
        self.lstm = self.load_from_json(lstm_json_path, lstm_weights_path)

        # hardcoded from final submission model
        self.scale = 16.
        self.timesteps = 100

        self.threshold_cnn = 0.1
        self.threshold_lstm = 0.4
        self.timestepped_x = np.empty((1, self.timesteps, 8960))
        self.nc_lstm = NCoverage(self.lstm, self.threshold_lstm)
        self.nc_encoder = NCoverage(self.encoder, self.threshold_cnn, exclude_layer=['pool', 'fc', 'flatten'],
                                    only_layer=only_layer)
        self.steps = deque()
        # print(self.lstm.summary())
        # self.nc = NCoverage(self.lstm,self.threshold)

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

    # def make_stateful_predictor(self):
    # steps = deque()

    def predict_fn(self, img, test=0):
        # test == 0: update the coverage only
        # test == 1: test if the input image will increase the current coverage
        steps = self.steps
        img = cv2.resize(img, (320, 240))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = img[120:240, :, :]
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img = ((img - (255.0 / 2)) / 255.0)
        img1 = img

        if test == 1:
            return self.nc_encoder.is_testcase_increase_coverage(img1.reshape((1, 120, 320, 3)))
        else:
            cnn_ndict = self.nc_encoder.update_coverage(img1.reshape((1, 120, 320, 3)))
            cnn_covered_neurons, cnn_total_neurons, cnn_p = self.nc_encoder.curr_neuron_cov()
            return cnn_covered_neurons, cnn_total_neurons, cnn_p

    # return predict_fn


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def image_translation(img, params):
    if not isinstance(params, list):
        params = [params, params]
    rows, cols, ch = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_scale(img, params):
    if not isinstance(params, list):
        params = [params, params]
    res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
    return res


def image_shear(img, params):
    rows, cols, ch = img.shape
    factor = params * (-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_rotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha
    # new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

    return new_img


def image_brightness(img, params):
    beta = params
    new_img = cv2.add(img, beta)  # new_img = img*alpha + beta

    return new_img


def image_blur(img, params):
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    if params == 9:
        blur = cv2.blur(img, (6, 6))
    if params == 10:
        blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def rotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params[0], 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_brightness1(img, params):
    w = img.shape[1]
    h = img.shape[0]
    if params > 0:
        for xi in xrange(0, w):
            for xj in xrange(0, h):
                if 255 - img[xj, xi, 0] < params:
                    img[xj, xi, 0] = 255
                else:
                    img[xj, xi, 0] = img[xj, xi, 0] + params
                if 255 - img[xj, xi, 1] < params:
                    img[xj, xi, 1] = 255
                else:
                    img[xj, xi, 1] = img[xj, xi, 1] + params
                if 255 - img[xj, xi, 2] < params:
                    img[xj, xi, 2] = 255
                else:
                    img[xj, xi, 2] = img[xj, xi, 2] + params
    if params < 0:
        params = params * (-1)
        for xi in xrange(0, w):
            for xj in xrange(0, h):
                if img[xj, xi, 0] - 0 < params:
                    img[xj, xi, 0] = 0
                else:
                    img[xj, xi, 0] = img[xj, xi, 0] - params
                if img[xj, xi, 1] - 0 < params:
                    img[xj, xi, 1] = 0
                else:
                    img[xj, xi, 1] = img[xj, xi, 1] - params
                if img[xj, xi, 2] - 0 < params:
                    img[xj, xi, 2] = 0
                else:
                    img[xj, xi, 2] = img[xj, xi, 2] - params

    return img


def image_brightness2(img, params):
    beta = params
    b, g, r = cv2.split(img)
    b = cv2.add(b, beta)
    g = cv2.add(g, beta)
    r = cv2.add(r, beta)
    new_img = cv2.merge((b, g, r))
    return new_img


def chauffeur_guided(dataset_path, baseline_dir, group_number):
    model_name = "cnn"
    image_size = (128, 128)
    threshold = 0.2

    root = ""
    seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "Ch2_001/center/")
    seed_labels2 = os.path.join(dataset_path, "Ch2_001/CH2_final_evaluation.csv")
    new_input = "./new/"
    # Model build
    # ---------------------------------------------------------------------------------
    cnn_json_path = "./cnn.json"
    cnn_weights_path = "./cnn.weights"
    lstm_json_path = "./lstm.json"
    lstm_weights_path = "./lstm.weights"

    K.set_learning_phase(0)
    model = ChauffeurModel(
        cnn_json_path,
        cnn_weights_path,
        lstm_json_path,
        lstm_weights_path)


    # output_file information
    #csv_filename = 'Chauffeur_Cumulative_Coverage_Information' + str(file_type) + '_Group' + str(group_number) + '.csv'
    #txt_filename = 'Chauffeur_Cumulative_Coverage_Information' + str(file_type) + '_Group' + str(group_number) + '.txt'
    csv_filename = 'Chauffeur_Cumulative_Coverage_Information_t-way_' + '_Group' + str(group_number) + '.csv'
    txt_filename = 'Chauffeur_Cumulative_Coverage_Information_t-way_' + '_Group' + str(group_number) + '.txt'

    output_path = '/home/jagan/Dropbox/Self-driving-car-Results/Comparison/Chauffeur/'

    save_console_output = str(output_path) + '/' + txt_filename
    sys.stdout = open(save_console_output, 'w')

    output_as_csv = str(output_path) + '/' + csv_filename

    # load, predict and save the results
    with open(output_as_csv, 'ab', 0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['seed_image', 'covered_neurons', 'total_neurons', 'percentage_covered'])

        # Baseline coverage
        # seed_inputs1 = '/home/jagan/Desktop/Rambo/Baseline/Grp2/'
        seed_inputs1 = baseline_dir  # location of the original seed images
        filelist1 = []
        for file in sorted(os.listdir(seed_inputs1)):
            if file.endswith(".jpg"):
                filelist1.append(file)

        flag = 0

        Covered_Neurons_Baseline = 0  # covered neurons
        Percentage_Covered_Baseline = 0  # covered percentage
        Total_Neurons_Baseline = 0  # total neurons

        C = 0  # covered neurons
        P = 0  # covered percentage
        T = 0  # total neurons

        chauffeur_queue = deque()
        chauffeur_stack = []
        generated = 0

        cache = deque()

        # jagan changes starts
        # baseline neuron coverage
        print('now running baseline .... fetching seed inputs from .....' + seed_inputs1)
        input_images = xrange(99, 100)  # modified by Jagan
        for i in input_images:
            j = i  # modified by Jagan
            csvrecord = []
            image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))
            # image = cv2.imread(image_file)
            covered, total, p = model.predict_fn(image)

            Covered_Neurons_Baseline = covered
            Percentage_Covered_Baseline = p
            Total_Neurons_Baseline = total
            print(os.path.join(seed_inputs1, filelist1[j]))
            print("baseline neuron coverage information :  Covered Neurons- " + str(
                Covered_Neurons_Baseline) + ", Total Neurons:  " + str(
                Total_Neurons_Baseline) + ", Percentage covered: " +
                  str(Percentage_Covered_Baseline))
            print('Baseline coverage complete')
            print("---------------------------------------------------------------")
            print()

            #writing to csv
            # writing to CSV
            csvrecord.append(str(os.path.join(seed_inputs1, filelist1[j])))
            #csvrecord.append(model.threshold)
            csvrecord.append(Covered_Neurons_Baseline)
            csvrecord.append(Total_Neurons_Baseline)
            csvrecord.append(Percentage_Covered_Baseline)
            print("-----------")
            writer.writerow(csvrecord)
        # baseline coverage ends


        # test image coverage starts
        test_images_dir = dataset_path  # location of the synthetic images
        subdirs = [x[0] for x in os.walk((test_images_dir))]
        subdirs = natsorted(subdirs)
        # print(type(subdirs))

        for d in subdirs:
            if d != test_images_dir:
                print(d)
                test_inputs1 = d
                testimage_list = []
                for test_file in sorted(os.listdir(test_inputs1)):
                    if test_file.endswith(".jpg"):
                        testimage_list.append(test_file)

                input_images = xrange(99, 100)  # modified by Jagan
                for i in input_images:
                    j = i  # modified by Jagan
                    csvrecord=[]
                    test_image = cv2.imread(os.path.join(test_inputs1, testimage_list[j]))
                    print("Now running coverage for test image, " + os.path.join(test_inputs1, testimage_list[j]))
                if model.predict_fn(test_image, test=1):
                    print("Test image increases the coverage")
                    covered, total, p = model.predict_fn(test_image)
                    C = covered
                    T = total
                    P = p
                    print("Revised coverage - Covered Neurons: " + str(C) + ", Total Neurons: " + str(
                        T) + ", Percentage covered: " + str(P))

                    # writing to CSV
                    csvrecord.append(os.path.join(test_inputs1, testimage_list[j]))
                    #csvrecord.append(model.threshold)
                    csvrecord.append(C)
                    csvrecord.append(T)
                    csvrecord.append(P)
                    print("---------------------------------------------------------------")
                    writer.writerow(csvrecord)

                else:
                    print("Test image does not increase the  coverage")
                    covered, total, p = model.predict_fn(test_image)
                    C = covered
                    T = total
                    P = p
                    print("Revised coverage - Covered Neurons: " + str(C) + ", Total Neurons: " + str(
                        T) + ", Percentage covered: " + str(P))
                    # writing to CSV
                    csvrecord.append(os.path.join(test_inputs1, testimage_list[j]))
                    #csvrecord.append(model.threshold)
                    csvrecord.append(C)
                    csvrecord.append(T)
                    csvrecord.append(P)
                    csvrecord.append('-')
                    print("---------------------------------------------------------------")
                    writer.writerow(csvrecord)
        print("done")
        # jagan changes ends


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--baseline', type=str)
    parser.add_argument('--group', type=str)

    args, unknown = parser.parse_known_args()
    print('Calling the Chauffeur model now ----- ')
    print(args.dataset, args.baseline, args.group)
    chauffeur_guided(args.dataset, args.baseline, args.group)

