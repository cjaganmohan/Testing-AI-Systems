from __future__ import print_function

import argparse
import csv
import cv2
import numpy as np
import os
import shutil
import sys
from collections import deque
from keras import backend as K
from keras import metrics
from keras.models import model_from_json
from natsort import natsorted, ns
from ncoverage import NCoverage
from scipy import misc
import tensorflow as tf
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="0,1" #To use multiple GPU's

reload(sys)
sys.setdefaultencoding('utf8')

# keras 1.2.2 tf:1.2.0
# file modified by Jagan....

class ChauffeurModel(object):
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
        self.nc_encoder = NCoverage(self.encoder, self.threshold_cnn,
                                    exclude_layer=['pool', 'fc', 'flatten'],
                                    only_layer=only_layer)
        self.steps = deque()
        #print(self.lstm.summary())
        #self.nc = NCoverage(self.lstm,self.threshold)

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
            img = ((img-(255.0/2))/255.0)

            return self.cnn.predict_on_batch(img.reshape((1, 120, 320, 3)))[0, 0] / self.scale

        return predict_fn

    #def make_stateful_predictor(self):
        #steps = deque()

    def predict_fn(self, img, dummy=2):
        # preprocess image to be YUV 320x120 and equalize Y histogram
        steps = self.steps
        img = cv2.resize(img, (320, 240))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = img[120:240, :, :]
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img = ((img-(255.0/2))/255.0)
        img1 = img
        # apply feature extractor
        start_time = datetime.now() # Jagan
        img = self.encoder.predict_on_batch(img.reshape((1, 120, 320, 3)))
        end_time = datetime.now() #Jagan
        print('Model Prediction time', str(end_time-start_time))
         # initial fill of timesteps
        if not len(steps):
            for _ in xrange(self.timesteps):
                steps.append(img)

        # put most recent features at end
        steps.popleft()
        steps.append(img)
        #print(len(steps))
        #timestepped_x = np.empty((1, self.timesteps, img.shape[1]))
        if dummy == 0:
            return 0, 0, 0, 0, 0, 0, 0
        for i, img in enumerate(steps):
            self.timestepped_x[0, i] = img

        '''
        self.nc.update_coverage(timestepped_x)
        covered_neurons, total_neurons, p = self.nc.curr_neuron_cov()
        print('input covered {} neurons'.format(covered_neurons))
        print('total {} neurons'.format(total_neurons))
        print('percentage {}'.format(p))
        '''
        cnn_ndict = self.nc_encoder.update_coverage(img1.reshape((1, 120, 320, 3)))
        cnn_covered_neurons, cnn_total_neurons, p = self.nc_encoder.curr_neuron_cov()
        if dummy == 1:
            return cnn_ndict, cnn_covered_neurons, cnn_total_neurons, 0, 0, 0, 0
        lstm_ndict = self.nc_lstm.update_coverage(self.timestepped_x)
        lstm_covered_neurons, lstm_total_neurons, p = self.nc_lstm.curr_neuron_cov()
        return cnn_ndict, cnn_covered_neurons, cnn_total_neurons, lstm_ndict,\
        lstm_covered_neurons, lstm_total_neurons,\
        self.lstm.predict_on_batch(self.timestepped_x)[0, 0] / self.scale



    #return predict_fn

def image_translation(img, params):

    rows, cols, ch = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_scale(img, params):

    res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
    return res

def image_shear(img, params):
    rows, cols, ch = img.shape
    factor = params*(-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_rotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha
    #new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

    return new_img

def image_brightness(img, params):
    beta = params
    new_img = cv2.add(img, beta)                                  # new_img = img*alpha + beta

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

def rmse(y_true, y_pred):
    '''Calculates RMSE
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

metrics.rmse = rmse

def chauffeur_testgen_coverage(dataset_path,transformation_name, directory_name, group_number):

    #Jagan changes starts
    csv_filename = 'Chauffeur-' + 'Group_' + str(group_number) +'_' + str(transformation_name) + '_Neuron-Coverage.csv'
    txt_filename = 'Chauffeur-' + 'Group_' + str(group_number) +'_' + str(transformation_name) + '_Neuron-Coverage.txt'

    save_console_output = '/home/jagan/Desktop/chauffer-deubgging/prediction-in-batches/Results' \
                          '/Subject_Image_Transformed_Coverage/Grp' + str(
        group_number) + '/' + txt_filename
    sys.stdout = open(save_console_output, 'w')
    #Jagan changes ends


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

    #seed_inputs1 = os.path.join(dataset_path, "hmb3/")
    #seed_labels1 = os.path.join(dataset_path, "hmb3/hmb3_steering.csv")
    #seed_inputs2 = os.path.join(dataset_path, "center/")
    seed_inputs1 = os.path.join(dataset_path, "testData/")
    seed_labels1 = os.path.join(dataset_path, "testData/test_steering.csv")
    seed_inputs2 = os.path.join(dataset_path, "center-copy/")
    seed_labels2 = os.path.join(dataset_path, "final_evaluation.csv")


    filelist1 = []
    for file in sorted(os.listdir(seed_inputs1)):
        if file.endswith(".jpg"):
            filelist1.append(file)

    with open(seed_labels1, 'rb') as csvfile1:
        label1 = list(csv.reader(csvfile1, delimiter=',', quotechar='|'))
    label1 = label1[1:]

    # Jagan code changes
    filelist2 = []
    # for image_file in sorted(os.listdir(seed_inputs2)):
    #     if image_file.endswith(".jpg"):
    #         filelist2.append(image_file)
    with open(seed_labels2, 'rb') as csvfile2:
        label2 = list(csv.reader(csvfile2, delimiter=',', quotechar='|'))

    label2 = label2[1:]
    file_counter = 1

    for i in label2:

        if file_counter % 100 == 0:

            sourceLocation_transformedImage = directory_name + str(i[0]) + ".jpg"
            print('Copying the transformed image for group from  ' + sourceLocation_transformedImage)

            destination = dataset_path + 'center-copy/'
            shutil.copy(sourceLocation_transformedImage, destination)

            print('Copying the transformed image ---' + i[0] + ' for group completed')
            filelist2.append(i[0] + ".jpg")
        else:
            # print(i[0]+".jpg")
            filelist2.append(i[0] + ".jpg")
        # filelist2.append(i[0]+".jpg")
        print(file_counter)
        file_counter = file_counter + 1

    # Jagan code changes
    filename = '/home/jagan/Desktop/chauffer-deubgging/prediction-in-batches/Results/Subject_Image_Transformed_Coverage/Grp' + str(
        group_number) + '/' + csv_filename

    #seed inputs
    with open(filename, 'ab', 0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(['index', 'image', 'tranformation', 'param_name', 'param_value',
        #                  'cnn_threshold', 'cnn_covered_neurons', 'cnn_total_neurons',
        #                  'cnn_covered_detail', 'lstm_threshold', 'lstm_covered_neurons',
        #                  'lstm_total_neurons', 'lstm_covered_detail', 'y_hat', 'label'])

        writer.writerow(['File name', 'cnn_covered_neurons' ,
              'cnn_total_neurons ' , 'lstm_covered_neurons', 'lstm_total_neurons', 'Predicted Steering Angle'])

        #Jagan - code changes begins
        for f in filelist2:
            seed_image = cv2.imread(os.path.join(seed_inputs2, f))
            csvrecord = []
            #seed_image = cv2.imread(os.path.join(seed_inputs1, filelist1[j]))

            cnn_ndict, cnn_covered_neurons, cnn_total_neurons, lstm_ndict, \
            lstm_covered_neurons, lstm_total_neurons, yhat = model.predict_fn(seed_image)

            tempk = []
            for k in cnn_ndict.keys():
                if cnn_ndict[k]:
                    tempk.append(k)
            tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
            cnn_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

            tempk = []
            for k in lstm_ndict.keys():
                if lstm_ndict[k]:
                    tempk.append(k)
            tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
            lstm_covered_detail = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

            writer.writerow([f, str(cnn_covered_neurons), str(cnn_total_neurons), str(lstm_covered_neurons), str(
                lstm_total_neurons), str(yhat)])

            model.nc_encoder.reset_cov_dict()
            model.nc_lstm.reset_cov_dict()
            # print(covered_neurons)
            # covered_neurons = nc.get_neuron_coverage(test_x)
            # print('input covered {} neurons'.format(covered_neurons))
            # print('total {} neurons'.format(total_neurons))

            # print("File name:" + f + " cnn_covered_neurons: " + str(cnn_covered_neurons) +
            #       " cnn_total_neurons: " + str(cnn_total_neurons) +
            #       " lstm_covered_neurons: " + str(lstm_covered_neurons) + " lstm_total_neurons: " + str(
            #     lstm_total_neurons) +  " Predicted Steering Angle: " + str(
            #     yhat)+" cnn_covered_detail: " + str(cnn_covered_detail)+ "lstm_covered_detail: "+ str(lstm_covered_detail))

            # writer.writerow([f, str(cnn_covered_neurons) , str(cnn_total_neurons) , str(lstm_covered_neurons) , str(
            #     lstm_total_neurons) , str(yhat)])

            # filename, ext = os.path.splitext(str(f))
            # if label1[j][0] != filename:
            #     print(filename + " not found in the label file")
            #     continue

            # csvrecord.append(seed_inputs1)
            # csvrecord.append(str(filelist1[seed_inputs1]))
            # csvrecord.append('-')
            # csvrecord.append('-')
            # csvrecord.append('-')
            # csvrecord.append(model.threshold_cnn)
            #
            # csvrecord.append(cnn_covered_neurons)
            # csvrecord.append(cnn_total_neurons)
            # csvrecord.append(cnn_covered_detail)
            # csvrecord.append(model.threshold_lstm)
            #
            # csvrecord.append(lstm_covered_neurons)
            # csvrecord.append(lstm_total_neurons)
            # csvrecord.append(lstm_covered_detail)
            #
            # csvrecord.append(yhat)
            # #csvrecord.append(label1[j][1])
            # print(csvrecord[:8])
            # writer.writerow(csvrecord)

        print("seed input done")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/media/yuchi/345F-2D0F/',
                        help='path for dataset')
    parser.add_argument('--transformation', type=str)
    parser.add_argument('--directory', type=str)
    parser.add_argument('--group', type=str)
    args, unknown = parser.parse_known_args()
    #args = parser.parse_args()
    print(args.dataset)
    print(args.transformation)
    print(args.directory)
    print(args.group)
    chauffeur_testgen_coverage(args.dataset, args.transformation, args.directory, args.group)
    #args = parser.parse_args()
    #group_number = 2
    #chauffeur_testgen_coverage(args.dataset,'TransformedImages_Blur_Gaussian_3', '/home/jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations/TransformedImages_Blur_Gaussian_3/', group_number)
    # for item in natsorted(
    #         os.listdir('/home/jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations')):
    #     if not item.startswith('.'):
    #         transformation_name = item
    #         directory_name = '/home/jagan/Desktop/chauffer-deubgging/prediction-in-batches/IndividualTransformations' + "/" + transformation_name + "/"
    #         chauffeur_testgen_coverage(args.dataset, transformation_name, directory_name, group_number)  # updated by Jagan
