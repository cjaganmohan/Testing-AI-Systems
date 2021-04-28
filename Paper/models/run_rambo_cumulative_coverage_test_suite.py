import argparse
import csv
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import deque
from keras.models import Model as Kmodel
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from natsort import natsorted
from ncoverage import NCoverage
from scipy import misc
from scipy.misc import imread, imresize, imsave
from scipy.misc import imshow
from skimage.exposure import rescale_intensity
import theano
import keras

reload(sys)
sys.setdefaultencoding('ISO-8859-1')


class Model(object):
    def __init__(self,
                 model_path,
                 X_train_mean_path):

        self.model = load_model(model_path)
        self.model.compile(optimizer="adam", loss="mse")
        self.X_mean = np.load(X_train_mean_path)
        self.mean_angle = np.array([-0.004179079])
        print self.mean_angle
        self.img0 = None
        self.state = deque(maxlen=2)

        self.threshold = 0.2
        # self.nc = NCoverage(self.model,self.threshold)
        s1 = self.model.get_layer('sequential_1')
        self.nc1 = NCoverage(s1, self.threshold)
        # print(s1.summary())

        s2 = self.model.get_layer('sequential_2')
        # print(s2.summary())
        self.nc2 = NCoverage(s2, self.threshold)

        s3 = self.model.get_layer('sequential_3')
        # print(s3.summary())
        self.nc3 = NCoverage(s3, self.threshold)

        i1 = self.model.get_layer('input_1')

        self.i1_model = Kmodel(input=self.model.inputs, output=i1.output)

    def predict(self, img):
        img_path = 'test.jpg'
        misc.imsave(img_path, img)
        img1 = load_img(img_path, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)

        if self.img0 is None:
            self.img0 = img1
            return 0, 0, self.mean_angle[0], 0, 0, 0, 0, 0, 0

        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)  # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            return 0, 0, self.mean_angle[0], 0, 0, 0, 0, 0, 0

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

            # print(self.model.summary())
            # for layer in self.model.layers:
            # print (layer.name)

            i1_outputs = self.i1_model.predict(X)
            '''
            layerlist1 = self.nc1.update_coverage(i1_outputs)
            covered_neurons1, total_neurons1, p = self.nc1.curr_neuron_cov()
            c1 = covered_neurons1
            t1 = total_neurons1
            
            layerlist2 = self.nc2.update_coverage(i1_outputs)
            covered_neurons2, total_neurons2, p = self.nc2.curr_neuron_cov()
            c2 = covered_neurons2
            t2 = total_neurons2
            
            layerlist3 = self.nc3.update_coverage(i1_outputs)
            covered_neurons3, total_neurons3, p = self.nc3.curr_neuron_cov()
            c3 = covered_neurons3
            t3 = total_neurons3
            covered_neurons = covered_neurons1 + covered_neurons2 + covered_neurons3
            total_neurons  = total_neurons1 + total_neurons2 + total_neurons3
            '''
            rs1 = self.s1_model.predict(i1_outputs)
            rs2 = self.s2_model.predict(i1_outputs)
            rs3 = self.s3_model.predict(i1_outputs)
            # return covered_neurons, total_neurons, self.model.predict(X)[0][0],c1,t1,c2,t2,c3,t3
            return 0, 0, self.model.predict(X)[0][0], rs1[0][0], rs2[0][0], rs3[0][0], 0, 0, 0

    def predict1(self, img, transform, params):
        img_path = 'test.jpg'
        misc.imsave(img_path, img)
        img1 = load_img(img_path, grayscale=True, target_size=(192, 256))
        img1 = img_to_array(img1)

        if self.img0 is None:
            self.img0 = img1
            return 0, 0, self.mean_angle[0], 0, 0, 0, 0, 0, 0, 0, 0, 0

        elif len(self.state) < 1:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)  # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            return 0, 0, self.mean_angle[0], 0, 0, 0, 0, 0, 0, 0, 0, 0

        else:
            img = img1 - self.img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)  # to replicate initial model
            self.state.append(img)
            self.img0 = img1

            X = np.concatenate(self.state, axis=-1)

            if transform != None and params != None:
                X = transform(X, params)

            X = X[:, :, ::-1]
            X = np.expand_dims(X, axis=0)
            X = X.astype('float32')
            X -= self.X_mean
            X /= 255.0

            # print(self.model.summary())
            # for layer in self.model.layers:
            # print (layer.name)

            i1_outputs = self.i1_model.predict(X)

            d1 = self.nc1.update_coverage(i1_outputs)
            covered_neurons1, total_neurons1, p = self.nc1.curr_neuron_cov()
            c1 = covered_neurons1
            t1 = total_neurons1

            d2 = self.nc2.update_coverage(i1_outputs)
            covered_neurons2, total_neurons2, p = self.nc2.curr_neuron_cov()
            c2 = covered_neurons2
            t2 = total_neurons2

            d3 = self.nc3.update_coverage(i1_outputs)
            covered_neurons3, total_neurons3, p = self.nc3.curr_neuron_cov()
            c3 = covered_neurons3
            t3 = total_neurons3
            covered_neurons = covered_neurons1 + covered_neurons2 + covered_neurons3
            total_neurons = total_neurons1 + total_neurons2 + total_neurons3

            return covered_neurons, total_neurons, self.model.predict(X)[0][0], c1, t1, d1, c2, t2, d2, c3, t3, d3
            # return 0, 0, self.model.predict(X)[0][0],rs1[0][0],rs2[0][0],rs3[0][0],0,0,0

    def hard_reset(self):

        self.mean_angle = np.array([-0.004179079])
        # print self.mean_angle
        self.img0 = None
        self.state = deque(maxlen=2)
        self.threshold = 0.2
        # self.nc.reset_cov_dict()
        self.nc1.reset_cov_dict()
        self.nc2.reset_cov_dict()
        self.nc3.reset_cov_dict()

    def soft_reset(self):

        self.mean_angle = np.array([-0.004179079])
        print self.mean_angle
        self.img0 = None
        self.state = deque(maxlen=2)
        self.threshold = 0.2


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
        blur = cv2.GaussianBlur(img, (4, 4), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 4)
    if params == 9:
        blur = cv2.medianBlur(img, 5)
    if params == 10:
        blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def update_dict(dict1, covdict):
    '''
    Update neuron coverage dictionary dict1 with covered neurons in covdict
    '''
    r = False
    for k in covdict.keys():
        if covdict[k] and not dict1[k]:
            dict1[k] = True
            r = True
    return r


def is_update_dict(dict1, covdict):
    '''
    Return True if there are neurons covered in dictionary covdict but not covered in dict1
    '''
    for k in covdict.keys():
        if covdict[k] and not dict1[k]:
            return True
    return False


def get_current_coverage(covdict):
    '''
    Extract the covered neurons from the neuron coverage dictionary defined in ncoverage.py.
    '''
    covered_neurons = len([v for v in covdict.values() if v])
    total_neurons = len(covdict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def get_dict_contents(covdict):
    '''
    Print all the key values from the dictonary
    :param dict:
    :return:
    '''
    for key in covdict.keys():
        print(key, '->', covdict[key])


def is_update_dict_count(dict1, covdict):
    '''
    Return the number of additional neurons covered in covdict but not covered in dict1
    '''
    counter = 0
    for k in covdict.keys():
        if covdict[k] and not dict1[k]:
            counter = counter + 1
    return counter


def rambo_testgen_coverage(dataset_path, group_number, file_type, output_path):
    # model = Model("../models/final_model.hdf5", "../models/X_train_mean.npy")
    model = Model("./final_model.hdf5", "./X_train_mean.npy")

    # output_file information
    csv_filename = 'Rambo-Coverage_2-way-new' + '_Group' + str(group_number) + '.csv'
    txt_filename = 'Rambo-Coverage_2-way-new' + '_Group' + str(group_number) + '.txt'

    save_console_output = str(output_path) + '/' + txt_filename
    sys.stdout = open(save_console_output, 'w')
    output_as_csv = str(output_path) + '/' + csv_filename

    # Baseline coverage starts
    dict1 = dict(model.nc1.cov_dict)
    dict2 = dict(model.nc2.cov_dict)
    dict3 = dict(model.nc3.cov_dict)

    Covered_Neurons_Baseline = 0  # covered neurons
    Percentage_Covered_Baseline = 0  # covered percentage
    Total_Neurons_Baseline = 0  # total neurons

    seed_inputs1 = '/home/jagan/Desktop/Rambo/Baseline/' + 'Grp' + str(group_number)
    print(seed_inputs1)
    filelist1 = []
    for file in sorted(os.listdir(seed_inputs1)):
        if file.endswith(".jpg"):
            filelist1.append(file)

    # load, predict and save the results
    with open(output_as_csv, 'ab', 0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['seed_image', 'predicted_value', 'threshold', 'covered_neurons', 'total_neurons',
                         's1_covered', 's1_total',
                         's2_covered', 's2_total',
                         's3_covered', 's3_total',
                         's1_Additional', 's2_Additional', 's3_Additional',
                         's1_Cumulative', 's2_Cumulative', 's3_Cumulative'
                         ])

        # running for Original image (baseline)
        input_images = xrange(2, 3)
        for i in input_images:
            # j = i * 5
            j = i
            csvrecord = []
            seed_image = imread(os.path.join(seed_inputs1, filelist1[j - 2]))
            print(os.path.join(seed_inputs1, filelist1[j - 2]))
            new_covered, new_total, result, c1, t1, d1, c2, t2, d2, c3, t3, d3 = model.predict1(seed_image, None, None)

            seed_image = imread(os.path.join(seed_inputs1, filelist1[j - 1]))
            print(os.path.join(seed_inputs1, filelist1[j - 1]))
            new_covered, new_total, result, c1, t1, d1, c2, t2, d2, c3, t3, d3 = model.predict1(seed_image, None, None)

            seed_image = imread(os.path.join(seed_inputs1, filelist1[j]))
            print(os.path.join(seed_inputs1, filelist1[j]))
            new_covered, new_total, result, c1, t1, d1, c2, t2, d2, c3, t3, d3 = model.predict1(seed_image, None, None)
            print(new_covered, new_total, result, c1, t1, c2, t2, c3, t3)

            # update  coverage, dict1, dict2, dict3
            update_dict(dict1, d1)
            update_dict(dict2, d2)
            update_dict(dict3, d3)
            # get cumulative coverage
            covered1, total1, p1 = get_current_coverage(dict1)
            covered2, total2, p2 = get_current_coverage(dict2)
            covered3, total3, p3 = get_current_coverage(dict3)

            # get baseline coverage
            Covered_Neurons_Baseline = c1 + c2 + c3  # covered neurons
            Total_Neurons_Baseline = t1 + t2 + t3  # total neurons
            Percentage_Covered_Baseline = Covered_Neurons_Baseline / float(Total_Neurons_Baseline)  # covered percentage
            print(c1, c2, c3)
            print('Baseline coverage complete')
            print("--------------------------")
            print()

            # writing to CSV
            csvrecord.append(str(os.path.join(seed_inputs1, filelist1[j])))
            csvrecord.append(result)
            csvrecord.append(model.threshold)
            csvrecord.append(new_covered)
            csvrecord.append(new_total)

            csvrecord.append(c1)
            csvrecord.append(t1)

            csvrecord.append(c2)
            csvrecord.append(t2)

            csvrecord.append(c3)
            csvrecord.append(t3)

            csvrecord.append('-')
            csvrecord.append('-')
            csvrecord.append('-')

            csvrecord.append(covered1)
            csvrecord.append(covered2)
            csvrecord.append(covered3)
            print("-----------")
            writer.writerow(csvrecord)
            # reset model
            model.hard_reset()

            # print(d1)
        # Baseline coverage ends

        # t-way images
        r = []
        for subdir in natsorted([x[0] for x in os.walk(dataset_path)]):
            if subdir != dataset_path:
                print(subdir)
                seed_inputs1 = subdir
                # seed_inputs1 = os.path.join(dataset_path, "center/")
                # seed_labels1 = os.path.join(dataset_path, "final_evaluation.csv")
                # seed_inputs2 = os.path.join(dataset_path, "testData/")
                # seed_labels2 = os.path.join(dataset_path, "testData/test_steering.csv")

                # model = Model("../models/final_model.hdf5", "../models/X_train_mean.npy")
                print('cleaning up the filelist1')
                filelist1 = []
                for file in sorted(os.listdir(seed_inputs1)):
                    if file.endswith(".jpg"):
                        filelist1.append(file)

                C = 0  # covered neurons
                P = 0  # covered percentage
                T = 0  # total neurons

                # seed input
                input_images = xrange(2, 3)
                for i in input_images:
                    # j = i * 5
                    j = i
                    csvrecord = []
                    seed_image = imread(os.path.join(seed_inputs1, filelist1[j - 2]))
                    # seed_image = imread(os.path.join(seed_inputs1, filelist1[j]))
                    print(os.path.join(seed_inputs1, filelist1[j - 2]))
                    new_covered, new_total, result, c1, t1, d1, c2, t2, d2, c3, t3, d3 = model.predict1(seed_image,
                                                                                                        None, None)

                    seed_image = imread(os.path.join(seed_inputs1, filelist1[j - 1]))

                    print(os.path.join(seed_inputs1, filelist1[j - 1]))
                    new_covered, new_total, result, c1, t1, d1, c2, t2, d2, c3, t3, d3 = model.predict1(seed_image,
                                                                                                        None, None)

                    seed_image = imread(os.path.join(seed_inputs1, filelist1[j]))
                    print(os.path.join(seed_inputs1, filelist1[j]))
                    new_covered, new_total, result, c1, t1, d1, c2, t2, d2, c3, t3, d3 = model.predict1(seed_image,
                                                                                                        None, None)
                    print(new_covered, new_total, result, c1, t1, c2, t2, c3, t3)

                    print(c1, c2, c3)

                    # check if some cumulative coverage is increased
                    b1 = is_update_dict(dict1, d1)
                    b2 = is_update_dict(dict2, d2)
                    b3 = is_update_dict(dict3, d3)

                    print("transformed_image_coverage_completed")
                    print("--------- now for comparison ---------")
                    print('Baseline:  ', Covered_Neurons_Baseline, ' synthetic_image_2-way:  ', c1 + c2 + c3)

                    print('CNN_1:', b1)
                    print('CNN_2:', b2)
                    print('CNN_3:', b3)

                    print('Coverage increased')
                    print('CNN 1 increased by --', is_update_dict_count(dict1, d1))
                    print('CNN 2 increased by --', is_update_dict_count(dict2, d2))
                    print('CNN 3 increased by --', is_update_dict_count(dict3, d3))

                    # writing to CSV
                    csvrecord.append(str(os.path.join(seed_inputs1, filelist1[j])))
                    csvrecord.append(result)
                    csvrecord.append(model.threshold)
                    csvrecord.append(new_covered)
                    csvrecord.append(new_total)

                    csvrecord.append(c1)
                    csvrecord.append(t1)

                    csvrecord.append(c2)
                    csvrecord.append(t2)

                    csvrecord.append(c3)
                    csvrecord.append(t3)

                    csvrecord.append(is_update_dict_count(dict1, d1))
                    csvrecord.append(is_update_dict_count(dict2, d2))
                    csvrecord.append(is_update_dict_count(dict3, d3))

                    # get cumulative coverage for output
                    bcovered1, btotal1, bp1 = get_current_coverage(dict1)
                    bcovered2, btotal2, bp2 = get_current_coverage(dict2)
                    bcovered3, btotal3, bp3 = get_current_coverage(dict3)
                    print('Before update_ cumulative coverage:   ', bcovered1, bcovered2, bcovered3)
                    # update cumulative coverage
                    update_dict(dict1, d1)
                    update_dict(dict2, d2)
                    update_dict(dict3, d3)

                    # get cumulative coverage for output
                    covered1, total1, p1 = get_current_coverage(dict1)
                    covered2, total2, p2 = get_current_coverage(dict2)
                    covered3, total3, p3 = get_current_coverage(dict3)
                    print('synthetic_image_2-way image coverage:', c1, c2, c3)
                    print('Revised cumulative coverage:   ', covered1, covered2, covered3)

                    csvrecord.append(covered1)
                    csvrecord.append(covered2)
                    csvrecord.append(covered3)

                    writer.writerow(csvrecord)
                    model.hard_reset()

                    if b1 or b2 or b3:
                        print('Coverage increased')
                        # print('Coverage increased')
                        # print('CNN 1 increased by --', is_update_dict_count(dict1, d1))
                        # print('CNN 2 increased by --', is_update_dict_count(dict2, d2))
                        # print('CNN 3 increased by --', is_update_dict_count(dict3, d3))
                        #
                        # # get cumulative coverage for output
                        # bcovered1, btotal1, bp1 = get_current_coverage(dict1)
                        # bcovered2, btotal2, bp2 = get_current_coverage(dict2)
                        # bcovered3, btotal3, bp3 = get_current_coverage(dict3)
                        # print('Before update_ cumulative coverage:   ', bcovered1, bcovered2, bcovered3)
                        # # update cumulative coverage
                        # update_dict(dict1, d1)
                        # update_dict(dict2, d2)
                        # update_dict(dict3, d3)
                        #
                        # # get cumulative coverage for output
                        # covered1, total1, p1 = get_current_coverage(dict1)
                        # covered2, total2, p2 = get_current_coverage(dict2)
                        # covered3, total3, p3 = get_current_coverage(dict3)
                        #
                        # model.hard_reset()
                        # # C = covered1 + covered2 + covered3
                        # T = total1 + total2 + total3
                        # # P = C / float(T)
                        # print('T-way image coverage:', c1, c2, c3)
                        # print('Revised cumulative coverage:   ', covered1, covered2, covered3)
                        #

                    print("-----------")
                print("images_from_2-way_transformation_testcase_ complete")
        print("all done")
        print('---------------')
        print('Environment information: ', '  Backend - Theano: ', theano.__version__, '  Keras: ', keras.__version__)
        print('Completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--group', type=str)
    parser.add_argument('--file_type', type=str)  # {'Baseline', 'Individual_transformation','2-way'}
    parser.add_argument('--output_path', type=str)

    args, unknown = parser.parse_known_args()
    print('Calling the Rambo model now ----- ')
    print(args.dataset, args.group, args.file_type, args.output_path)
    rambo_testgen_coverage(args.dataset, args.group, args.file_type, args.output_path)
