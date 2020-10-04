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
from ncoverage import NCoverage
from scipy import misc
from scipy.misc import imread, imresize, imsave
from scipy.misc import imshow
from skimage.exposure import rescale_intensity

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
        # img_path = 'test.jpg'
        # misc.imsave(img_path, img)
        print(img)
        img1 = load_img(img, grayscale=True, target_size=(192, 256))
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

            print(self.model.summary())
            for layer in self.model.layers:
                print (layer.name)

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


def rambo_testgen_coverage(dataset_path, group_number, file_type, output_path):
    # Load model weights
    model = Model("../models/final_model.hdf5", "../models/X_train_mean.npy")

    # sort the images
    images = []
    for root, sub_dirs, files in os.walk(dataset_path):
        for f in files:
            if '.jpg' in f or '.png' in f:
                images.append((root, f))
    images.sort(key=lambda x: x[1])

    # output_file information
    csv_filename = 'Rambo-Coverage_Information' + str(file_type) + '_Group' + str(group_number) + '.csv'
    txt_filename = 'Rambo-Coverage_Information' + str(file_type) + '_Group' + str(group_number) + '.txt'

    save_console_output = str(output_path) + '/' + txt_filename
    sys.stdout = open(save_console_output, 'w')
    output_as_csv = str(output_path) + '/' + csv_filename

    # load, predict and save the results
    with open(output_as_csv, 'ab', 0) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(['index', 'image', 'tranformation', 'param_name',
        #                  'param_value','threshold','covered_neurons', 'total_neurons',
        #                  's1_covered', 's1_total','s1_detail',
        #                  's2_covered', 's2_total','s2_detail',
        #                  's3_covered', 's3_total','s3_detail',
        #                  'y_hat','label'])

        # writer.writerow(['File name', 'covered_neurons', 'total_neurons',
        #                  's1_covered', 's1_total','s1_detail',
        #                  's2_covered', 's2_total','s2_detail',
        #                  's3_covered', 's3_total','s3_detail',
        #                  'y_hat'])
        writer.writerow(['File name', 'covered_neurons', 'total_neurons',
                         's1_covered', 's1_total',
                         's2_covered', 's2_total',
                         's3_covered', 's3_total',
                         'y_hat'])

        for image in images:
            # prediction = model.predict(os.path.join(image[0], image[1]))
            # print(image[1] + ',', str(prediction)[1:-1])
            print image
            img_path = os.path.join(image[0], image[1])
            # print img_path
            # print imread(os.path.join(image[0], image[1]))

            new_covered, new_total, result, c1, t1, d1, c2, t2, d2, c3, t3, d3 = model.predict1(img_path, None, None)
            new_covered, new_total, result, c1, t1, d1, c2, t2, d2, c3, t3, d3 = model.predict1(img_path, None, None)
            new_covered, new_total, result, c1, t1, d1, c2, t2, d2, c3, t3, d3 = model.predict1(img_path, None, None)

            tempk = []
            for k in d1.keys():
                if d1[k]:
                    tempk.append(k)
            tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
            # covered_detail1 = ';'.join(str(x) for x in tempk).replace(',', ':')
            covered_detail1 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')
            tempk = []
            for k in d2.keys():
                if d2[k]:
                    tempk.append(k)
            tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
            covered_detail2 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

            tempk = []
            for k in d3.keys():
                if d3[k]:
                    tempk.append(k)
            tempk = sorted(tempk, key=lambda element: (element[0], element[1]))
            covered_detail3 = ';'.join("('" + str(x[0]) + "', " + str(x[1]) + ')' for x in tempk).replace(',', ':')

            # writer.writerow([img_path,new_covered,new_total,
            #                  c1,t1,covered_detail1,
            #                  c2,t2,covered_detail2,
            #                  c3,t3,covered_detail3,
            #                  result])

            writer.writerow([img_path, new_covered, new_total,
                             c1, t1,
                             c2, t2,
                             c3, t3,
                             result])
            model.hard_reset()

        print("seed input done")

        # Jagan code change ends


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
