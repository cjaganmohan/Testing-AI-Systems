'''
Results video generator Udacity Challenge 2
Original By: Comma.ai Revd: Chris Gundling -- https://github.com/commaai/research/blob/master/view_steering_model.py
Source referenced from: /Udacity/Challenge-2/steering-models/communitu-models/cg23/epoch_viewer.py
'''

from __future__ import print_function

import argparse
import cv2
import h5py
import json
import matplotlib
import numpy as np
import pandas as pd
import pdb
# import pygame
import sys
# from keras.models import model_from_json
import time
from os import path

matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import pylab
# from pygame.locals import *

# from data_TS import *

# pygame.init()
# size = (320 * 2, 160 * 3)
# size2 = (640,160)
# pygame.display.set_caption("Steering angle prediction - visualizer")
# screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
# screen.set_alpha(None)

# camera_surface = pygame.surface.Surface((320,160),0,24).convert()
# camera_surface = pygame.surface.Surface((320, 160), 0, 24).convert()
# clock = pygame.time.Clock()

# ***** get perspective transform for images *****
from skimage import transform as tf

rsrc = \
    [[43.45456230828867, 118.00743250075844],
     [104.5055617352614, 69.46865203761757],
     [114.86050156739812, 60.83953551083698],
     [129.74572757609468, 50.48459567870026],
     [132.98164627363735, 46.38576532847949],
     [301.0336906326895, 98.16046448916306],
     [238.25686790036065, 62.56535881619311],
     [227.2547443287154, 56.30924933427718],
     [209.13359962247614, 46.817221154818526],
     [203.9561297064078, 43.5813024572758]]
rdst = \
    [[10.822125594094452, 1.42189132706374],
     [21.177065426231174, 1.5297552836484982],
     [25.275895776451954, 1.42189132706374],
     [36.062291434927694, 1.6376192402332563],
     [40.376849698318004, 1.42189132706374],
     [11.900765159942026, -2.1376192402332563],
     [22.25570499207874, -2.1376192402332563],
     [26.785991168638553, -2.029755283648498],
     [37.033067044190524, -2.029755283648498],
     [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))


#
# def perspective_tform(x, y):
#     p1, p2 = tform3_img((x, y))[0]
#     return p2, p1
#
#
# # ***** functions to draw lines *****
# def draw_pt(img, x, y, color, sz=1):
#     row, col = perspective_tform(x, y)
#     if row >= 0 and row < img.shape[0] and \
#             col >= 0 and col < img.shape[1]:
#         img[row - sz:row + sz, col - sz:col + sz] = color
#
#
# def draw_path(img, path_x, path_y, color):
#     for x, y in zip(path_x, path_y):
#         draw_pt(img, x, y, color)
#
#
# # ***** functions to draw predicted path *****
#
# def calc_curvature(v_ego, angle_steers, angle_offset=0):
#     deg_to_rad = np.pi / 180.
#     slip_fator = 0.0014  # slip factor obtained from real data
#     steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
#     wheel_base = 2.67  # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/
#
#     angle_steers_rad = (angle_steers - angle_offset)  # * deg_to_rad
#     curvature = angle_steers_rad / (steer_ratio * wheel_base * (1. + slip_fator * v_ego ** 2))
#     return curvature
#
#
# def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
#     # *** this function returns the lateral offset given the steering angle, speed and the lookahead distance
#     curvature = calc_curvature(v_ego, angle_steers, angle_offset)
#
#     # clip is to avoid arcsin NaNs due to too sharp turns
#     y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999)) / 2.)
#     return y_actual, curvature
#
#
# def draw_path_on(img, speed_ms, angle_steers, color=(0, 0, 255)):
#     path_x = np.arange(0., 50.1, 0.5)
#     path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
#     draw_path(img, path_x, path_y, color)

def calc_curvature(v_ego, angle_steers, angle_offset=0):
    deg_to_rad = np.pi / 180.
    slip_fator = 0.0014  # slip factor obtained from real data
    steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
    wheel_base = 2.67  # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

    angle_steers_rad = (angle_steers - angle_offset)  # * deg_to_rad
    curvature = angle_steers_rad / (steer_ratio * wheel_base * (1. + slip_fator * v_ego ** 2))
    return curvature


def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
    # *** this function returns the lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999)) / 2.)
    return y_actual, curvature


# taken from https://github.com/commaai/research
def draw_path_on(img, speed_ms, angle_steers, color=(0, 0, 255)):
    height = img.shape[0]
    max_line_height = height // 3
    width_midpoint = img.shape[1] / 2
    height_offset = np.arange(0., max_line_height, 1.0)
    x_offset, _ = calc_lookahead_offset(speed_ms, angle_steers, height_offset)

    pts = [(width_midpoint + point[0], height - point[1]) for point in zip(x_offset, height_offset)]

    return draw_path(img, pts, color=color)


def draw_path(img, pts, thickness=1, color=(0, 0, 255)):
    for point in pts:
        # (x, y)
        top_left = (int(point[0] - thickness), int(point[1] - thickness))
        bottom_right = (int(point[0] + thickness), int(point[1] + thickness))
        cv2.line(img, top_left, bottom_right, color, thickness)

    return img


# ***** main loop *****
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path viewer')
    args = parser.parse_args()

    img_list = []
    # img_list.append(
    #     '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/Grp2_4289_4389/center/1479425660620933516.jpg')
    # print(img_list[0])
    #
    # img1 = '/Users/Jagan/Desktop/1479425660620933516_Contrast_2.6.jpg'
    # img2 = '/Users/Jagan/Desktop/1479425660620933516_Contrast_2.0.jpg'

    input_img = cv2.imread(
        '/home/jagan/Desktop/Rambo/2-way/Grp9/TestCase_100/1479425719031130839_TC_100_Grp9_Thres-0.1_Combination_2way.jpg')

    # df_test = pd.read_csv(
    #     '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/Results/Subject-Image-Transformed-IndiviualTransformation/Grp2.csv',
    #     usecols=['frame_id', 'predicted_angle'], index_col=None)
    # df_truth = pd.read_csv(
    #     '/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/Results/Subject-Image-Transformed-IndiviualTransformation/Grp2.csv',
    #     usecols=['frame_id', 'ground_truth'], index_col=None)

    red = (255, 0, 0)
    blue = (0, 0, 255)
    # myFont = pygame.font.SysFont("monospace", 18)
    # randNumLabel = myFont.render('Human Steer Angle:', 1, blue)
    # randNumLabel2 = myFont.render('Model Steer Angle:', 1, red)
    speed_ms = 5  # log['speed'][i]

    # Run through all images
    for i in range(1):
        # if i%100 == 0:
        #    print('%.2f seconds elapsed' % (i/20))
        # img = test_x[i, :, :, :].swapaxes(0, 2).swapaxes(0, 1)

        # img = input_img.swapaxes(0, 2).swapaxes(0, 1)
        img = input_img.copy()
        # predicted_steers = df_test['predicted_angle'].loc[i]
        # actual_steers = df_truth['ground_truth'].loc[i]
        output_dir = '/home/jagan/Dropbox/Self-driving-car-Results/Images/'
        fileName_withExtension = 'Autumn_Grp_9_TC_100.jpg'

        predicted_steers = -0.05902291  # predicted steering value for the original image # blue color
        actual_steers = -0.37407798  # predicted value of the synthetic image # red color

        print(predicted_steers)
        print(actual_steers)
        # img = img.transpose((0,1,2))
        draw_path_on(img, speed_ms, actual_steers / 5.0)
        draw_path_on(img, speed_ms, predicted_steers / 5.0, red)
        # pdb.set_trace()
        img = img.astype('u1')
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('t', img)
        outputFileDestination = output_dir + fileName_withExtension
        print(outputFileDestination)
        cv2.imwrite(outputFileDestination, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # draw on
        # pygame.surfarray.blit_array(camera_surface, img.swapaxes(0, 1))
        # img1 = img.transpose((0,2,1))
        # img1 = img.transpose((0,2,1))
        # img2 = cv2.resize(img1, (160, 320))
        # pygame.surfarray.blit_array(camera_surface, img2)
        #
        # camera_surface_2x = pygame.transform.scale2x(camera_surface)
        # screen.blit(camera_surface_2x, (0, 0))
        # pygame.display.flip()
