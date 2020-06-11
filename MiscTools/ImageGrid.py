'''
Display jpeg images in a grid -- https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib

'''
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import ImageGrid
from natsort import natsorted, ns

fig = plt.figure(figsize=(11, 11))
input_dir = '/Users/Jagan/Desktop/2-way/Grp2'
image_array= []

for image_file in natsorted(os.listdir(input_dir)):
    if not image_file.startswith(".") and image_file.endswith(".jpg"):
        filename = input_dir + image_file
        print(filename)
        img = cv2.imread(filename)
        #print(size)
        image_array.append(img)  # inserting the frames into an image array

print(len(image_array))

# for i in range(120):
#
#     #plt.subplot(12,12,i+1)
#     plt.axis("off")  # To hide the axis lables
#     plt.grid(b=None)  # To hide the grid
#     plt.title('Test image # '+ str(i))
#     plt.imshow(image_array[i])
#
#     plt.show()


for i in range(len(image_array)):
    plt.subplot(12,12,i+1)
    plt.axis("off")  # To hide the axis lables
    plt.grid(b=None)  # To hide the grid
    plt.title(str(i+1))
    #plt.tight_layout()
    plt.imshow(image_array[i])
plt.show()