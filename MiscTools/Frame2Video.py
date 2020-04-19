# reference URL: https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
# reference URL: https://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html

import cv2
import numpy as np
import os
import argparse
from natsort import natsorted, ns


def convert_frames_to_video(input_Frames, videofile_location, fps):
    frame_array = []
    #files = []
    #print(input_Frames)
    #for image_file in sorted(os.listdir(input_Frames)):
    for image_file in natsorted(os.listdir(input_Frames)):
        if not image_file.startswith(".") and image_file.endswith(".jpg"):
            filename = input_Frames + image_file
            print(filename)
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            #print(size)
            frame_array.append(img)  # inserting the frames into an image array
            #files.append(image_file)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    #out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    out = cv2.VideoWriter(videofile_location, fourcc, fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=str,help='path for input frames')
    parser.add_argument('--video', type=str,help='path for output (video)')
    args, unknown = parser.parse_known_args()
    fps = 25.0
    #print(args.frames)
    #print(args.video)

    convert_frames_to_video(args.frames, args.video,fps)