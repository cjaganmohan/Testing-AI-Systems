# file to test natsort library

import argparse
import os
from natsort import natsorted, ns

def sort_and_print_file_names(filedir):
    for file in natsorted(os.listdir(filedir)):
        if not file.startswith(".") and file.endswith(".jpg"):
            fileName = filedir+file
            print(fileName)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,help="path for input directory")
    args, unknown = parser.parse_known_args()
    print(args.input)
    sort_and_print_file_names(args.input)