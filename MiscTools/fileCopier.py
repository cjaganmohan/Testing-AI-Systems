'''
Template python that is used for copying files/folders.
'''
import argparse
import os
import shutil
from natsort import natsorted, ns


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error in creating the directory  ' + directory)

def copyFilesAndFolder(rootDir, sourceFileDirectory):

    source_file1 = sourceFileDirectory + '/final_evaluation.csv'
    print(source_file1)
    source_file2 = sourceFileDirectory + '/testData/test_steering.csv'
    print(source_file2)
    subDir =[]

    # Identify all the subfolders
    for item in os.listdir(rootDir):
        if not item.startswith("."):
            subDir_name = rootDir + "/" + item + "/"
            subDir.append(subDir_name)


    # Copy the files
    for destination_folder in natsorted(subDir):
        destination_folder_with_testData = destination_folder+'testData/'
        createFolder(destination_folder_with_testData)
        # print(destination_folder_with_testData)
        # print(destination_folder)
        shutil.copy(source_file1,destination_folder)
        shutil.copy(source_file2,destination_folder_with_testData)

def printFolderInfo(rootDir):  # Logic to print the list of files for a given folder
    for item in natsorted(os.listdir(rootDir)):
        if not item.startswith("."):
            subDir_name = rootDir + "/" + item + "/"
            print(item)
            print(subDir_name)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,help="path for input directory")
    #parser.add_argument('--source', type=str, help="path for source file directory")
    args, unknown = parser.parse_known_args()
    #print(args.input)
    #copyFilesAndFolder(args.input, args.source)
    #printFolderInfo(args.input)