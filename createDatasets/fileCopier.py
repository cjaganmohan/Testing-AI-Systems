from __future__ import print_function
import os
import csv
import shutil

input_file = 'CH2_final_evaluation.csv'
list_of_files =[]
file_name =[]
output_file_name = 'final_evaluation.csv'


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error in creating the directory  ' + directory)


def run_prediction_for_smaller_dataset():
    with open(input_file, 'rb') as input_csv_file:
        readCSV = csv.reader(input_csv_file, delimiter=',', quotechar='|')
        next(readCSV) # to skip the header information
        for row in readCSV:
            list_of_files.append(row)
            file_name.append(row[0])

    file_index = file_name.index('1479425719031130839')
    print(file_index)

    start_index = file_index-99 # we are considering the previous 99 frames
    end_index = file_index+1 # we are incrementing the file_index to include the file as well (Check how range() works in python for further explanation)

    for item in list_of_files[start_index:end_index]:
        print(item[:2])

    #Directory creation
    dirName = './'+str(start_index)+'_'+str(end_index)+'/center/'
    createFolder(dirName)
    output_file = './'+str(start_index)+'_'+str(end_index)+ '/'+ output_file_name
    #print(os.path.dirname(os.path.abspath(output_file)))
    #print(output_file)

    # copy files to center folder
    source_file_location = '/Users/Jagan/Desktop/CH2_001/center/'
    with open(output_file, 'wb') as output_csv_file:
        writerCSV = csv.writer(output_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writerCSV.writerow(['frame_id','steering_angle'])
        for item in list_of_files[start_index:end_index]:
            writerCSV.writerow([item[0],item[1]])
            source_file = source_file_location+item[0]+'.jpg'
            shutil.copy(source_file,dirName)

    output_csv_file.close()

    #copy files to center folder
    source_file_location = '/Users/Jagan/Desktop/CH2_001/center/'

if __name__ == "__main__":
    run_prediction_for_smaller_dataset()

