from __future__ import print_function

import csv
import os
import shutil

input_file = 'CH2_final_evaluation.csv'
list_of_files =[]
file_name =[]
output_file_name = 'final_evaluation.csv'
start_index=0
end_index=0

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error in creating the directory  ' + directory)



def run_prediction_for_smaller_dataset(image):
    global start_index
    global end_index
    with open(input_file, 'rb') as input_csv_file:
        readCSV = csv.reader(input_csv_file, delimiter=',', quotechar='|')
        next(readCSV) # to skip the header information
        for row in readCSV:
            list_of_files.append(row)
            file_name.append(row[0])

    file_index = file_name.index(image)
    print(file_index)

    start_index = file_index-4 # we are considering the previous (99 frames for chauffeur, 2 frames for rambo)
    end_index = file_index+1 # we are incrementing the file_index to include the candidate image as well (Check how range() works in python for further explanation)

    for item in list_of_files[start_index:end_index]:
        print(item[:2])
    return list_of_files

def create_consolidated_datasets():
    # candidate_image_list = ['1479425660620933516', '1479425534498905778',
    #                         '1479425619063583545', '1479425660020827157',
    #                         '1479425535099058605', '1479425496442340584',
    #                         '1479425537999541591', '1479425719031130839',
    #                         '1479425712029955155', '1479425706078866287',
    #                         '1479425527947728896', '1479425468287290727',
    #                         '1479425470287629689', '1479425499292775434',
    #                         '1479425488540828515', '1479425652219428572',
    #                         '1479425654520220380', '1479425654069742649',
    #                         '1479425653569688917']
    candidate_image_list = ['1479425535099058605']

    global start_index
    global end_index
    counter = 6;
    for image in candidate_image_list:
        list_of_files = run_prediction_for_smaller_dataset(image)

        # for item in list_of_files[start_index:end_index]:
        #     print(item[:2])
        print('Start_index' + str(start_index))
        print('End_index' + str(end_index))
        # Directory creation
        dirName= './Grp' + str(counter) +'_'+ str(start_index) + '_' + str(end_index)+'/center/'
        createFolder(dirName)
        output_file = './Grp' + str(counter) +'_'+ str(start_index) + '_' + str(end_index) + '/' + output_file_name
        print(os.path.dirname(os.path.abspath(output_file)))
        print(output_file)

        # copy files to center folder
        source_file_location = '/Users/Jagan/Desktop/CH2_001/center/'  # source location to copy images
        with open(output_file, 'ab') as output_csv_file:
            writerCSV = csv.writer(output_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writerCSV.writerow(['frame_id','steering_angle','Group'])
            for item in list_of_files[start_index:end_index]:
                writerCSV.writerow([item[0],item[1],counter])
                source_file = source_file_location+item[0]+'.jpg'
                shutil.copy(source_file,dirName)

        output_csv_file.close()
        counter = counter+1
        print(counter)


if __name__ == "__main__":
    create_consolidated_datasets()

