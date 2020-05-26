import argparse
import csv
import os
from natsort import natsorted

input_rows=[]
output_file = '/Users/Jagan/Desktop/Consolidated-Grp20.csv'


def sort_and_print_file_names(filedir):
    for file in natsorted(os.listdir(filedir)):
        if not file.startswith(".") and file.endswith(".csv"):
            fileName = filedir+'/'+file
            #print(file)

            with open(fileName, 'rb') as input_file, open(output_file, 'ab') as output_csv_file:
                inputCSV = csv.reader(input_file, delimiter=',', quotechar='|')
                writeCSV = csv.writer(output_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                #writeCSV.writerow(['File-name', 'Transformation-name', 'Predicted','GroundTruth'])  # header information for the output.csv file
                next(inputCSV) #skip the header (row # 1)
                file_counter = 1
                for row in inputCSV:
                    if file_counter == 100:
                        #print(row[0])
                        #print(row[1])
                        writeCSV.writerow([row[0], file, row[2], row[1]])
                    file_counter = file_counter+1
                    # input_rows.append(row)
                    # input_rows.append(file)
                    #print(row)


















                # print(input_rows[99][0])
                # print(input_rows[99][1])
                # print(input_rows[99][2])
                #writeCSV.writerow([input_rows[99][0], file, input_rows[99][2], input_rows[99][1]])



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,help="path for input directory")
    args, unknown = parser.parse_known_args()
    #print(args.input)
    sort_and_print_file_names(args.input)