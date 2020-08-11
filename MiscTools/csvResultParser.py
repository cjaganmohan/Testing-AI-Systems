import argparse
import csv
import os
from natsort import natsorted

input_rows=[]
#output_file = '/Users/Jagan/Desktop/Autumn_Consolidated_Grp16_2way.csv'
#output_file = '/Users/Jagan/Desktop/Autumn_Consolidated_Threshold_0.1_Grp'+str(groupNumber)+'.csv'
output_file_name = '/Users/Jagan/Desktop/Chauffeur_Grp'
#output_file_name = '/Users/Jagan/Desktop/Rambo_Theano_IndividualTransformation_'


def sort_and_print_file_names(filedir):
    for file in natsorted(os.listdir(filedir)):
        if not file.startswith(".") and file.endswith(".csv"):
            fileName = filedir+'/'+file
            #print(file)

            output_file = output_file_name + filedir.rsplit('/',1)[1]+'.csv'
            #print('*******', output_file)
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
                        #writeCSV.writerow([file, row[0], row[1], row[3], row[4],row[5], row[7], row[8], row[9]])
                        #writeCSV.writerow([file, row[0], row[1], row[3], row[4],row[5]])
                        #writeCSV.writerow([row[0], file, row[2], row[1]])
                        writeCSV.writerow([row[0], file, row[2]])
                        #writeCSV.writerow([row[0], row[1], row[2]])
                    file_counter = file_counter+1
                    # input_rows.append(row)
                    # input_rows.append(file)
                    #print(row)




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,help="path for input directory")
    args, unknown = parser.parse_known_args()
    print(args.input.rsplit('/',1)[1])
    #print(args.input)
    sort_and_print_file_names(args.input)