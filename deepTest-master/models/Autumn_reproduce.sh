#! /bin/sh

slash="/"
transformation_name=Original
directory_name=NA
group_number=5
dataset='/home/jagan/Desktop/Autumn/prediction-in-batches/Grp5_4372_4377/'
python2 autumn_reproduce_modified_V2.py --transformation $transformation_name --directory $directory_name --group $group_number --dataset $dataset
#for d in $(find '/home/jagan/Desktop/IndividualTransformations/' -maxdepth 1 -type d | natsort)
#do
#  transformation_name=$(basename $d)  # to print the folder name --- https://stackoverflow.com/a/3362952
#  directory_name=$d$slash  # to print the physical path
#  if [ $transformation_name !=  IndividualTransformations ]
#  then
#    #echo $count
#    echo $transformation_name
#    echo $directory_name
#    echo $group_number
#    echo $dataset
#    #sudo killall -9 python
#    python2 autumn_reproduce_modified_V2.py --transformation $transformation_name --directory $directory_name --group $group_number --dataset $dataset
#    #sudo killall -9 python
#  fi
#done