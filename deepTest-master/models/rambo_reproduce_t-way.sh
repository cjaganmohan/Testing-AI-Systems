#! /bin/sh

slash="/"
#transformation_name=Original
#directory_name=NA
group_number=2
dataset='/home/jagan/Desktop/Rambo/prediction-in-batches/Grp2_4386_4389/'
#python2 rambo_reproduce_modified_V2.py --transformation $transformation_name --directory $directory_name --group $group_number --dataset $dataset
#for d in $(find '/home/jagan/Desktop/IndividualTransformations/' -maxdepth 1 -type d | natsort)
for d in $(find '/home/jagan/Desktop/Rambo/prediction-in-batches/t-way_test_inputs/2-way/Grp2/' -maxdepth 1 -type f | natsort) # test images
do
  file_name=$(basename $d)  # to print the folder name --- https://stackoverflow.com/a/3362952
#  directory_name=$d$slash  # to print the physical path
  directory_name=$d
  if [ $file_name !=  .DS_Store ]
  then
    #echo $count
    echo $file_name
    echo $directory_name
    echo $group_number
    echo $dataset
    #sudo killall -9 python
    echo $d
    #cp $d $dataset/center/
    python2 rambo_reproduce_modified_V2_tway.py --transformation $file_name --directory $directory_name --group $group_number --dataset $dataset
    #sudo killall -9 python
  fi
done