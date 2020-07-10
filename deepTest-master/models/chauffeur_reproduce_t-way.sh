#! /bin/sh

slash="/"

group_number=
dataset='/home/jagan/Desktop/chauffer-deubgging/prediction-in-batches/t-way/Grp2/'
#dataset='/Users/Jagan/Desktop/chauffer-deubgging/prediction-in-batches/Results/t-way/Grp2/'
for d in $(find '/home/jagan/Desktop/Combination/Grp6/' -maxdepth 1 -type f | natsort) # test images
#for d in $(find '/Users/Jagan/Desktop/2-way' -maxdepth 1 -type f | natsort)
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
    python chauffeur_reproduce_modified_V2_tway.py --transformation $file_name --directory $directory_name --group $group_number --dataset $dataset
    #sudo killall -9 python
  fi
done