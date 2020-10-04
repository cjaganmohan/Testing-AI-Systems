#! /bin/sh

slash="/"

group_number=10
dataset='/home/jagan/Desktop/chauffer-deubgging/prediction-in-batches/Grp10_5317_5417/'
for d in $(find '/home/jagan/Desktop/Chauffeur/Grp10' -maxdepth 1 -type d | natsort)
do
  transformation_name=$(basename $d)  # to print the folder name --- https://stackoverflow.com/a/3362952
  directory_name=$d$slash  # to print the physical path
  if [ $transformation_name !=  Grp5 ]
  then
    #echo $count
    echo $transformation_name
    echo $directory_name
    echo $group_number
    echo $dataset
    #sudo killall -9 python
    python chauffeur_reproduce_modified_V3.py --transformation $transformation_name --directory $directory_name --group $group_number --dataset $dataset
    #sudo killall -9 python
  fi
done