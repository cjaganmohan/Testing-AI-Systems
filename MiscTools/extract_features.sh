#!/bin/sh

for d in $(find '/Users/Jagan/Desktop/self-driving-car-project/Feature_Extraction_Trials/Single_Transformation/Grp2' -maxdepth 1 type d | natsort)
a
do
  echo $(basename $d)
  echo $d
done