#!/bin/bash

echo "Bash version ${BASH_VERSION} ..."
source_folder='/home/jagan/Desktop/Rambo/prediction-in-batches/Results/t-way/T-0.3/2-way/'
for i in {2..20..1}
  do
    echo "Grp$i"
    mkdir $source_folder"Grp"$i
  done