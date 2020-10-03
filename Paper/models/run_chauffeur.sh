#! /bin/sh

slash="/"

for group_number in 15 16 17 18 19 20
do
  #group_number=14
  group='Grp'
  print $group$group_number
  #output_destination='/home/jagan/Dropbox/Self-driving-car-Results/Chauffeur/Results/TestDataset' # to run test dataset
  output_destination='/home/jagan/Dropbox/Self-driving-car-Results/Chauffeur/Results/2-way/Grp'$group_number

  for d in $(find '/home/jagan/Desktop/Chauffeur/2-way/Grp'$group_number$slash -maxdepth 1 -type d | natsort)

  do
    file_name=$(basename $d)  # to print the folder name --- https://stackoverflow.com/a/3362952
    #directory_name=$d
    if [ $file_name !=  $group$group_number ]
    then
      echo $d
      echo $group_number
      echo $file_name  # transformation_name
      echo $output_destination
      echo '*********'
      python2 run_chauffeur.py --dataset $d --group $group_number --file_type $file_name --output_path $output_destination
    fi
  done

#  #d='/home/jagan/Desktop/Baseline/center/'  # to run test dataset
#  d='/home/jagan/Desktop/Chauffeur/Baseline/Grp'$group_number$slash
#  file_name='baseline'
#  python2 run_chauffeur.py --dataset $d --group $group_number --file_type $file_name --output_path $output_destination

  group_number=$((group_number+1))

done

