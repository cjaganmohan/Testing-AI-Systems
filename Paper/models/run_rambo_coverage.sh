#! /bin/sh

slash="/"
#3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
for group_number in 2 3 4 5 6; do
  #group_number=7
  echo $group_number
  group='Grp'
  print $group$group_number
  #output_destination='/home/jagan/Dropbox/Self-driving-car-Results/Rambo/Results/Coverage/Baseline/Grp'$group_number # to run test dataset
  #output_destination='/home/jagan/Dropbox/Self-driving-car-Results/Rambo/Results/Cumulative_Coverage/single_transformation_deepTest/Grp'$group_number
  output_destination='/home/jagan/Dropbox/Self-driving-car-Results/Rambo/Results/Cumulative_Coverage/2-way_new/'

  d='/home/jagan/Desktop/Rambo/2-way/Grp'$group_number
  file_name='single_transformation_deepTest'
  echo $d
  echo $group_number
  echo $file_name # transformation_name
  echo $output_destination
  echo '*********'
  export KERAS_BACKEND=theano # changing the backend to Theano
  #python2 run_rambo_coverage_working_version.py --dataset $d --group $group_number --file_type $file_name --output_path $output_destination
  python2 run_rambo_cumulative_coverage_test_suite.py --dataset $d --group $group_number --file_type $file_name --output_path $output_destination


#
#  for d in $(find '/home/jagan/Desktop/Rambo/Single_Transformation/Grp'$group_number$slash -maxdepth 1 -type d | natsort); do
#    file_name=$(basename $d) # to print the folder name --- https://stackoverflow.com/a/3362952
#    #directory_name=$d
#    if [ $file_name != $group$group_number ]; then
#      echo $d
#      echo $group_number
#      echo $file_name # transformation_name
#      echo $output_destination
#      echo '*********'
#      export KERAS_BACKEND=theano # changing the backend to Theano
#      #python2 run_rambo_coverage_working_version.py --dataset $d --group $group_number --file_type $file_name --output_path $output_destination
#      python2 run_rambo_cumulative_coverage_test_suite.py --dataset '/home/jagan/Desktop/Rambo/Single_Transformation/Grp'$group_number --group $group_number --file_type $file_name --output_path $output_destination
#    fi
#  done

  #  ##d='/home/jagan/Desktop/Baseline/center/'  # to run test dataset
  #  d='/home/jagan/Desktop/Rambo/Baseline/Grp'$group_number$slash
  #  file_name='baseline'
  #  export KERAS_BACKEND=theano # changing the backend to Theano
  #  python2 run_rambo_coverage_working_version.py --dataset $d --group $group_number --file_type $file_name --output_path $output_destination

  group_number=$((group_number + 1))

done

#group_number=8
#group='Grp'
#print $group$group_number
##output_destination='/home/jagan/Dropbox/Self-driving-car-Results/Rambo/Results/TestDataset' # to run test dataset
#output_destination='/home/jagan/Dropbox/Self-driving-car-Results/Rambo/Results/Single_Transformation/Grp'$group_number
##nvidia-smi
#for d in $(find '/home/jagan/Desktop/Rambo/Single_Transformation/Grp'$group_number$slash -maxdepth 1 -type d | natsort)
#
#do
#  file_name=$(basename $d)  # to print the folder name --- https://stackoverflow.com/a/3362952
#  #directory_name=$d
#  if [ $file_name !=  $group$group_number ]
#  then
#    echo $d
#    echo $group_number
#    echo $file_name  # transformation_name
#    echo $output_destination
#    echo '*********'
#    export KERAS_BACKEND=theano # changing the backend to Theano
#    python2 run_rambo.py --dataset $d --group $group_number --file_type $file_name --output_path $output_destination
#  fi
#done

##d='/home/jagan/Desktop/Baseline/center/'  # to run test dataset
#d='/home/jagan/Desktop/Rambo/Baseline/Grp'$group_number$slash
#file_name='baseline'
#export KERAS_BACKEND=theano # changing the backend to Theano
#python2 run_rambo.py --dataset $d --group $group_number --file_type $file_name --output_path $output_destination
