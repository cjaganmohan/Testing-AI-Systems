#! /bin/sh

slash="/"
#3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
for group_number in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  #group_number=7
  echo $group_number
  group='Grp'
  print $group$group_number
  #d='/home/jagan/Desktop/Chauffeur/Single_Transformation/Grp'$group_number
  d='/home/jagan/Desktop/Rambo/Single_Transformation_Comparison/Grp'$group_number
  file_name='single_transformation_deepTest'
  baseline='/home/jagan/Desktop/Chauffeur/Baseline/Grp'$group_number
  #output_destination='/home/jagan/Dropbox/Self-driving-car-Results/Rambo/Results/Coverage/Baseline/Grp'$group_number # to run test dataset
  output_destination='/home/jagan/Dropbox/Self-driving-car-Results/Chauffeur/Results/Cumulative_Coverage/single_transformation_deepTest/'

  python2 run_chauffeur_cumulative_coverage.py --dataset $d --baseline $baseline --output_destination $output_destination --group $group_number
done