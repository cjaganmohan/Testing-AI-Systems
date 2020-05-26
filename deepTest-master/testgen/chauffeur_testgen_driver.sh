#!/bin/bash
mkdir result
mv result/chauffeur_rq2_70000_images.csv result/chauffeur_rq2_70000_images.csv.bak
#touch result/chauffeur_rq2_70000_images.csv
touch result/chauffeur_test_dataset_images_coverage.csv

#python chauffeur_testgen_coverage_copy.py --index $1 --dataset $2
#for i in `seq 0 141`;
for i in `seq 0 1`;
do
        echo $i
        python chauffeur_testgen_coverage.py --index $i --dataset $1  # commmented by Jagan
        #python chauffeur_testgen_coverage_copy.py --index $i --dataset $1
        #python chauffeur_testgen_coverage_debugging.py --index $i --dataset $1
done
