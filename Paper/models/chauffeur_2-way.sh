#!/bin/bash
#mkdir resultReport

img_1="1479425441182877835"
img_2="1479425441232704425"
img_3="1479425441282730750"
underscore="_"
extension=".jpg"

for i in `seq 1 184`
do
  echo $i

  source_image_1=$img_1$underscore$i$extension
  source_image_2=$img_2$underscore$i$extension
  source_image_3=$img_3$underscore$i$extension

  destination_image_1=$img_1$extension
  destination_image_2=$img_2$extension
  destination_image_3=$img_3$extension

  echo $source_image_1, $destination_image_1
  echo $source_image_2, $destination_image_2
  echo $source_image_3, $destination_image_3


  cp /Users/Jagan/Desktop/chauffer-deubgging/first-10-images/transformed-images-first-10/image1/$source_image_1 /Users/Jagan/Desktop/chauffer-deubgging/first-10-images/center/$destination_image_1
  cp /Users/Jagan/Desktop/chauffer-deubgging/first-10-images/transformed-images-first-10/image2/$source_image_2 /Users/Jagan/Desktop/chauffer-deubgging/first-10-images/center/$destination_image_2
  cp /Users/Jagan/Desktop/chauffer-deubgging/first-10-images/transformed-images-first-10/image3/$source_image_3 /Users/Jagan/Desktop/chauffer-deubgging/first-10-images/center/$destination_image_3

  echo "File "$source_image_1"  moved  to " $destination_image_1
  echo "File "$source_image_2"  moved  to " $destination_image_2
  echo "File "$source_image_3"  moved  to " $destination_image_3


  python2 chauffeur_reproduce.py --dataset /Users/Jagan/Desktop/chauffer-deubgging/first-10-images > "/Users/Jagan/Desktop/chauffer-deubgging/first-10-images/transformed-images-first-10/TC-"$i".txt"

done

#echo "File "$source_image_1"  moved  to " $destination_image_1
#python2 chauffeur_reproduce.py --dataset /Users/Jagan/Desktop/chauffer-deubgging/first-10-images > /Users/Jagan/Desktop/trial-output.txt