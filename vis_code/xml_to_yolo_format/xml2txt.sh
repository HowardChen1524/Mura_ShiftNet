#!/bin/bash
xml_dir='/home/sallylab/min/typec+b1/xml/'
save_dir='/home/sallylab/min/typec+b1_yolo/labels/train'

python xml2txt.py \
-xd=$xml_dir \
-sd=$save_dir