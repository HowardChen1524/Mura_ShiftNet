#!/bin/bash
declare n_clusters=(2 3 4 5 6 7 8 9 10)

for n_cluster in ${n_clusters[@]}
do
    normal_dir="/home/sallylab/min/d23_merge/test/test_normal_4k"
    smura_dir="/home/sallylab/min/typed_shifted_${n_cluster}/img"
    save_dir="./"

    python gen_pixel_mean.py \
    -nd=$normal_dir \
    -md=$smura_dir \
    -sd=$save_dir \
    -nc=$n_cluster
done
