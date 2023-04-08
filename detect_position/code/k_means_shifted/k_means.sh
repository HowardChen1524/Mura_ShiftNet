#!/bin/bash
declare n_clusters=(3)

for n_cluster in ${n_clusters[@]}
do
    normal_dir="/home/sallylab/min/d23_merge/test/test_normal_4k"
    smura_dir="/home/sallylab/min/typed/img"
    save_dir="/home/sallylab/min/typed_shifted_${n_cluster}/img"

    python k_means.py \
    -nd=$normal_dir \
    -md=$smura_dir \
    -sd=$save_dir \
    -nc=$n_cluster
done
