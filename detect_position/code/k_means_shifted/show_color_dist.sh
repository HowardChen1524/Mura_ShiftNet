#!/bin/bash
declare n_clusters=(2 3 4 5 6 7 8 9 10)

for n_cluster in ${n_clusters[@]}
do
    python show_color_dist.py \
    -nc=$n_cluster
done
