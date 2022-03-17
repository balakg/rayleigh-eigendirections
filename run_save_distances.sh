#!/bin/bash

gpu="0 1"

save_funcs="id r_all_lo_2 r_all_hi_2" #Features we want to compute distances for.
results_dir="./results/"
dir="${results_dir}stylegan2/reds_local/r_all_lo_2,r_all_hi_2,0.9900000" # Path to latent vectors for experiment.
seed=0 # would be better to loop over all seeds. Need to change this.

python save_distances.py \
--gpu ${gpu} \
--save_dir ${dir} \
--save_funcs ${save_funcs} \
--seed ${seed}
