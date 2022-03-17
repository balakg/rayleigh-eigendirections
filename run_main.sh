#!/bin/bash

gpu="$1 $2"

# Constant args for all experiments
seed_file='seeds/biggan_sample_seeds.npz'
step=1
n_eig=6
n_step=2
beta_c=0.999
syn_alg="biggan"
im_size=128

f="r_all_hi_2"
c="r_all_lo_2"
#f="id r_hair_seg"
#c="l_all"
dir_alg="reds"
path_alg="local"
seed=1

beta_f_arr="$3"
for beta_f in ${beta_f_arr[@]};
do

n_f=$(echo -n $f | wc -w)
if [[ "$n_f" -eq 2 ]]; then
beta_f="$beta_f $beta_f"
fi

#for seed in {0,1}; #,2,4,7,9,10,19,20,21,25,27,35,36,38,40,42,43,46,47,48,49,51,52,55,57,58,60,61,63,64};
#do
#echo "$beta_f $seed"

python main.py \
--gpu ${gpu} \
--seed_file ${seed_file} \
--step ${step} \
--f ${f} \
--c ${c} \
--beta_f ${beta_f} \
--beta_c ${beta_c} \
--n_eig ${n_eig} \
--n_step ${n_step} \
--dir_alg ${dir_alg} \
--path_alg ${path_alg} \
--syn_alg ${syn_alg} \
--seed ${seed} \
--im_size ${im_size}
done
