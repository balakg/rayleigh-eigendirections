#!/bin/bash

gpu="$1 $2"

# Args for all StyleGAN2 experiments
seed_file='seeds/W_diverse.npy' #use biggan_sample_seeds.npz for biggan
step=1
n_eig=3
n_step=3
syn_alg="stylegan2"  #biggan
im_size=256   #128 for biggan


f="r_all_hi_2" # Fix high frequencies with sigma = 2 blur kernel
c="r_all_lo_2" # Change low frequencies with sigma = 2 blur kernel
beta_f=0.99 # Beta for fixed attribute.
beta_c=0.999  # Beta for changing attribute.
dir_alg="reds" # Use REDs algorithm
path_alg="local" # Use local version
seed=0 # Seed point

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
