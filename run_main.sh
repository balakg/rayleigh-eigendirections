#!/bin/bash

gpu="0 1"
results_dir="./results"
dir_alg="reds" # Use REDs algorithm
path_alg="local" # Use local version


# Args to recreate example-images/stylegan2-example.jpg
seed_file='seeds/stylegan2_sample_styles.npy' 
seed=1
step=1
n_eig=3
n_step=3
syn_alg="stylegan2"
im_size=256
f="id l_all" # Fix identity and all landmark points
c="r_hair_seg" # Change hair based on segmentation model
beta_f="0.99 0.99" 
beta_c=0.999


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
--im_size ${im_size} \
--results_dir ${results_dir}



# Args to recreate example-images/biggan-example.jpg
seed_file='seeds/biggan_sample_latents.npz' 
seed=1
step=1
n_eig=3
n_step=4
syn_alg="biggan"
im_size=128
f="r_all_hi_2" # Fix high frequencies with sigma = 2 Gaussian blur kernel
c="r_all_lo_2" # Change low frequencies with sigma = 2 Gaussian blur kernel
beta_f=0.99 
beta_c=0.999


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
--im_size ${im_size} \
--results_dir ${results_dir}
