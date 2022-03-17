#!/bin/bash

gpu="0 1"

base_dir="/data/vision/billf/scratch/balakg/reds/results/"
save_funcs="id l_all r_hair_seg"

save_dir1="${base_dir}reds_local/l_all,id,r_hair_seg,0.9000,0.9000"
save_dir2="${base_dir}reds_local/l_all,id,r_hair_seg,0.9500,0.9500"
save_dir3="${base_dir}reds_local/l_all,id,r_hair_seg,0.9900,0.9900"
save_dir4="${base_dir}reds_local/l_all,id,r_hair_seg,0.9990,0.9990"

d="${save_dir1} ${save_dir2} ${save_dir3} ${save_dir4}"

for save_dir in ${d[@]};
do
echo ${save_dir}

python save_distances.py \
--gpu ${gpu} \
--save_dir ${save_dir} \
--save_funcs ${save_funcs}
done

#python save_distances.py \
#--gpu ${gpu} \
#--save_dir ${save_dir2} \
#--save_funcs ${save_funcs}

#python save_distances.py \
#--gpu ${gpu} \
#--save_dir ${save_dir3} \
#--save_funcs ${save_funcs}

#python save_distances.py \
#--gpu ${gpu} \
#--save_dir ${save_dir4} \
#--save_funcs ${save_funcs}
