import os
import numpy as np



para_folder = '/mnt/workspace2020/heng/project/data/supp'
shape_set = np.genfromtxt(os.path.join(para_folder, 'shape_list_test.csv'), delimiter=',', dtype=np.str)
scale_set = np.genfromtxt(os.path.join(para_folder, 'scale_list_64_64.csv'), delimiter=',', dtype=np.int)
base_data_dir = '/mnt/workspace2020/heng/project/data/output_dir_near_light'
gpu_id = 4

scale_set = [[256, 256]]
commit_id = '0fb0f354'
for resolution in scale_set:
    for shape_name in shape_set[4:]:
        filename_str = 'orthographic/lambertian/scale_{rx}_{ry}/wo_castshadow/shading/'.format(
            rx=int(resolution[0]), ry=int(resolution[1]))
        data_dir = os.path.join(base_data_dir, shape_name.split('.')[0], filename_str)

        os.system("/root/anaconda3/envs/siren/bin/python ../train_nearPS.py --data_folder {} --gpu_id {} --code_id {}".format(data_dir, gpu_id, commit_id))


