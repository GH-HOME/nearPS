import os
import numpy as np



para_folder = '/mnt/workspace2020/heng/project/data/supp'
shape_set = np.genfromtxt(os.path.join(para_folder, 'shape_list.csv'), delimiter=',', dtype=np.str)
scale_set = np.genfromtxt(os.path.join(para_folder, 'scale_list_64_64.csv'), delimiter=',', dtype=np.int)
base_data_dir = '/mnt/workspace2020/heng/project/data/output_dir_near_light'
gpu_id = 6
commit_id = '158bb260'
use_SV_albedo = True
scale_set = [[512, 512]]
index = np.array([4, 3])
for resolution in scale_set:
    for shape_name in shape_set[index]:
        filename_str = 'perspective/lambertian/scale_{rx}_{ry}/wo_castshadow/shading/'.format(
            rx=int(resolution[0]), ry=int(resolution[1]))
        data_dir = os.path.join(base_data_dir, shape_name.split('.')[0], filename_str)

        os.system(
            "/usr/local/bin/python ../train_nearPS.py --data_folder {} --gpu_id {} --code_id {} --sv_albedo {}".format(
                data_dir, gpu_id, commit_id, use_SV_albedo))
