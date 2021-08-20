import numpy as np
from matplotlib import pyplot as plt
import os
from Reference.QY18 import run_QW18


para_folder = '../data/supp'
shape_set = np.genfromtxt(os.path.join(para_folder, 'shape_list.csv'), delimiter=',', dtype=np.str)
scale_set = np.genfromtxt(os.path.join(para_folder, 'scale_list_64_64.csv'), delimiter=',', dtype=np.int)
base_data_dir = r'Y:\workspace2020\heng\project\data\output_dir_near_light'

commit_id = '8e3608bc'

scale_set = [[256, 256]]
shape_set = ['07_JoyfulYell.obj']
depth_set = np.arange(0.5, 5, 0.5)
for resolution in scale_set:
    for shape_name in shape_set:
        for depth_offset in depth_set:
            filename_str = 'perspective/lambertian/scale_{rx}_{ry}/wo_castshadow/shading/'.format(
                rx=int(resolution[0]), ry=int(resolution[1]))
            data_dir = os.path.join(base_data_dir, shape_name.split('.')[0], filename_str)
            output_folder_str = '{}_depth_offset_{}'.format(commit_id, depth_offset)

            run_QW18.process(data_dir, output_folder_str= output_folder_str, sv_albedo = True, depth_offset = depth_offset)
