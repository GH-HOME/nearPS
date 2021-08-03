import numpy as np
import os
from matplotlib import pyplot as plt
from hutils.fileio import createDir

base_dir = r'F:\Project\blender_2.8_rendering\tool\nearlight\output_dir_near_light\04_bunny\orthographic\lambertian\scale_1024_1024\wo_castshadow\shading'
out_dir = './bunny_ear/orthographic/lambertian/scale_256_256\wo_castshadow\shading'

custom_mask = os.path.join(base_dir, 'render_para/mask.npy')
custom_image = os.path.join(base_dir, 'render_img/imgs.npy')
custom_LEDs = os.path.join(base_dir, 'render_para/LED_locs.npy')
custom_depth = os.path.join(base_dir, 'render_para/depth.npy')
custom_normal = os.path.join(base_dir, 'render_para/normal_world.npy')


N_gt = np.load(custom_normal)
mask = np.load(custom_mask)
img = np.load(custom_image)
depth = np.load(custom_depth)
LED_loc = np.load(custom_LEDs)


mask_c = mask[40:40+256, 40:40+256]
img_c = img[:, 40:40+256, 40:40+256]
N_gt_c = N_gt[40:40+256, 40:40+256]
depth_c = depth[40:40+256, 40:40+256]


createDir(out_dir)
createDir(os.path.join(out_dir, 'render_img'))
createDir(os.path.join(out_dir, 'render_para'))

np.save(os.path.join(out_dir, 'render_img/imgs.npy'), img_c)
np.save(os.path.join(out_dir, 'render_para/mask.npy'), mask_c)
np.save(os.path.join(out_dir, 'render_para/depth.npy'), depth_c)
np.save(os.path.join(out_dir, 'render_para/normal_world.npy'), N_gt_c)
np.save(os.path.join(out_dir, 'render_para/LED_locs.npy'), LED_loc)