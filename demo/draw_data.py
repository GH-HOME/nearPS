from hutils.visualization import createImgGIF, N_2_N_show, save_normal_no_margin, create_gif
import glob
import cv2
import os
import numpy as np

img_dir = r'F:\Project\SIREN\siren\experiment_scripts\logs\flat_2021_07_18_19_43_33'

normal_path_list = glob.glob(os.path.join(img_dir, 'iter_*_N_est.png'))
shape_path_list = glob.glob(os.path.join(img_dir, 'iter_*_Z_est.png'))
Err_N_path_list = glob.glob(os.path.join(img_dir, 'iter_*_ang_err*.png'))

normal_gif_path = os.path.join(img_dir, '0_normal_anim.gif')
shape_gif_path = os.path.join(img_dir, '0_shape_anim.gif')
Err_gif_path = os.path.join(img_dir, '0_Err_N_anim.gif')

interval = 5
createImgGIF(normal_gif_path, normal_path_list[::interval], fps = 3)
createImgGIF(shape_gif_path, shape_path_list[::interval], fps = 3)
createImgGIF(Err_gif_path, Err_N_path_list[::interval], fps = 3)

data_folder = os.path.join(img_dir, 'data')
normal_gt = np.load(os.path.join(data_folder, 'normal.npy'))
mask = np.ones([normal_gt.shape[0], normal_gt.shape[1]]).astype(np.bool)
save_normal_no_margin(normal_gt, mask, os.path.join(data_folder,'normal.png'))

img_path = os.path.join(data_folder, 'img_set.npy')
img_set = np.load(img_path)
create_gif(img_set, os.path.join(data_folder,'img.gif'), fps = 1)
