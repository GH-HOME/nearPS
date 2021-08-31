import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import cv2
import tqdm
from hutils.fileio import createDir, readNormal16bitRGB
from data.real_data.io_nf import readNFSetup, read_gt_depth, load_images

# step 1: load camera matrix and LED position
base_folder = r'C:\project\neuralnlps\data\real_data\Luces\data'
object_name = 'Ball'
para_folder = os.path.join(base_folder, object_name)

scale = 1.0
numLight, ncols, nrows, f, x0, y0, mean_distance, Lpos, Ldir, Phi, mu = readNFSetup(os.path.join(para_folder, 'led_params.txt'), scale)
camera_K = np.array([f, 0, x0, 0, f, y0, 0, 0, 1]).reshape(3, 3)

Lpos[:, 0] *= (-1)
Lpos[:, 1] *= (-1)
Lpos = Lpos * 1e-3  # mm to m

mask = cv2.imread(os.path.join(para_folder, 'mask.png'), 0).astype(np.bool)
mask = cv2.resize(mask.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(np.bool)
N_gt_obj = readNormal16bitRGB(os.path.join(para_folder, 'normals.png'))
N_gt_obj = cv2.resize(N_gt_obj, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
N_gt_obj[~mask] = 0
N_gt_obj[mask] = N_gt_obj[mask] / np.linalg.norm(N_gt_obj[mask], axis=1, keepdims=True)
N_gt_world = np.copy(N_gt_obj)
N_gt_world[:, :, 0] *= (-1)
N_gt_world[:, :, 2] *= (-1)

depth_gt = read_gt_depth(para_folder, scale)
depth_gt = cv2.resize(depth_gt, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
depth_gt = depth_gt * (1e-3)

imgs = load_images(para_folder, nrows, ncols, numLight)
img_set = imgs.transpose([3, 0, 1, 2])

save_dir = os.path.join(para_folder, 'scale_{}'.format(scale))
createDir(os.path.join(save_dir, 'render_img'))
createDir(os.path.join(save_dir, 'render_para'))

# now begin to save
np.save(os.path.join(save_dir, 'render_img/imgs_luces.npy'), img_set)
np.save(os.path.join(save_dir, 'render_para/LED_locs.npy'), Lpos)
np.save(os.path.join(save_dir, 'render_para/camera_K.npy'), camera_K)
np.save(os.path.join(save_dir, 'render_para/normal_obj.npy'), N_gt_obj)
np.save(os.path.join(save_dir, 'render_para/normal_world.npy'), N_gt_world)
np.save(os.path.join(save_dir, 'render_para/depth.npy'), depth_gt)


run_save_config = True

if run_save_config:
    import configparser
    config = configparser.ConfigParser()
    save_path = os.path.join(save_dir, 'render_para/save.ini')
    camera_K_crop_raw = np.load(os.path.join(save_dir, 'render_para/camera_K.npy'))
    crop_rec_center = [ncols / 2, nrows / 2]
    configinfo = {
            "focal_len": "8", "fx": "{}".format(camera_K_crop_raw[0, 0]), "fy": "{}".format(camera_K_crop_raw[1, 1]),
            "cx": "{}".format(crop_rec_center[0] - camera_K_crop_raw[0, 2]),
            "cy": "{}".format(crop_rec_center[1] - camera_K_crop_raw[1, 2]),
            "img_h": "{}".format(img_set.shape[1]), "img_w": "{}".format(img_set.shape[2]),
            "numImg": "{}".format(img_set.shape[0])}

    config.add_section('configInfo')
    for key in configinfo.keys():
        config.set('configInfo', key, configinfo[key])

    with open(save_path, 'w') as configfile:  # save
        config.write(configfile)







