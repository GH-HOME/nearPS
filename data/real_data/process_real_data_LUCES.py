import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import cv2
import tqdm
from hutils.fileio import createDir, readNormal16bitRGB
from data.real_data.io_nf import readNFSetup, read_gt_depth, load_images

# step 1: load camera matrix and LED position


def organzie_data(base_folder, object_name, scale):
    para_folder = os.path.join(base_folder, object_name)
    numLight, ncols, nrows, f, x0, y0, mean_distance, Lpos, Ldir, Phi, mu = readNFSetup(os.path.join(para_folder, 'led_params.txt'), scale)
    camera_K = np.array([f, 0, x0, 0, f, y0, 0, 0, 1]).reshape(3, 3)

    Lpos[:, 0] *= (-1)
    Lpos[:, 1] *= (-1)
    Lpos = Lpos * 1e-3  # mm to m

    mask = cv2.imread(os.path.join(para_folder, 'mask.png'), 0).astype(np.bool)
    mask = cv2.resize(mask.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(np.bool)
    N_gt_obj, mask_n = readNormal16bitRGB(os.path.join(para_folder, 'normals.png'))
    N_gt_obj = cv2.resize(N_gt_obj, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    N_gt_obj[~mask] = 0
    N_gt_obj[mask] = N_gt_obj[mask] / np.linalg.norm(N_gt_obj[mask], axis=1, keepdims=True)
    N_gt_world = np.copy(N_gt_obj)
    N_gt_world[:, :, 0] *= (-1)
    N_gt_world[:, :, 2] *= (-1)

    depth_gt = read_gt_depth(para_folder, scale)
    depth_gt = depth_gt * (1e-3)


    row, column = mask.shape
    yy, xx = np.mgrid[:row, :column]
    yy = np.flip(yy, axis=0)
    xx = np.flip(xx, axis=1)
    pixel_coords = np.stack([xx, yy], axis=-1).astype(np.float32)  ## -yy, xx
    pixel_coords[:, :, 0] = pixel_coords[:, :, 0] / (column - 1)  # xx
    pixel_coords[:, :, 1] = pixel_coords[:, :, 1] / (row - 1)
    pixel_coords -= 0.5
    pixel_coords *= 2.

    focal_len = 8 #mm
    sensor_w = focal_len * ncols / f *1e-3
    sensor_h = focal_len * nrows / f *1e-3
    pixel_coords[:, :, 0] = pixel_coords[:, :, 0] * (sensor_w / 2) * depth_gt / (focal_len * 1e-3)
    pixel_coords[:, :, 1] = pixel_coords[:, :, 1] * (sensor_h / 2) * depth_gt / (focal_len * 1e-3)
    point_set_world = np.concatenate([pixel_coords, depth_gt[:, :, np.newaxis]], axis=2)

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
    np.save(os.path.join(save_dir, 'render_para/point_set_world.npy'), point_set_world)

    from hutils.draw_3D import generate_mesh
    point_3d_name = os.path.join(save_dir, 'render_para/shape.png')
    generate_mesh(point_set_world, mask, point_3d_name, window_size = (1920, 1080), title='shape')


    run_save_config = True

    if run_save_config:
        import configparser
        config = configparser.ConfigParser()
        save_path = os.path.join(save_dir, 'render_para/save.ini')
        camera_K_crop_raw = np.load(os.path.join(save_dir, 'render_para/camera_K.npy'))
        crop_rec_center = [ncols / 2, nrows / 2]
        configinfo = {
                "focal_len": "{}".format(focal_len), "fx": "{}".format(camera_K_crop_raw[0, 0]), "fy": "{}".format(camera_K_crop_raw[1, 1]),
                "cx": "{}".format(crop_rec_center[0] - camera_K_crop_raw[0, 2]),
                "cy": "{}".format(crop_rec_center[1] - camera_K_crop_raw[1, 2]),
                "img_h": "{}".format(img_set.shape[1]), "img_w": "{}".format(img_set.shape[2]),
                "numImg": "{}".format(img_set.shape[0])}

        config.add_section('configInfo')
        for key in configinfo.keys():
            config.set('configInfo', key, configinfo[key])

        with open(save_path, 'w') as configfile:  # save
            config.write(configfile)


if __name__ == '__main__':
    base_folder = r'F:\Project\SIREN\siren\data\real_data\LUCES\Luces\data'
    file_names = os.listdir(base_folder)
    for name in file_names:
        folder_path = os.path.join(base_folder, name)
        if os.path.isdir(folder_path):
            object_name = name
            for scale in [0,125, 0.25]:
                print(folder_path, scale)
                organzie_data(base_folder, object_name, scale)





