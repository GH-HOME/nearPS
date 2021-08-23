import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import cv2
import tqdm
from hutils.fileio import createDir

# step 1: load camera matrix and LED position
para_folder = r'G:\Dropbox\realdata_NLPS\near_light_calib\20210815'
camera_para = np.load(os.path.join(para_folder, 'params_camera_undist.npz'))
camera_K = camera_para['intrinsic']
mapx, mapy = camera_para['mapx'], camera_para['mapy']
LED_position = np.load(os.path.join(para_folder, 'params_light.npz'))['led_pt']
target_width = 256
is_square = True
run_undist = False
run_crop = False
run_mask_generation = True
run_light_selection = True
run_save_config = True

# step 1: undist image observation
img_dir = '/mnt/workspace2020/heng/project/data/real_data/FLIR/2021_08_16_13_37_gray_traveler/render_img'
save_folder_crop_raw = os.path.join(img_dir, 'crop_to_size_{}'.format(target_width))
createDir(save_folder_crop_raw)
createDir(os.path.join(save_folder_crop_raw, 'render_img'))
createDir(os.path.join(save_folder_crop_raw, 'render_para'))

img_paths = glob.glob(os.path.join(img_dir, '*.npy'))

if run_undist:
    for i in tqdm.trange(len(img_paths)):
        img = np.load(img_paths[i])
        print(img_paths[i])
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        np.save(img_paths[i][:-4]+'_undist.npy', dst)
        cv2.imwrite(img_paths[i][:-4] + '_undist.png', (dst / dst.max() * 255).astype(np.uint8))

if run_crop:
    # now crop the image to get a square shape
    png_paths = glob.glob(os.path.join(img_dir, '*undist.png'))
    img = cv2.imread(png_paths[0], cv2.IMREAD_UNCHANGED)
    h, w = img.shape
    from hutils.gui.rect_select import rect_drawer
    myrect = rect_drawer(img)
    myrect.build_mouse()
    myrect.finish()
    rec = myrect.rect

    np.save(os.path.join(img_dir, 'origin_crop_rec.npy'), np.array(rec))

    resize_ratio = target_width / rec[2]
    rec_resize = (resize_ratio * np.array(rec)).astype(np.int)
    resize_h, resize_w = int(h * resize_ratio), int(w * resize_ratio)

    npy_paths = glob.glob(os.path.join(img_dir, '*undist.npy'))
    # new begin to resize and crop the images
    for path in npy_paths:
        print(path)
        img = np.load(path)
        img_resize  = cv2.resize(img, (resize_w, resize_h), cv2.INTER_NEAREST)
        img_crop = img_resize[rec_resize[1]:rec_resize[1] + rec_resize[3], rec_resize[0]:rec_resize[0] + rec_resize[2]]

        file_name = os.path.basename(path)
        file_name = file_name[:-4] + '_crop_raw.npy'
        save_path = os.path.join(save_folder_crop_raw, file_name)
        np.save(save_path, img_crop)
        cv2.imwrite(save_path[:-4] + '.png', (img_crop / img_crop.max() * 255).astype(np.uint8))


    # now generate the camera intrinsic for this crop
    camera_K_crop_raw = camera_K * resize_ratio
    camera_K_crop_raw[2, 2] = 1
    np.save(os.path.join(save_folder_crop_raw, 'render_para/camera_K.npy'), camera_K_crop_raw)
    np.save(os.path.join(save_folder_crop_raw, 'render_para/crop_rec.npy'), np.array(rec_resize))


if run_mask_generation:
    mask_img = cv2.imread(os.path.join(save_folder_crop_raw, 'mask_img.png'), cv2.IMREAD_UNCHANGED)
    mask = mask_img[:,:, -1] > 0
    np.save(os.path.join(save_folder_crop_raw, 'render_para/mask.npy'), mask)

if run_light_selection:

    light_44 = np.arange(98, 112).tolist() + np.arange(114, 131, 2).tolist() + np.arange(131, 144).tolist() + np.arange(241, 256, 2).tolist()
    light_44_label = np.array(light_44)
    light_44_index = np.array(light_44) - 1 # LED label to LED index
    LED_set = LED_position[light_44_index]
    LED_set[:, 0] *= (-1)
    LED_set[:, 1] *= (-1)
    LED_set = LED_set * 1e-3  # mm to m

    img_set = []
    for i in light_44_label:
        img_path = glob.glob(os.path.join(save_folder_crop_raw, 'right_FLIR*_{}_undist_crop_raw.npy').format(i))[0]
        img = np.load(img_path)
        img_set.append(img)
    img_set = np.array(img_set) / 65535.0

    np.save(os.path.join(save_folder_crop_raw, 'render_img/imgs_real_44.npy'), img_set)
    np.save(os.path.join(save_folder_crop_raw, 'render_para/LED_locs.npy'), LED_set)


if run_save_config:
    import configparser
    config = configparser.ConfigParser()
    save_path = os.path.join(save_folder_crop_raw, 'render_para/save.ini')
    camera_K_crop_raw = np.load(os.path.join(save_folder_crop_raw, 'render_para/camera_K.npy'))
    resize_crop_rec = np.load(os.path.join(save_folder_crop_raw, 'render_para/crop_rec.npy'))
    crop_rec_center = np.array([resize_crop_rec[0] + resize_crop_rec[2] / 2, resize_crop_rec[1] + resize_crop_rec[3] / 2])

    configinfo = {
            "focal_len": "16", "fx": "{}".format(camera_K_crop_raw[0, 0]), "fy": "{}".format(camera_K_crop_raw[1, 1]),
            "cx": "{}".format(crop_rec_center[0] - camera_K_crop_raw[0, 2]), "cy": "{}".format(crop_rec_center[1] - camera_K_crop_raw[1, 2]),
            "img_h": "{}".format(img_set.shape[1]), "img_w": "{}".format(img_set.shape[2]),
            "numImg": "{}".format(img_set.shape[0])}

    config.add_section('configInfo')
    for key in configinfo.keys():
        config.set('configInfo', key, configinfo[key])

    with open(save_path, 'w') as configfile:  # save
        config.write(configfile)







