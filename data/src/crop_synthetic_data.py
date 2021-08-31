import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import cv2
import tqdm
from hutils.fileio import createDir, readEXR

# step 1: load camera matrix and LED position
# load camera K from ini


target_width = 256
is_square = True
run_undist = False
run_crop = False
run_mask_generation = True
run_light_selection = True
run_save_config = True

# step 1: undist image observation
img_dir = r'F:\Project\SIREN\siren\data\output_dir_near_light\09_reading\perspective\lambertian\scale_512_512\wo_castshadow\shading'

ini_file = os.path.join(img_dir, 'filenames.ini')
import configparser
config = configparser.ConfigParser()
config.optionxform = str
config.read(ini_file)
camera_K = np.fromstring(config['info']['camera_intrinsic'][2:-1], dtype=float, sep=',').reshape(3, 3)

save_folder_crop_raw = os.path.join(img_dir, 'crop_to_size_{}'.format(target_width))
createDir(save_folder_crop_raw)
createDir(os.path.join(save_folder_crop_raw, 'render_img'))
createDir(os.path.join(save_folder_crop_raw, 'render_para'))

img_paths = glob.glob(os.path.join(img_dir, '*.exr'))


if run_crop:
    # now crop the image to get a square shape
    png_paths = glob.glob(os.path.join(img_dir, '*.png'))
    img = cv2.imread(png_paths[0], cv2.IMREAD_UNCHANGED)
    h, w, _ = img.shape
    from hutils.gui.rect_select import rect_drawer
    scale = 1
    img_r = cv2.resize(img, (int(w/scale), int(h/scale)))
    myrect = rect_drawer(img_r)
    myrect.build_mouse()
    myrect.finish()
    rec = np.array(myrect.rect) * scale

    np.save(os.path.join(img_dir, 'origin_crop_rec.npy'), np.array(rec))

    resize_ratio = target_width / rec[2]
    rec_resize = (resize_ratio * np.array(rec)).astype(np.int)
    resize_h, resize_w = int(h * resize_ratio), int(w * resize_ratio)

    npy_paths = glob.glob(os.path.join(img_dir, '*.exr'))
    # new begin to resize and crop the images
    for path in npy_paths:
        print(path)
        img = readEXR(path)
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

    img_set = []
    img_paths = glob.glob(os.path.join(save_folder_crop_raw, '*_crop_raw.npy'))
    for img_path in img_paths:
        img = np.load(img_path)
        img_set.append(img)
    img_set = np.array(img_set)
    LED_set = np.load(os.path.join(img_dir, 'render_para/LED_locs.npy'))

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
        "focal_len": "50", "fx": "{}".format(camera_K_crop_raw[0, 0]), "fy": "{}".format(camera_K_crop_raw[1, 1]),
        "cx": "{}".format(crop_rec_center[0] - camera_K_crop_raw[0, 2]), "cy": "{}".format(crop_rec_center[1] - camera_K_crop_raw[1, 2]),
        "img_h": "{}".format(img_set.shape[1]), "img_w": "{}".format(img_set.shape[2]),
        "numImg": "{}".format(img_set.shape[0])}

    config.add_section('configInfo')
    for key in configinfo.keys():
        config.set('configInfo', key, configinfo[key])

    with open(save_path, 'w') as configfile:  # save
        config.write(configfile)







