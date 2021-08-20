import numpy as np
import cv2
import glob
import os
from hutils.fileio import create_folder

def cropImgs(data_folder):

    png_paths = glob.glob(os.path.join(data_folder, '*.png'))
    img = cv2.imread(png_paths[0], cv2.IMREAD_UNCHANGED)
    h, w, _ = img.shape
    from hutils.gui.rect_select import rect_drawer
    myrect = rect_drawer(img)
    myrect.build_mouse()
    myrect.finish()
    rec = myrect.rect
    mask_crop = np.zeros([h, w]).astype(np.bool)
    mask_crop[rec[1]:rec[1] + rec[3], rec[0]:rec[0] + rec[2]] = True
    save_folder = os.path.join(data_folder, 'crop')
    create_folder(save_folder)
    for path in png_paths:
        print(path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_crop = img[mask_crop].reshape(rec[3], rec[2], -1)

        file_name = os.path.basename(path)
        file_name = file_name[:-4] + '_crop.png'
        save_path = os.path.join(save_folder, file_name)
        cv2.imwrite(save_path, img_crop)

def change_oritation(folder):
    mask = np.load(os.path.join(folder, 'mask_0.4.npy'))
    file_names = [
        'Ours_SRT4direct_cvxopt_rot_12_N_est.npy',
        'CS16_N_est.npy',
        'OS18SRT2_N_est.npy',
        'Ours_SRT4direct_cvxopt_4_N_est.npy',
        'Ours_SRT4direct_cvxopt_12_N_est.npy',
    ]
    for name in file_names:
        N_est_path = os.path.join(folder, name)
        N_est = np.load(N_est_path)
        N_flat = N_est[mask]
        Nx = np.copy(N_flat[:, 0])
        N_flat[:, 0] = -N_flat[:, 1]
        N_flat[:, 1] = Nx
        N_est[mask] =  N_flat
        from hutils.visualization import N_2_N_show, save_transparent_img

        normal_show = N_2_N_show(N_est, mask, white=True)
        normal_show = np.uint8(normal_show * 255)
        cv2.imwrite(N_est_path[:-3] + 'png', normal_show[:, :, ::-1])
        save_transparent_img(N_est_path[:-3] + 'png', mask)


# change_oritation(r'G:\real_data_MPS\FLIR_OBJ_2021\2021_02_24_14_52_shell\scale_0.4\rotation')
cropImgs(r'Y:\workspace2019\heng\share\lz01\Project\script\para_adjust\release_output_20210715')


