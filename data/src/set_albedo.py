#%%

import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from hutils.draw_3D import generate_mesh
import glob
import cv2


def set_albedo(data_dir, albedo_folder):
    imgs = np.load(os.path.join(data_dir, 'render_img/imgs_blender.npy'))
    mask = np.load(os.path.join(data_dir, 'render_para/mask.npy'))


    textures_path = glob.glob(os.path.join(albedo_folder, '*.jpg'))

    shading = imgs[40]
    plt.imshow(shading)
    plt.show()

    h, w = mask.shape

    fig, axes = plt.subplots(4, int(len(textures_path) / 4) + 1)
    axes = axes.ravel()

    scale_set = np.zeros([len(textures_path), 3])
    albedo_set = []
    for i, tex_path in enumerate(textures_path):
        albedo = cv2.imread(tex_path)[:, :, ::-1]

        albedo = cv2.resize(albedo, (h, w)).transpose([1, 0, 2])
        # albedo = albedo[:h, :w]
        img = shading * albedo
        scale_set[i] = 1.0 / np.max(img.reshape(-1, 3), axis=0)

        albedo_set.append(scale_set[i][np.newaxis, np.newaxis, :] * albedo)
        img = scale_set[i][np.newaxis, np.newaxis, :] * img
        axes[i].imshow(img)
        axes[i].set_title('{}: {}'.format(i, os.path.basename(tex_path)))
        axes[i].axis('off')

    plt.show()

    # choose texture
    choice = -1
    try:
        choice = int(input('Input texture label:'))
    except ValueError:
        print("Not a number")

    fig, axes = plt.subplots(1, 2)
    axes = axes.ravel()
    if choice != -1:
        chosen_albedo = albedo_set[choice]
        axes[0].imshow(chosen_albedo)
        axes[1].imshow(chosen_albedo * shading)
        plt.show()

        # save_albedo
        albedo_path = os.path.join(data_dir, 'render_para/albedo.npy')
        np.save(albedo_path, chosen_albedo)


# Task: eval whether the result from the perspective case is correct

# step 1: load data
data_dir = r'F:\Project\SIREN\siren\data\output_dir_near_light\09_reading\perspective\lambertian\scale_256_256\wo_castshadow\shading'
albedo_folder = r'F:\Project\SIREN\siren\data\texture'
set_albedo(data_dir, albedo_folder)

