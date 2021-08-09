#%%

import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from hutils.draw_3D import generate_mesh
import glob
import cv2

#%%

# Task: eval whether the result from the perspective case is correct

# step 1: load data
data_dir = r'F:\Project\SIREN\siren\data\output_dir_near_light\09_reading\perspective\lambertian\scale_512_512\wo_castshadow\shading'
imgs = np.load(os.path.join(data_dir, 'render_img/imgs_blender.npy'))
mask = np.load(os.path.join(data_dir, 'render_para/mask.npy'))

albedo_folder = r'F:\Project\SIREN\siren\data\texture'
textures_path = glob.glob(os.path.join(albedo_folder, '*.jpg'))

shading = imgs[40]
plt.imshow(shading)
plt.show()

h, w = mask.shape

fig, axes = plt.subplots(4, int(len(textures_path)/4) + 1)
axes = axes.ravel()

for i, tex_path in enumerate(textures_path):
    albedo = cv2.imread(tex_path)[:,:,::-1]

    # albedo = cv2.resize(albedo, (h, w))
    albedo = albedo[:h, :w]
    img = shading * albedo
    for j in range(3):
        img[:,:,j] = img[:,:,j] / img[:,:,j].max()
    axes[i].imshow(img)

plt.show()

