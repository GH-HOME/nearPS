import numpy as np
import os
from matplotlib import pyplot as plt
from hutils.fileio import createDir
from hutils.visualization import scatter_3d

def render_one_LED(Normal_ana, point_cloud, LED_loc, attach_shadow = True, mask = None, cast_shadow = True):

    h, w, _ = Normal_ana.shape
    img = np.zeros([h, w])
    if mask is None:
        mask = np.ones([h, w]).astype(np.bool)

    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                x_3d = point_cloud[i ,j]
                n = Normal_ana[i, j]
                light_falloff = 1.0 / np.square(np.linalg.norm(x_3d - LED_loc))
                light_dir = (LED_loc - x_3d) / np.linalg.norm(x_3d - LED_loc)
                pix = light_falloff * np.dot(n, light_dir)
                if attach_shadow:
                    pix = np.maximum(pix, 0.0)
                if cast_shadow:
                    pass

                img[i, j] = pix

    return img


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
LED_locs = np.load(custom_LEDs)


mask_c = mask[40:40+256, 50:50+256]
N_gt_c = N_gt[40:40+256, 50:50+256]
depth_c = depth[40:40+256, 50:50+256]

kernel = np.ones((3, 3), np.uint8)
import cv2
mask_c = cv2.erode(mask_c.astype(np.uint8), kernel, iterations=1).astype(np.bool)

row, column = mask_c.shape
yy, xx = np.mgrid[:row, :column]
yy = np.flip(yy, axis=0)
xx = np.flip(xx, axis=1)
pixel_coords = np.stack([xx, yy], axis=-1)[None, ...].astype(np.float32)  ## -yy, xx
pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (row - 1)  # xx
pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (column - 1)

pixel_coords -= 0.5
pixel_coords *= 2.
point_cloud = np.dstack([pixel_coords.squeeze(), depth_c])


img_set = []
for i, LED_loc in enumerate(LED_locs):
    shading = render_one_LED(N_gt_c, point_cloud, LED_loc, attach_shadow=True, mask = mask_c)
    img = shading
    img_set.append(img)

img_set = np.array(img_set)

createDir(out_dir)
createDir(os.path.join(out_dir, 'render_img'))
createDir(os.path.join(out_dir, 'render_para'))

np.save(os.path.join(out_dir, 'render_img/imgs.npy'), img_set)
np.save(os.path.join(out_dir, 'render_para/mask.npy'), mask_c)
np.save(os.path.join(out_dir, 'render_para/depth.npy'), depth_c)
np.save(os.path.join(out_dir, 'render_para/normal_world.npy'), N_gt_c)
np.save(os.path.join(out_dir, 'render_para/LED_locs.npy'), LED_locs)