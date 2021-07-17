import numpy as np
from matplotlib import pyplot as plt
import os
from hutils.visualization import scatter_3d, save_normal_no_margin, plt_error_map_cv2
from hutils.PhotometricStereoUtil import evalsurfaceNormal
import cv2
import matplotlib.ticker as ticker
from hutils.fileio import createDir
import matplotlib.animation as animation

def create_gif(imgs, save_path):
    ims = []
    fig = plt.figure()
    for i in range(len(imgs)):
            im = plt.imshow(imgs[i], animated=True, cmap = plt.cm.gray)
            plt.axis('off')
            ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    ani.save(save_path, writer='pillow', dpi=300)

def fmt(x, pos):
    a, b = '{:.3e}'.format(x).split('e')
    b = int(b)
    return r'${}$'.format(a)

def generate_poly_surface_unit_coord(coe, radius):

    patch_size = 2 * radius + 1
    yy, xx = np.meshgrid(np.linspace(-1, 1, patch_size), np.linspace(-1, 1, patch_size))
    yy = np.flip(yy, axis=0)
    zz = coe[0] * xx * xx + coe[1] * yy * yy + coe[2] *xx * yy + coe[3] * xx + coe[4] * yy
    dx = 2 * coe[0] * xx + coe[2] * yy + coe[3]
    dy = 2 * coe[1] * yy + coe[2] * xx + coe[4]


    sphere_radius = 1.5
    zz = np.sqrt((sphere_radius) ** 2 - xx ** 2 - yy ** 2)
    zz = zz - zz.mean()
    dx = -xx * np.power((sphere_radius) ** 2 - xx ** 2 - yy ** 2, -0.5)
    dy = -yy * np.power((sphere_radius) ** 2 - xx ** 2 - yy ** 2, -0.5)



    point_cloud = np.array([xx, yy, zz]).transpose([1, 2, 0])

    scatter_3d(point_cloud.reshape(-1, 3))

    dz = np.ones_like(dx) * (-1)

    Normal_ana = np.array([dx, dy, dz]).transpose([1, 2, 0])
    Normal_ana = -Normal_ana / np.linalg.norm(Normal_ana, axis=2, keepdims=True)

    plt.imshow(Normal_ana/2 + 0.5)
    plt.title('Normal map')
    plt.show()

    return Normal_ana, point_cloud



def render_one_LED(Normal_ana, point_cloud, LED_loc, attach_shadow = True):

    h, w, _ = Normal_ana.shape
    img = np.zeros([h, w])

    for i in range(h):
        for j in range(w):

            x_3d = point_cloud[i ,j]
            n = Normal_ana[i, j]
            light_falloff = 1.0 / np.square(np.linalg.norm(x_3d - LED_loc))
            light_dir = (LED_loc - x_3d) / np.linalg.norm(x_3d - LED_loc)
            pix = light_falloff * np.dot(n, light_dir) * 1e5
            if attach_shadow:
                pix = np.maximum(pix, 0.0)

            img[i, j] = pix

    return img


def generate_LEDs(radius, numx, numy, z):

    LEDxy = np.meshgrid(np.arange(-numx, numx+1), np.arange(-numy, numy+1))
    LEDz = np.ones([2 * numx + 1, 2 * numx + 1, 1]) * z
    LEDxy = radius * np.array(LEDxy).transpose([1, 2, 0])
    LED_array = np.dstack([LEDxy, LEDz])
    return LED_array


if __name__ == '__main__':

    coe = np.random.random(6)
    # coe[3:] = 0

    # coe = np.array([1, 2,  3, -5, -8])
    # [0.00234301 0.00297627 0.0057204  0.00752655 0.00024565]
    # coe = np.array([0, 0, 1, 0, 0])
    print(coe)
    radius = 64
    N_gt, point_cloud = generate_poly_surface_unit_coord(coe, radius)

    LEDs = generate_LEDs(25, 1, 1, 250)
    img_set = []
    LEDs = LEDs.reshape(-1, 3)
    for LED_loc in LEDs:
        img = render_one_LED(N_gt, point_cloud, LED_loc, attach_shadow=True)
        img_set.append(img)

    out_dir = './poly2d'
    createDir(out_dir)
    np.save(os.path.join(out_dir, 'normal.npy'), N_gt)
    np.save(os.path.join(out_dir, 'point_cloud.npy'), point_cloud)
    np.save(os.path.join(out_dir, 'depth.npy'), point_cloud[:, :, 2])
    np.save(os.path.join(out_dir, 'LEDs.npy'), LEDs)
    np.save(os.path.join(out_dir, 'img_set.npy'), np.array(img_set))
    create_gif(img_set, os.path.join(out_dir, 'show.gif'))