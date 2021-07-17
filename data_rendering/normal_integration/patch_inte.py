import numpy as np
from matplotlib import pyplot as plt
import os
from hutils.visualization import scatter_3d, save_normal_no_margin, plt_error_map_cv2
from hutils.PhotometricStereoUtil import evalsurfaceNormal
import cv2
import matplotlib.ticker as ticker
from hutils.fileio import createDir


def fmt(x, pos):
    a, b = '{:.3e}'.format(x).split('e')
    b = int(b)
    return r'${}$'.format(a)

def generate_poly_surface(coe, radius):
    """
    generate the poly surface
    Parameters
    ----------
    coe : [5 * 1]
    radius: patch radius

    Returns: depth map [radius * radius]
           : surface normal map [radius * radius * 3]
           :
    -------

    """
    patch_size = 2 * radius + 1
    yy, xx = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
    yy -= radius
    xx -= radius

    zz = coe[0] * xx * xx + coe[1] * yy * yy + coe[2] *xx * yy + coe[3] * xx + coe[4] * yy
    dx = 2 * coe[0] * xx + coe[2] * yy + coe[3]
    dy = 2 * coe[1] * yy + coe[2] * xx + coe[4]

    scale = 2.5
    sphere_radius = 50
    zz = np.sqrt((sphere_radius*scale) ** 2 - xx **2 - yy **2)
    zz = zz - zz.mean()
    dx = -xx * np.power((sphere_radius*scale) ** 2 - xx **2 - yy **2, -0.5)
    dy = -yy * np.power((sphere_radius * scale) ** 2 - xx ** 2 - yy ** 2, -0.5)

    point_cloud = np.array([xx+radius, yy+radius, zz]).transpose([1, 2, 0])

    scatter_3d(point_cloud.reshape(-1, 3))

    dz = np.ones_like(dx) * (-1)

    Normal_ana = np.array([dx, dy, dz]).transpose([1, 2, 0])
    Normal_ana = -Normal_ana / np.linalg.norm(Normal_ana, axis=2, keepdims=True)

    plt.imshow(Normal_ana/2 + 0.5)
    plt.title('Normal map')
    plt.show()

    return Normal_ana, point_cloud


if __name__ == '__main__':

    coe = np.arange(1, 6) / 1000
    coe[3:] = 0

    # coe = np.array([1, 2,  3, -5, -8])
    # [0.00234301 0.00297627 0.0057204  0.00752655 0.00024565]
    # coe = np.array([0, 0, 1, 0, 0])
    print(coe)
    radius = 75
    N_gt, point_cloud = generate_poly_surface(coe, radius)


    out_dir = './poly2d'
    createDir(out_dir)
    np.save(os.path.join(out_dir, 'normal.npy'), N_gt)
    np.save(os.path.join(out_dir, 'point_cloud.npy'), point_cloud)
    np.save(os.path.join(out_dir, 'depth.npy'), point_cloud[:,:,2])