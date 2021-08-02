import numpy as np
from matplotlib import pyplot as plt
import os
from hutils.visualization import scatter_3d, save_normal_no_margin, plt_error_map_cv2, create_gif
from hutils.PhotometricStereoUtil import evalsurfaceNormal
import cv2
import matplotlib.ticker as ticker
from hutils.fileio import createDir
import matplotlib.animation as animation


def fmt(x, pos):
    a, b = '{:.3e}'.format(x).split('e')
    b = int(b)
    return r'${}$'.format(a)

def generate_poly_surface_unit_coord(coe, radius):

    patch_size = 2 * radius + 1
    xx, yy = np.meshgrid(np.linspace(-1, 1, patch_size), np.linspace(-1, 1, patch_size))
    yy = np.flip(yy, axis=0)
    zz = coe[0] * xx * xx + coe[1] * yy * yy + coe[2] *xx * yy + coe[3] * xx + coe[4] * yy + coe[5]
    dx = 2 * coe[0] * xx + coe[2] * yy + coe[3]
    dy = 2 * coe[1] * yy + coe[2] * xx + coe[4]


    sphere_radius = 1.5
    zz = np.sqrt((sphere_radius) ** 2 - xx ** 2 - yy ** 2)
    zz = zz - zz.mean() + coe[5]
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

def generate_Sphere(coe, radius):

    patch_size = 2 * radius + 1
    xx, yy = np.meshgrid(np.linspace(-1, 1, patch_size), np.linspace(-1, 1, patch_size))
    yy = np.flip(yy, axis=0)

    sphere_radius = 0.99
    zz = np.sqrt((sphere_radius) ** 2 - xx ** 2 - yy ** 2)
    mask = ~np.isnan(zz)
    zz[~mask] = 0
    zz = zz + coe[5]
    dx = -xx * np.power((sphere_radius) ** 2 - xx ** 2 - yy ** 2, -0.5)
    dy = -yy * np.power((sphere_radius) ** 2 - xx ** 2 - yy ** 2, -0.5)

    point_cloud = np.array([xx, yy, zz]).transpose([1, 2, 0])

    scatter_3d(point_cloud.reshape(-1, 3))
    dz = np.ones_like(dx) * (-1)

    Normal_ana = np.array([dx, dy, dz]).transpose([1, 2, 0])
    Normal_ana[~mask] = 0
    Normal_ana = -Normal_ana / np.linalg.norm(Normal_ana, axis=2, keepdims=True)
    Normal_ana[~mask] = 0
    plt.imshow(Normal_ana/2 + 0.5)
    plt.title('Normal map')
    plt.show()

    return Normal_ana, point_cloud, mask


def generate_SurfaceTest(radius, offset = 0):
    x = np.linspace(-1, 1, num=radius)
    y = np.linspace(-1, 1, num=radius)
    XX, YY = np.meshgrid(x, y)
    YY = np.flip(YY, axis=0)
    XX = np.flip(XX, axis=1)
    slope = 0.6
    z = np.zeros_like(XX)
    zx = np.zeros_like(XX)
    zy = np.zeros_like(XX)

    mask_top = np.logical_and.reduce((XX>-0.8, XX<0.8, YY>0, YY<0.8))
    zy[mask_top] = slope
    z[mask_top] = -0.8 * slope +slope * YY[mask_top]
    mask_bottom = np.logical_and.reduce((XX>-0.8, XX<0.8, YY<0, YY>-0.8))
    zy[mask_bottom] = -slope
    z[mask_bottom] = -0.8 * slope - slope * YY[mask_bottom]
    z = z + offset

    point_cloud = np.array([XX, YY, z]).transpose([1, 2, 0])
    scatter_3d(point_cloud.reshape(-1, 3))
    n_s = np.concatenate((zx[..., None], zy[..., None], -np.ones_like(zx)[..., None]), axis=-1)
    Normal_ana = n_s / np.linalg.norm(n_s, axis=2, keepdims=True)
    plt.imshow(Normal_ana/2 + 0.5)
    plt.show()

    h, w, _ = Normal_ana.shape
    mask = np.ones([h, w]).astype(np.bool)
    return Normal_ana, point_cloud, mask

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


def generate_LEDs(radius, numx, numy, z):

    LEDxy = np.meshgrid(np.arange(-numx, numx+1), np.arange(-numy, numy+1))
    LEDz = np.ones([2 * numx + 1, 2 * numx + 1, 1]) * z
    LEDxy = radius * np.array(LEDxy).transpose([1, 2, 0])
    LED_array = np.dstack([LEDxy, LEDz])
    return LED_array


if __name__ == '__main__':

    offset = 3
    coe = np.random.random(6) /2
    coe[5] = offset

    print(coe)
    radius = 128

    # N_gt, point_cloud = generate_poly_surface_unit_coord(coe, radius)
    N_gt, point_cloud, mask = generate_SurfaceTest(radius, offset)
    # N_gt, point_cloud, mask = generate_Sphere(coe, radius)
    # N_gt, point_cloud, mask = generate_bowl(coe, radius)

    LEDs = generate_LEDs(0.5, 2, 2, 0)
    # LEDs = generate_LEDs(0.7, 1, 1, 3)
    img_set = []
    LEDs = LEDs.reshape(-1, 3)

    albedo = cv2.imread(r'F:\dataset\albedo\1.jpg', 0)
    albedo = albedo / albedo.max()
    albedo = cv2.resize(albedo, (513, 513))
    albedo = np.ones([128, 128])
    # albedo = np.ones((2*radius+1, 2 * radius + 1))
    # LED_ins_set = np.random.random(len(LEDs)) + 1
    # LED_ins_set = (np.arange(len(LEDs)) + 1) / 50
    LED_ins_set = np.ones(len(LEDs))
    print(LED_ins_set)
    shading_set = []
    for i, LED_loc in enumerate(LEDs):
        shading = render_one_LED(N_gt, point_cloud, LED_loc, attach_shadow=True, mask = mask)
        img = shading * albedo * LED_ins_set[i]
        img_set.append(img)
        shading_set.append(shading)


    data_name = 'Camp'
    base_dir = '../data/output_dir_near_light'
    out_dir = os.path.join(base_dir, '{}/orthographic/lambertian/scale_{}_{}/wo_castshadow/shading'.format(data_name, radius, radius))

    createDir(out_dir)
    createDir(os.path.join(out_dir, 'render_img'))
    createDir(os.path.join(out_dir, 'render_para'))

    N_obj = np.copy(N_gt)
    N_obj[:,:,0] *= (-1)
    N_obj[:, :, 2] *= (-1)
    plt.imshow(N_obj/2 + 0.5)
    plt.show()

    np.save(os.path.join(out_dir, 'render_para/normal_world.npy'), N_gt)
    np.save(os.path.join(out_dir, 'render_para/point_set_world.npy'), point_cloud)
    np.save(os.path.join(out_dir, 'render_para/depth.npy'), point_cloud[:, :, 2])
    np.save(os.path.join(out_dir, 'render_para/LED_locs.npy'), LEDs)
    np.save(os.path.join(out_dir, 'render_para/mask.npy'), np.array(mask))
    np.save(os.path.join(out_dir, 'render_img/imgs.npy'), np.array(img_set))
    create_gif(img_set, os.path.join(out_dir, 'show.gif'), mask, fps = 1)