import numpy as np
import os
from matplotlib import pyplot as plt
import tqdm
from hutils.fileio import  createDir
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
font = FontProperties()
FontProperties(fname="C:\\Windows\\Fonts\\times.ttf")
font = {'family' : "Times New Roman",
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

def get_residue(z_range, LED_loc, img_one_pix, uv_coord, normal_map, mask):
    """
    LED_loc: [f,  3]
    """

    h, w = mask.shape
    residue_set = np.zeros([len(z_range), h, w])
    for i, z in enumerate(z_range):

        x = np.array([[uv_coord[0], uv_coord[1], z]])

        lights = LED_loc - x
        light_norm = np.linalg.norm(lights, axis=1)
        light_dir = lights / light_norm[:, np.newaxis]
        light_dir = light_dir / np.square(light_norm)[:, np.newaxis]
        N_dir = normal_map[mask]
        Shading = light_dir @ N_dir.T
        Shading = np.maximum(Shading, 0.0)
        residue = img_one_pix[np.newaxis, :] - Shading.T
        residue_map = np.zeros_like(mask).astype(np.float)
        residue_map[mask] = np.linalg.norm(residue, axis=1) / np.sum(mask)

        residue_set[i] = residue_map

    idxs = np.where(residue_set[:, mask] < 1e-6)
    N_show = normal_map/2 + 0.5
    N_show_flat =  N_show[mask]
    N_show_flat[idxs[1]] = np.array([0, 0, 0])
    N_show[mask] = N_show_flat
    plt.imshow(N_show)
    plt.axis('off')
    plt.show()

    plt.hist(z_range[idxs[0]])
    plt.show()

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 1)
    im = axes.imshow(residue_set[:, mask], vmin=0, vmax = 1e-5, cmap=plt.cm.jet)
    plt.yticks(np.arange(len(z_range))[::200], ['{:.2f}'.format(z) for z in z_range[::200]])
    axes.set_xlabel('Index of surface normal')
    axes.set_ylabel('Depth candidate')

    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    # cbar.formatter.set_powerlimits((0, -4))
    cbar.update_ticks()



    plt.show()



    return residue_set, z_range[np.argmin(residue_set[:, mask])]

def trace_all_pixel(point_cloud, mask, img_Set, LED_loc, z_range, normal_map,  save_dir):

    h, w = mask.shape

    residue_set = []
    z_best = np.zeros([h, w])
    for i in [32]:
        for j in [32]:
            if mask[i, j]:

                uv_coord = point_cloud[i, j, :2]
                img_one_pix = img_Set[:, i, j]
                residue, z_pre = get_residue(z_range, LED_loc, img_one_pix, uv_coord, normal_map, mask)
                residue_set.append(residue)
                z_best[i, j] = z_pre


    np.save(os.path.join(save_dir, 'z_best.npy'), z_best)
    np.save(os.path.join(save_dir, 'residue_set.npy'), np.array(residue_set))
    np.save(os.path.join(save_dir, 'z_range.npy'), np.array(z_range))



def draw_ambiguity(z_range_path, residue_path, mask):

    z_range = np.load(z_range_path)
    residue_set = np.load(residue_path)
    sort_residue = np.sort(residue_set, axis = 1)
    ratio = sort_residue[:, 0]
    ratio_map = np.zeros_like(mask).astype(np.float32)
    ratio_map[mask] = ratio
    plt.imshow(ratio_map, vmax = 1e-2)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':

    shape_name = 'ball_albedo'
    data_dir = r'F:\Project\SIREN\siren\data_rendering\normal_integration\poly2d\{}'.format(shape_name)

    save_dir = './NL_ambiguity/{}'.format(shape_name)
    createDir(save_dir)
    img_path = os.path.join(data_dir, 'img_set.npy')
    mask_path = os.path.join(data_dir, 'mask.npy')
    point_cloud_path = os.path.join(data_dir, 'point_cloud.npy')
    LEDs_path = os.path.join(data_dir, 'LEDs.npy')
    normal_path = os.path.join(data_dir, 'normal.npy')
    depth_path =  os.path.join(data_dir, 'depth.npy')

    depth_map = np.load(depth_path)

    z_range = np.arange(-5, 3, 0.005)
    trace_all_pixel(np.load(point_cloud_path), np.load(mask_path), np.load(img_path), np.load(LEDs_path), z_range, np.load(normal_path), save_dir)

    z_range_path = os.path.join(save_dir, 'z_range.npy')
    residue_set_path = os.path.join(save_dir, 'residue_set.npy')
    draw_ambiguity(z_range_path, residue_set_path, np.load(mask_path))


