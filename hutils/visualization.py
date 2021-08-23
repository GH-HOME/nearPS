import cv2
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
import numpy as np
import matplotlib
import os
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
# font = FontProperties()
# FontProperties(fname="C:\\Windows\\Fonts\\times.ttf")
# font = {'family' : "Times New Roman",
#         'weight' : 'normal',
#         'size'   : 18}
#
# matplotlib.rc('font', **font)


def saveFigMultiPage(filename, figs=None, dpi=200):
    """
    Save all the plt plot into a single pdf file
    """
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    plt.close('all')


def scatter_3d(PointSet):
    """
    Visualize the scattered 3D points
    PointSet: [N, 3] point coordinates
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(PointSet[:, 0], PointSet[:, 1], PointSet[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    plt.show()


def create_gif(imgs, save_path, mask=None, fps = 10):
    """
    create gif from images
    :param imgs: [N, H, W]
    :param save_path:
    :param mask:
    :return:
    """

    fig = plt.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    import imageio
    with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
        for i in range(len(imgs)):
            if mask is not None:
                import cv2
                idx = mask != 0
                imgs[i][~idx] = np.NaN
                imgs[i] = imgs[i] / imgs[i][idx].max()
            else:
                imgs[i] = imgs[i] / imgs[i].max()

            writer.append_data(imgs[i])




def plot_imgset(imgs, normalize = True, interval = 0.02, save_path = None):
    """
    plot a set of images with interval second
    Parameters
    ----------
    imgs: [N, H, W, 3]
    interval : time interval in second

    Returns None
    -------

    """
    [N, H, W] = imgs.shape[0:3]
    for i in range(N):
        plt.figure()
        if normalize:
            plt.imshow(imgs[i]/imgs[i].max())
        else:
            plt.imshow(imgs[i])
        plt.title('img_{}'.format(i))
        # plt.pause(0.2)

    if save_path is not None:
        saveFigMultiPage(save_path)

def save_fig_no_margin(img, file_name):
    plt.imshow(img)
    plt.axis('off')

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(file_name, bbox_inches='tight',
                pad_inches=0)

# def plt_normal(normal):
#     im = plt.imshow(normal/2 + 0.5)
#     plt.axis('off')
#     return im
#



def save_normal_no_margin(N, mask, file_name, white = False):
    img = N_2_N_show(N, mask, white)
    img = img[:,:,::-1] * 255
    img = np.uint8(img)
    cv2.imwrite(file_name, img)


def save_plt_fig_with_title(file_name, title, dpi=300, transparent = False):
    plt.title(title)
    plt.axis('off')
    plt.savefig(file_name, dpi=dpi, bbox_inches='tight',
                pad_inches=0, transparent = transparent)


def save_plt_fig_no_margin(file_name, dpi=100, transparent = False):
    plt.savefig(file_name, dpi=dpi, bbox_inches='tight',
                pad_inches=0, transparent = transparent)

def set_zero_margin():
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())


def plt_normal(normal, mask=None, savepath=None):
    N_show = normal/2 + 0.5
    if mask is not None:
        N_show[~mask] = 0
    im = plt.imshow(N_show)
    plt.axis('off')
    set_zero_margin()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)
    return im


def plt_error_map(err_map, mask, vmin=0, vmax=40, withbar=False, title=None, img_path=None):
    err_map[~mask] = 0
    fig, axes = plt.subplots(1, 1)
    im = axes.imshow(err_map, vmin=vmin, vmax = vmax, cmap=plt.cm.jet)
    plt.axis('off')
    if title is not None:
        plt.title(title)

    if withbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        # cbar.formatter.set_powerlimits((0, -4))
        cbar.update_ticks()

    if img_path is not None:
        plt.savefig(img_path, transparent=True, dpi = 300, bbox_inches='tight')
        plt.close('all')

    return im

def plt_error_map_cv2(err_map, mask, vmin=0, vmax=40):
    err_map = np.maximum(err_map, vmin)
    err_map = np.minimum(err_map, vmax)
    err_map = err_map / (vmax - vmin) * 255
    err_map = err_map.astype(np.uint8)
    im_color = cv2.applyColorMap(err_map, cv2.COLORMAP_JET)
    im_color[~mask] = 255
    return im_color



def N_2_N_show(N, mask, white=False):
    N_show = N/2 + 0.5
    N_show[~mask] = 0
    if white:
        N_show[~mask] = 1
    return N_show


def visualize_GT_EST_ERR_map(N_gt, N_est, Err_map, mask, vmax=20):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, axis = plt.subplots(1, 3, figsize = (12, 4.2))
    axis[0].imshow(N_2_N_show(N_gt, mask))
    axis[0].set_title('GT')
    axis[1].imshow(N_2_N_show(N_est, mask))
    axis[1].set_title('Est')

    im = axis[2].imshow(Err_map, plt.cm.jet, vmax = vmax)
    axis[2].set_title('MAE: {:.2f}'.format(np.mean(Err_map[mask])))
    divider = make_axes_locatable(axis[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')

    [axi.set_axis_off() for axi in axis.ravel()]
    plt.tight_layout()

def visualize_EST_ERR_map(N_est, Err_map, mask, vmax=40):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axis = plt.subplots(1, 2, figsize = (8, 4.2))
    axis[0].imshow(N_2_N_show(N_est, mask))
    # axis[0].set_title('Est')

    im = axis[1].imshow(Err_map, plt.cm.jet, vmax = vmax)
    axis[1].set_title('MAE: {:.2f}'.format(np.mean(Err_map[mask])))
    divider = make_axes_locatable(axis[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')

    [axi.set_axis_off() for axi in axis.ravel()]
    plt.tight_layout()
    return np.mean(Err_map[mask])


def visualize_EST_ERR_albedo_map(A_est, A_gt, mask, vmax = 1):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # erode mask
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask_erode = cv2.erode(mask.astype(np.uint8), kernel).astype(np.bool)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_erode2 = cv2.erode(mask.astype(np.uint8), kernel).astype(np.bool)
    A_gt[~mask_erode2] = 0
    A_est[~mask_erode2] = 0

    fig, axis = plt.subplots(1, 2, figsize = (8, 4.2))
    A_est = A_est[:, :, 0:3]
    A_gt = A_gt[:, :, 0:3]
    A_gt = A_gt / A_gt[mask_erode].max()
    A_gt_mean = np.mean(A_gt[mask_erode])
    A_est = A_est / A_est[mask_erode].mean() * A_gt_mean
    axis[0].imshow(A_est)

    Err_map = np.zeros_like(A_gt)
    Err_map[mask] = np.sqrt(np.square(A_est[mask] - A_gt[mask]))

    im = axis[1].imshow(Err_map, plt.cm.jet, vmax = vmax)
    axis[1].set_title('MSE: {:.2f}'.format(np.mean(Err_map[mask])))
    divider = make_axes_locatable(axis[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')

    [axi.set_axis_off() for axi in axis.ravel()]
    plt.tight_layout()

    return np.mean(Err_map[mask])


def visualize_albedo_distribution_RGB(albedo_vector, num_pt=100, save_path=None):
    """
    show the RGB albedo distribution in 3D scatter
    :param albedo_vector: [p, 3]
    :return:
    """
    from matplotlib import style

    style.use('classic')
    fig = plt.figure()
    index = np.random.permutation(len(albedo_vector))[:num_pt]
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(albedo_vector[index, 0], albedo_vector[index, 1], albedo_vector[index, 2], marker='o')
    ax.set_xlabel(r'$\lambda_R$')
    ax.set_ylabel(r'$\lambda_G$')
    ax.set_zlabel(r'$\lambda_B$')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.view_init(15, 30)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def draw_light_distribution(light_dir, color=None, save_path=None):
    light_dir = light_dir / np.linalg.norm(light_dir, axis=1, keepdims=True)
    light_ins = np.linalg.norm(light_dir, axis=1)
    p = light_dir[:, 0] #/ light_dir[:, 2]
    q = light_dir[:, 1] #/ light_dir[:, 2]

    from matplotlib.patches import Circle
    sphere_center = np.array([0, 0])
    circle  = plt.Circle((sphere_center[0], sphere_center[1]), 0.9, fill=False, linewidth=2, alpha=0.8, edgecolor='darkred')
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect(1)
    ax.add_artist(circle)
    for i in range(len(p)):
        if i==1:
            continue
        ax.text(p[i]-0.03, q[i]-0.13, '{}'.format(i + 1), fontsize=35)
    if color is not None:
        ax.scatter(p, q, c=color, s=180, alpha = 0.4, edgecolors='darkred', linewidth=2)

    else:
        ax.scatter(p, q, c=light_ins, s=20)
        ax.axis('off')
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    set_zero_margin()
    if save_path is not None:
        plt.savefig(save_path, dpi=1000, transparent=True, bbox_inches='tight',
                pad_inches=0)
    else:
        plt.show()



def save_rgb_img_from_npy(input_img_path, mask_path):
    img = np.load(input_img_path)
    mask = np.load(mask_path).astype(np.bool)
    [h, w, f] = img.shape

    output_folder = os.path.dirname(input_img_path)
    for i in range(f):
        img_c = img[:,:,i]
        img_c_show = np.minimum(img_c/img_c.mean() * 40, 255)
        cv2.imwrite(os.path.join(output_folder, 'img_{}.png'.format(i)), img_c_show)
        save_transparent_img(os.path.join(output_folder, 'img_{}.png'.format(i)), mask)

def save_transparent_img(input_img_path, mask, output_img_path=None):
    """
    convert an image to a transparent image by setting the region outof mask as transparent
    """
    assert input_img_path is not None
    assert mask is not None
    if output_img_path is None:
        output_img_path = input_img_path

    img = Image.open(input_img_path)
    img = img.convert("RGBA")

    pixdata = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                pixdata[x, y] = (255, 255, 255, 0)

    img.save(output_img_path, "PNG")



def visualize_exp_data(save_dir, method_list, N_gt_path, mask_path, vmax=40):


    N_gt = np.load(N_gt_path)
    for method_name in method_list:

        N_est_path = os.path.join(save_dir, 'N_est_{}.npy'.format(method_name))
        mask_path_method =  os.path.join(save_dir, 'mask_{}.npy'.format(method_name))
        if os.path.exists(mask_path_method):
            mask = np.load(mask_path_method)
        else:
            mask = np.load(mask_path)

        N_est = np.load(N_est_path)
        if N_est.shape[0] != N_gt.shape[0]:
            N_est = cv2.resize(N_est, (N_gt.shape[0], N_gt.shape[1]), cv2.INTER_NEAREST)


        N_norm = np.linalg.norm(N_est, axis=2)
        mask_valid = np.logical_or(np.isnan(N_norm), np.isinf(N_norm))
        mask_method = np.logical_and(~mask_valid, mask)


        normal_png_path = os.path.join(save_dir, 'N_est_{}.png'.format(method_name))
        save_normal_no_margin(N_est, mask=mask_method, file_name=normal_png_path)
        from hutils.PhotometricStereoUtil import evalsurfaceNormal
        error_map, MAE, MedianE = evalsurfaceNormal(N_gt, N_est, mask_method)
        error_map_color = plt_error_map_cv2(error_map, mask_method, vmax = vmax)
        error_map_path = os.path.join(save_dir, 'err_N_{}_{:.2f}_{:.2f}.png'.format(method_name, MAE, MedianE))
        cv2.imwrite(error_map_path, error_map_color)
        save_transparent_img(error_map_path, mask_method)
        save_transparent_img(normal_png_path, mask_method)

        normal_png_path = os.path.join(save_dir, 'N_gt.png')
        save_normal_no_margin(N_gt, mask=mask, file_name=normal_png_path, white=True)
        save_transparent_img(normal_png_path, mask)


def createImgGIF(gif_path, img_filenames, fps = 10):
    import imageio, tqdm
    print('creating GIF...')
    with imageio.get_writer(gif_path, mode='I', fps=fps) as writer:
        for i, filename in enumerate(tqdm.tqdm(img_filenames)):
            image = imageio.imread(filename)
            writer.append_data(image)



if __name__ == '__main__':

    img_path = r'F:\Project\SIREN\siren\data\output_dir_near_light\09_reading\orthographic\lambertian\scale_256_256\wo_castshadow\shading\render_img\imgs.npy'
    img = np.load(img_path)
    create_gif(img, save_path = img_path[:-3]+'gif')