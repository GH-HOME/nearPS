import pyvista as pv
import os
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import cv2
pv.set_plot_theme("document")

def move_left(mask):
    return np.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]


def move_right(mask):
    return np.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]


def move_top(mask):
    return np.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]


def move_bottom(mask):
    return np.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]


def move_top_left(mask):
    return np.pad(mask, ((0, 1), (0, 1)), "constant", constant_values=0)[1:, 1:]


def move_top_right(mask):
    return np.pad(mask, ((0, 1), (1, 0)), "constant", constant_values=0)[1:, :-1]


def move_bottom_left(mask):
    return np.pad(mask, ((1, 0), (0, 1)), "constant", constant_values=0)[:-1, 1:]


def move_bottom_right(mask):
    return np.pad(mask, ((1, 0), (1, 0)), "constant", constant_values=0)[:-1, :-1]


def normalize_normal_map(N):
    H, W, C = N.shape
    N = np.reshape(N, (-1, C))
    N = normalize(N, axis=1)
    N = np.reshape(N, (H, W, C))
    return N


def construct_facets_from_depth_map_mask(mask):
    idx = np.zeros_like(mask, dtype=np.int)
    idx[mask] = np.arange(np.sum(mask))

    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)

    facet_top_left_mask = np.logical_and.reduce((facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    return np.hstack((4 * np.ones((np.sum(facet_top_left_mask), 1)),
               idx[facet_top_left_mask][:, None],
               idx[facet_bottom_left_mask][:, None],
               idx[facet_bottom_right_mask][:, None],
               idx[facet_top_right_mask][:, None])).astype(np.int)


def construct_vertices_from_depth_map_and_mask(mask, depth_map, step_size=1):
    H, W = mask.shape
    yy, xx = np.meshgrid(range(W), range(H))
    xx = np.flip(xx, 0)
    xx = xx * step_size
    yy = yy * step_size

    vertices = np.zeros((H, W, 3))
    vertices[..., 0] = xx
    vertices[..., 1] = yy
    vertices[..., 2] = depth_map
    return vertices[mask]



def map_depth_map_to_point_clouds(depth_map, mask, K):
    H, W = mask.shape
    yy, xx = np.meshgrid(range(W), range(H))
    xx = np.flip(xx, axis=0)
    u = np.zeros((H, W, 3))
    u[..., 0] = xx
    u[..., 1] = yy
    u[..., 2] = 1
    u = u[mask].T  # 3 x m
    p_tilde = (np.linalg.inv(K) @ u).T
    return p_tilde * depth_map[mask, np.newaxis]


def scatter_3D_default_view(obj_path = None, mesh = None, img_name = None, window_size = [512, 384], title = None):
    """ save the screen shot of obj files in the same view point"""

    # specular parameter
    ambient_value = 0.3
    diffuse_value = 0.5
    specular_value = 0.3
    p = pv.Plotter(border=False)

    p.set_background(color="white")

    if mesh is None and obj_path is not None:
        mesh = pv.read(obj_path)


    # step 2: find the good camera pose
    cpos = [
        (0, 1, -2),
        (0, 0, 2),
        (0, 1, 0),
    ]

    cpos_init, img = pv.plot(mesh,
                 cpos=cpos,
                 color="w",
                 smooth_shading=True,
                 screenshot=True,
                 off_screen=True,
                 diffuse=diffuse_value,
                 ambient=ambient_value,
                 specular=specular_value,
                 show_scalar_bar=False,
                 show_axes=False,
                 window_size = window_size,
                 text = title)


    # if title is not None:
    #     p.add_title(title, font_size=24)

    pv.close_all()
    if img_name is not None:
        cv2.imwrite(img_name, img)
    else:
        pv.plot()



def scatter_3D_default_view_debug(obj_path = None, mesh = None, img_name = None, window_size = [512, 384], title = None):
    """ save the screen shot of obj files in the same view point"""

    # specular parameter
    ambient_value = 0.3
    diffuse_value = 0.5
    specular_value = 0.3
    p = pv.Plotter(border=False)

    p.set_background(color="white")

    if mesh is None and obj_path is not None:
        mesh = pv.read(obj_path)

    # step 2: find the good camera pose

    cpos = [
        (0, 0, -2),
        (0, 0, 2),
        (0, 1, 0),
    ]

    cpos_init = pv.plot(mesh,
                cpos=cpos,
                 color="w",
                 smooth_shading=True,
                 # screenshot=True,
                 # off_screen=True,
                 diffuse=diffuse_value,
                 ambient=ambient_value,
                 specular=specular_value,
                 show_scalar_bar=False,
                 show_axes=False,
                 window_size = window_size,
                 text = title)
    print(cpos_init)

    # if title is not None:
    #     p.add_title(title, font_size=24)

    pv.close_all()
    if img_name is not None:
        cv2.imwrite(img_name, img)
    else:
        pv.plot()

def generate_mesh(pointcloud, mask, img_name, window_size, title):
    """
    pointcloud: [H, W, 3]
    """

    facets = construct_facets_from_depth_map_mask(mask)
    # vertices = construct_vertices_from_depth_map_and_mask(mask, depth_map, step_size)
    vertices = pointcloud[mask]
    surface = pv.PolyData(vertices, facets)
    pv.save_meshio(img_name[:-3]+'obj', surface)
    scatter_3D_default_view(mesh = surface, img_name = img_name, window_size = window_size, title = title)



def compare_two_mesh(mesh_path1, mesh_path2):

    ambient_value = 0.3
    diffuse_value = 0.5
    specular_value = 0.3
    p = pv.Plotter(border=False)

    mesh_gt = pv.read(mesh_path1)
    mesh_est = pv.read(mesh_path2)

    cpos_init = p.add_mesh(mesh_gt,
                           color="w",
                           smooth_shading=True,
                           # screenshot=True,
                           # off_screen=True,
                           diffuse=diffuse_value,
                           ambient=ambient_value,
                           specular=specular_value,
                           show_scalar_bar=False,
                           opacity=0.75
                           )

    cpos_init = p.add_mesh(mesh_est,
                           color="cyan",
                           smooth_shading=True,
                           # screenshot=True,
                           # off_screen=True,
                           diffuse=diffuse_value,
                           ambient=ambient_value,
                           specular=specular_value,
                           show_scalar_bar=False,
                           opacity=1
                           )


    # p.camera_position = cpos
    p.show()

if __name__ == "__main__":
    cpos = [
        (0, 0, -2),
        (0, 0, 2),
        (0, 1, 0),
    ]

    dargs = dict(show_edges=True)
    # Rotate the mesh to have a second mesh
    mesh_gt = pv.read(r'F:\Project\blender_2.8_rendering\tool\nearlight\output_dir_near_light\09_reading\perspective\lambertian\scale_256_256\wo_castshadow\shading\render_para\shape.obj')
    mesh_est = pv.read(r'F:\Project\SIREN\siren\data\output_dir_near_light\09_reading\perspective\lambertian\scale_256_256\wo_castshadow\shading\nearPS\2021_08_09_23_05_58_finite_diff_l1\test\iter_08500_Z_est.obj')

    # specular parameter
    ambient_value = 0.3
    diffuse_value = 0.5
    specular_value = 0.3
    p = pv.Plotter(border=False)

    p.set_background(color="white")

    # step 2: find the good camera pose
    cpos = [
        (0, 1, -2),
        (0, 0, 2),
        (0, 1, 0),
    ]

    cpos_init, img = pv.plot(mesh_gt,
                             cpos=cpos,
                             color="w",
                             smooth_shading=True,
                             diffuse=diffuse_value,
                             ambient=ambient_value,
                             specular=specular_value,
                             show_scalar_bar=False,
                             show_axes=False,
                             window_size=(1280, 720)
                             )

    # if title is not None:
    #     p.add_title(title, font_size=24)

    pv.close_all()
    pv.plot()

    import sys
    sys.exit(0)

    ambient_value = 0.3
    diffuse_value = 0.5
    specular_value = 0.3
    p = pv.Plotter(notebook=0, border=False, off_screen=True)

    cpos_init = p.add_mesh(mesh_gt,
                 color="w",
                 smooth_shading=True,
                 # screenshot=True,
                 # off_screen=True,
                 diffuse=diffuse_value,
                 ambient=ambient_value,
                 specular=specular_value,
                 show_scalar_bar=False,
                    opacity= 0.75
                )

    cpos_init = p.add_mesh(mesh_est,
                        color="cyan",
                        smooth_shading=True,
                        # screenshot=True,
                        # off_screen=True,
                        diffuse=diffuse_value,
                        ambient=ambient_value,
                        specular=specular_value,
                        show_scalar_bar=False,
                        opacity=1
                        )

    p.add_mesh(mesh_gt, color="blue", opacity=0.35, **dargs)
    p.add_mesh(mesh_est, color="white", opacity=0.35, **dargs)
    p.camera_position = cpos

    # p.show(auto_close=False)
    p.show()
    # p.open_gif("test.gif")

    # # Update camera and write a frame for each updated position
    # nframe = 100
    #
    # radius = 5
    # for i in range(nframe):
    #     x = np.cos(i * np.pi / 15.0)
    #     z = np.sqrt(radius**2 - x **2)
    #     p.camera_position = [
    #         (x, 0, z),
    #         (0, 0, 3),
    #         (0, 1, 0),
    #     ]
    #     p.write_frame()
    #
    # # Close movie and delete object
    # p.close()

