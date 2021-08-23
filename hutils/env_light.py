import numpy as np
from matplotlib import pyplot as plt

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def build_sphere_map(light_dir, env_map, radius = 100, principle_dir=np.array([0,0,1])):
    assert light_dir is not None
    assert env_map is not None

    direction_num = len(light_dir)
    assert direction_num == len(env_map)

    sphere_map = np.zeros([2*radius + 1, 2*radius+1, 3])
    index = np.dot(light_dir, principle_dir) >= 0
    value = env_map[index]

    x = np.array(light_dir[index, 0] * radius).astype(np.int) + radius
    y = np.array(light_dir[index, 1] * radius).astype(np.int) + radius

    for i in range(len(index)):
        sphere_map[x[i], y[i], :] = value[i]

    return sphere_map

# light_dir = np.load('../experiment/static_analysis_intensity_profile/sample_light_dir.npy')
# env_maps = np.load('../experiment/static_analysis_intensity_profile/env_map_array_self.npy')
# build_sphere_map(light_dir, env_maps[0])