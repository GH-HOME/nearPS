import numpy as np
from skimage.transform import resize

def worldDir2ObjDir(worldDirection):
    """
    transfer the direction from worldirection to object direction
    :param worldDirection: [N, 3]
    :return: ObjectDirection
    """

    assert worldDirection is not None
    ObjectDirection = np.copy(worldDirection)
    ObjectDirection[:, 0] = - worldDirection[:, 0]
    ObjectDirection[:, 2] = - worldDirection[:, 2]
    return ObjectDirection

def worldDir2ObjDirM(worldDirection):
    """
    transfer the direction from worldirection to object direction
    :param worldDirection: [M, N, 3]
    :return: ObjectDirection
    """

    assert worldDirection is not None
    ObjectDirection = worldDir2ObjDir(worldDirection.reshape(-1, 3))
    ObjectDirection = ObjectDirection.reshape(worldDirection.shape)
    return ObjectDirection

def ObjDir2worldDir(ObjectDirection):
    """
    transfer the direction from worldirection to object direction
    :param ObjectDirection: [N, 3]
    :return: worldDirection
    """

    assert ObjectDirection is not None
    worldDirection = np.zeros_like(ObjectDirection)
    worldDirection[:, 0] = - ObjectDirection[:, 0]
    worldDirection[:, 2] = - ObjectDirection[:, 2]
    return worldDirection


def latitude_longitude_2_spherical_direction(env_map, resize_scale=None):

    if resize_scale is not None:
        env_map = resize(env_map, (resize_scale[0], resize_scale[1]))
    h, w, _ = env_map.shape

    theta = np.linspace(-np.pi / 2, np.pi / 2, num=h)
    phi = np.linspace(-np.pi, np.pi, num=w)
    r = 1
    x = r * np.dot(np.sin(theta)[:, np.newaxis], np.cos(phi)[np.newaxis, :])
    y = r * np.dot(np.sin(theta)[:, np.newaxis], np.sin(phi)[np.newaxis, :])
    z = r * np.dot(np.cos(theta)[:, np.newaxis], np.ones([1, len(phi)]))
    intensity = env_map.reshape(-1, 3)

    light_dir = np.dstack([x, y, z]).reshape(-1, 3)

    return [light_dir, intensity]
