import numpy as np

def pixel2acturalDistance(pixelWidth, imageWidth, camera_focal, CCD_width, depth):
    """
    Under perspective projection, find the actural length of a pixelWidth region in a image with imageWidth
    :param pixelWidth: region width
    :param imageWidth:
    :param camera_focal: e,g, 85 mm
    :param CCD_width:
    :param depth:  region depth
    :return:
    """
    return pixelWidth * depth * CCD_width/ (camera_focal *imageWidth)


def createIntrinsicMatrix(camera_focal, CCD_size, img_size):
    """
    create ideal camera intrinsic matrix from the param
    : camera_focal: camera actual focal length in mm
    :param CCD_size: [CCD width, CCD height] in mm
    :param img_size: [img width, img height] in pix
    """

    camera_k = np.eye(3)
    camera_k[0,0] = camera_focal / CCD_size[0] * img_size[0]
    camera_k[1, 1] = camera_focal / CCD_size[1] * img_size[1]
    camera_k[0, 2] = img_size[0]/2
    camera_k[1, 2] = img_size[1]/2

    return camera_k
