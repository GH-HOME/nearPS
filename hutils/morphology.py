import numpy as np
import cv2

def create_crop_mask_under_extern_mask(center, radius, mask_input):
    """
    create a croped mask part belongs to the mask_input
    Parameters
    ----------
    center: mask crop center
    radius: mask crop radius
    mask_input: the mask which used to indicate which part is valid

    Returns croped mask with shape and format equal to the mask_input
    -------

    """
    assert mask_input is not None
    h, w = mask_input.shape
    assert (h > center[0]) and (center[0] > -1)
    assert (w > center[1]) and (center[1] > -1)
    mask_crop = np.zeros_like(mask_input)
    mask_crop[center[0] - radius: center[0] + radius + 1, center[1] - radius: center[1] + radius + 1] = True
    mask_crop = np.logical_and(mask_input, mask_crop)
    return mask_crop


def create_crop_mask(center, radius, height, width):
    """
    create a croped mask with specific height and width
    Parameters
    ----------
    center: mask crop center
    radius: mask crop radius
    height: the mask height
    width:  the mask width

    Returns croped mask with shape and format equal to the mask_input
    -------

    """
    assert (height > center[0]) and (center[0] > -1)
    assert (width > center[1]) and (center[1] > -1)
    mask_crop = np.zeros([height, width]).astype(np.bool)
    mask_crop[center[0] - radius: center[0] + radius + 1, center[1] - radius: center[1] + radius + 1] = True
    return mask_crop