from hutils.fileio import readHDR
import cv2
import numpy as np

def tonemapping(img, gama=2.2, sat=0.5):
    tonemapDrago = cv2.createTonemapDrago(gama, sat)
    img = np.float32(img)
    ldrimg = tonemapDrago.process(img)
    im2_8bit = np.clip(ldrimg * 255, 0, 255).astype('uint8')
    return im2_8bit