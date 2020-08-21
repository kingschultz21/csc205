#!/usr/bin/python
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

def imshow(img, title=""):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="gray")
    _ = plt.yticks([])
    _ = plt.xticks([])
    plt.title(title)

#-----------------------------------PART 1-------------------------------------------------------#
img = cv2.imread("input.jpg", 0)
std = 2
filter1D = cv2.getGaussianKernel(31, std)
filterG = filter1D * filter1D.T
filterG /= filterG.sum()


yett = np.pad(filterG, ((305,305),(225,225)), mode='constant', constant_values=0)
#imshow(yett, "31 x 31 Gaussian Kernel with Std = {}".format(std))

#---------------------------------------------------------------------------------------#
smoothed = cv2.filter2D(img, cv2.CV_32F, filterG, borderType=cv2.BORDER_REFLECT_101)
detail = img - smoothed

sharpened = img + 1.0 * detail
img_disp = np.maximum(0, np.minimum(255, sharpened.astype(int)))
imshow(smoothed, "Sharpened Image")



plt.waitforbuttonpress()