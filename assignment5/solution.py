# solution.py ---
#
# Filename: solution.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Oct 15 13:06:48 2018 (-0700)
# Version:
#

# Commentary:
#
#
#
#

#References:
#   CSC 205 Slides (Filtering II, Spectral methods, Linear Filtering)
#   CSC 205 GitHub
#   OpenCv documentation
#   NumPy documentation
#

# Change Log:
#
#
#
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:

    #Submitted by Connor Schultz (V00872923)
    #Date of Submission: Oct.6th 2018

import time

import cv2
import numpy as np
from numpy.fft import fftshift
#from matplotlib import pyplot as plt
from numpy.fft import fft2
from numpy.fft import ifft2
import math


# Placeholder to stop auto-syntac checking from complaining
TODO = None

def imshow(img, title=""):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="gray")
    _ = plt.yticks([])
    _ = plt.xticks([])
    plt.title(title)

def main():

    # Read Image in grayscale and show
    img = cv2.imread("input.jpg", 0)
    cv2.imwrite("orig.png", img)

    # ----------------------------------------------------------------------
    # Create Filter
    # 
    # TODO: 3 Marks: Create sharpen filter from the lecture, but with a
    # Gaussian filter form the averaging instead of the mean filter. For the
    # Gaussian filter, use a kernel with size 31x31 with sigma 5. For the unit
    # impulse set the multiplier to be 2.
    impulse = np.ones((31,31))
    impulse[int(math.ceil(31/2))][int(math.ceil(31/2))] = 2

    '''
        Create Guassian Kernel
    '''
    kernel1D = cv2.getGaussianKernel(31, 5)
    kernelG = kernel1D * kernel1D.T
    kernelG /= kernelG.sum()
    kernelG = kernelG


    # ----------------------------------------------------------------------
    # Filter with FFT
    # 
    # TODO: 1 Mark: Pad filter with zeros to have the same size as the image,
    # but with the filter in the center. This creates a larger filter, that
    # effectively does the same thing as the original image.
    '''
        Pad Kernel with zeros
        Implentation based on slides from Dr. Yi
    '''
    kernel_padded = np.zeros_like(img).astype(float)
    pad_y = (img.shape[0]  - kernelG.shape[0]) // 2
    pad_x = (img.shape[1]  - kernelG.shape[1]) // 2
    kernel_padded[pad_y:pad_y+kernelG.shape[0], pad_x:pad_x+kernelG.shape[1]] = kernelG


    # Shift filter image to have origin on 0,0. This one is done for you. The
    # exact theory behind this was not explained in class so you may skip this
    # part. Drop by my office hours if you are interested.
    kernel_padded_shifted = fftshift(kernel_padded)
    

    # TODO: 1 Mark: Move all signal to Fourier space (DFT).
    img_fft = fft2(img)
    kernel_fft = fft2(kernel_padded_shifted)

    # Display signals in Fourier Space
    # I put some visualization here to help debugging :-)
    cv2.imwrite(
        "orig_fft.png",
        np.minimum(1e-5 * np.abs(fftshift(img_fft)), 1.0) * 255.)
    cv2.imwrite(
        "filt_fft.png",
        np.minimum(1e-1 * np.abs(fftshift(kernel_fft)), 1.0) * 255.)

    # TODO: 1 Mark: Do filtering in Fourier space
    '''
        Implentation based off of slides from Dr. Yi
    '''
    img_filtered = np.real(ifft2(fftshift(img_fft * kernel_fft)))
    img_filtered = np.maximum(0, np.minimum(255, img_filtered.astype(int)))
    
    fft_copy = img_fft.copy()
    mask = np.zeros_like(fft_copy)
    cx = mask.shape[0] // 2
    cy = mask.shape[1] // 2
    # TODO: 1 Mark: Bring back to Spatial domain (Inverse DFT)
    # TODO: 2 Marks: Throw away the imaginary part and clip between 0 and 255
    # to make it a real image.
    mask[cx-5000:cx+5000,cy-5000:cy+5000] = 1
    fft_copy = fftshift(fftshift(fft_copy*mask, axes=0), axes=1)
    img_inverse_fft = np.real(ifft2(fft_copy))
    
    details1 = img - img_inverse_fft
    img_sharpened = img + 1.0 * details1
    img_sharpened = np.maximum(0, np.minimum(255, img_sharpened.astype(int)))

    cv2.imwrite("res_fft.png", img_sharpened.astype(np.uint8))

    # ----------------------------------------------------------------------
    # Filter with OpenCV
    # TODO: 1 Mark: Use padded filter and cyclic padding (wrap) to get exact results
    # TOOD: 1 Mark: Clip image for display
    blur = cv2.filter2D(img, cv2.CV_32F, kernel_padded, borderType=cv2.BORDER_REFLECT_101)
    details2 = img - blur
    img_sharpened = img + 1.0 * details2
    img_sharpened = np.maximum(0, np.minimum(255, img_sharpened.astype(int)))
    cv2.imwrite("res_opencv.png", img_sharpened.astype(np.uint8))

    cv2.waitKey(-1)


if __name__ == "__main__":
    main()
    exit(0)

#
# solution.py ends here
