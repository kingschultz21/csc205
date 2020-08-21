# solution.py ---
#
# Filename: solution.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Nov  5 13:45:00 2018 (-0800)
# Version:
#

# Commentary:
#
#   Submitted by Connor Schultz (V00872823)
#   Date of Submission: Dec 4th
#   CSC 205 a7

# Change Log:
#
#
#
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:


import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import order_filter

TODO = None


def hough_transform(edges, num_r_bins, num_theta_bins):
    """Hough Transform.

    This function should return three things. The accumulator, the r values for
    each bin of the accumulator, the theta values for bins of the accumulator.

    """
    # Find all edge coordinates
    #
    # Behold the numpy magic!!! This function returns indices of where the
    # element is non-zero
    y, x = np.where(edges)
    x = x.reshape(-1, 1) - (edges.shape[1] - 1) / 2
    y = y.reshape(-1, 1) - (edges.shape[0] - 1) / 2
    
    #For diagonal length (maximum distance)
    height,width = edges.shape
    diag_len = np.sqrt(width * width + height * height) 

    #accumulator (rho vs theta)
    acc = np.zeros((num_r_bins,num_theta_bins))

    rs = np.zeros(num_r_bins)
    thetas = np.zeros(num_theta_bins)

    #Thetas from a specific range in radians
    #Bin using histograms
    rad_thets = np.deg2rad(np.arange(0, 360))
    t_hist = np.histogram(rad_thets,bins=num_theta_bins)

    rs = np.arange(0,diag_len/2)
    r_hist = np.histogram(rs,bins=num_r_bins)

    thetas = t_hist[1]
    
    print("Accumulating...")
    #Democracy Procedure begins
    for i in range(len(x)):
        #Get x and y edge points
        x_pt = x[i]
        y_pt = y[i]
        
        #vote in the accumulator
        for t in range(num_theta_bins):
            #rho = xcos(theta) + ysin(theta)
            #Calculate rho and adjust to fit in bins
            rho = x_pt * np.cos(t_hist[1][t]) + y_pt * np.sin(t_hist[1][t])
            rho = int(np.round(rho * (200/(diag_len*0.5))))
            #disregard negative rho values
            if(rho >= 0):
                acc[rho,t] += 1

    return acc, rs.flatten(), thetas.flatten()


def non_max_sup(acc):
    """Perform non-maximum suppresion with a 3-by-3 neighborhood
    
    """

    # TODO: 5 marks. As before, if you use for loops you will get 3 marks
    # instead of five. Hint: use the order_filter that I already imported. A
    # local maximum is strictly larger than all of its neighborhood.

    #Implementation based off hint give in Lecutre 31 slides
    domain = np.ones((3,3))
    nms_2largest = order_filter(acc, domain, 7)
    nms_map = np.subtract(acc,nms_2largest)
    #Binarize
    nms_map = np.where(nms_map > 0, 1, 0)
    return nms_map
    

def convert_to_r_theta(nms_map, rs, thetas):
    """ Gives the r, thetas in the nms map """

    r, theta = np.where(nms_map)

    return rs[r], thetas[theta]

def draw_line(img, r, theta):
    """ Draws a line using th r, theta representation """

    img_copy = img.copy()

    # r = x cos t + y sin t
    # y = - cost / sint * x + r / sint
    # x = - sint / cost * y + r / cost

    for _r, _theta in zip(r, theta):

        # _r = 7
        # print(_r, _theta * 180.0 / np.pi)
        # if _r > 8:
        #     continue

        cos = np.cos(_theta)
        sin = np.sin(_theta)

        if cos > sin:
            # this is when slope is large, we need to draw line based on y
            # extremes
            y_min = -(img.shape[0] - 1) / 2
            y_max = +(img.shape[0] - 1) / 2 
            x_min = _r / cos - sin / cos * y_min
            x_max = _r / cos - sin / cos * y_max
        else:
            # This is the opposite case
            x_min = -(img.shape[1] - 1) / 2 
            x_max = +(img.shape[1] - 1) / 2
            y_min = _r / sin - cos / sin * x_min
            y_max = _r / sin - cos / sin * x_max
        x_min += (img.shape[1] - 1) / 2 
        x_max += (img.shape[1] - 1) / 2 
        y_min += (img.shape[0] - 1) / 2 
        y_max += (img.shape[0] - 1) / 2 
        cv2.line(
            img_copy,
            (int(round(x_min)), int(round(y_min))),
            (int(round(x_max)), int(round(y_max))),
            (0,0,255), 1, cv2.LINE_AA)
            

    return img_copy


def main():

    # Read color image
    img = cv2.imread("input.jpg", 1)

    # We'll resize to a manageable size
    ds_rate = 8
    img = cv2.resize(img, (img.shape[1] // ds_rate, img.shape[0] // ds_rate))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find edges in images
    edges = cv2.Canny(img_gray, 200, 400)
    cv2.imwrite("edges.jpg", edges)
    
    # Get accumulator
    acc, rs, thetas = hough_transform(edges, 200, 200)

    # Save using matplotlib to look pretty
    plt.imsave("accumulator.jpg", acc)

    # Perform NMS map
    nms_map = non_max_sup(acc)
    plt.imsave("non_maximum_points.jpg", nms_map)

    # Select the top K points in the map to draw
    K = 10
    th = np.sort((nms_map * acc).flatten())[::-1][K]
    nms_map = nms_map * (acc > th)
    plt.imsave("top_k_points.jpg", nms_map)

    # Convert NMS map to r, theta
    r, theta = convert_to_r_theta(nms_map, rs, thetas)
    
    # Draw result
    res_img = draw_line(img, r, theta)
    cv2.imwrite("result.jpg", res_img)


if __name__ == "__main__":
    main()
    exit(0)


#
# solution.py ends here
