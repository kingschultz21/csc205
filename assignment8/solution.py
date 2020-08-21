# solution.py ---
#
# Filename: solution.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Nov 26 09:11:18 2018 (-0800)
# Version:
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:

#   Connor Schultz (V00872923)
#   Date of Submission: December 9th 2018
#   CSC 205 Assigment 8
#

import argparse
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument(
    "--img_dir",
    type=str,
    default="./imgs",
    help="path to directory containing input images")

parser.add_argument(
    "--patch_size",
    type=int,
    default=51,
    help="number of corner points to use per image")

parser.add_argument(
    "--method",
    type=str,
    default="median",
    help="Method used to merge images",
    choices=["median", "mean"])


def harris_corners(img, config):
    """This function is based on 

    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1),
                               criteria)

    # Check for corners that are near borders and remove them as we need to
    # extract patches that are all inside the image. We'll simply check if
    # their coordinates are at least, half of the patch size away.
    half_size = config.patch_size // 2
    good_corners_tl = (corners > np.array([half_size, half_size])).all(axis=1)
    good_corners_br = (corners < np.array(
        [img.shape[1] - half_size, img.shape[0] - half_size])).all(axis=1)
    corners = corners[good_corners_tl * good_corners_br]

    return np.round(corners).astype(np.int)


def extract_patches(img, corner, config):

    # Create a copy of the image that is zero-padded so that we can ignore the
    # border effect

    pad = config.patch_size
    img_padded = np.pad(
        img, [(pad, pad), (pad, pad), (0, 0)],
        mode="constant",
        constant_values=0)

    patches = []
    for x, y in corner:

        xs = x - config.patch_size // 2 + pad
        ys = y - config.patch_size // 2 + pad
        xe = xs + config.patch_size
        ye = ys + config.patch_size

        patches += [img_padded[ys:ye, xs:xe, :]]

    return np.array(patches)


def draw_patches(img, corner, config):
    """ Draws patches on the image """

    for x, y in corner:

        xs = x - config.patch_size // 2
        ys = y - config.patch_size // 2
        xe = xs + config.patch_size
        ye = ys + config.patch_size
        cv2.rectangle(img, (xs, ys), (xe, ye), (0, 0, 255), 1)


def merge_images(imgs, method="median"):
    """Merge images 
    
    Parameters
    ----------

    imgs: numpy.ndarray, float
    
        This array is of shape (N, H, W, C), where N is the number of aligned
        images, H is the height of the entire alignment, W is the width, and
        C==3 is the number of channels. This array is also filled with NaNs,
        for points where no data point exists. You can use for example
        `numpy.nanmean` and `numpy.nanmedian` on this image, along axis=0, for
        example np.nanmean(imgs, axis=0) to average out all things except for
        the NaNs in the aligned image to get a ``blended'' image.

    """


    # This is so that the program does not crash. This will make your images
    # look weird
    TODO = np.mean(imgs, axis=0)

    if method == "median":
        print("median")
        # TODO: 1 Mark: implement the blending for nanmedian
        img_merged = np.nanmedian(imgs, axis=0)
        return img_merged.astype(np.uint8)

    elif method == "mean":
        print("mean")
        # TODO: 1 Mark: implement the blending for nanmean
        img_merged = np.nanmean(imgs,axis=0)
        return img_merged.astype(np.uint8)

    else:
        raise NotImplementedError("{} not supported".format(method))


def stitch_image(imgs_ref, img, config):

    # Squash the aligned images to create a single image we use to match
    img_ref = merge_images(imgs_ref, config.method)

    # Extract corners for both reference and the image to stitch
    corner_ref = harris_corners(img_ref, config)
    corner_img = harris_corners(img, config)

    # Extract patches centered at corners
    patches_ref = extract_patches(img_ref, corner_ref, config)
    patches_img = extract_patches(img, corner_img, config)

    # Find correspondences
    dists = np.sum(
        (patches_ref.reshape(1, -1, config.patch_size**2 * 3)
         - patches_img.reshape(-1, 1, config.patch_size**2 * 3))**2,
        axis=-1)
    # Create a threshold for the top 10 matches
    th = np.sort(dists.flatten())[10]
    img_to_ref = np.argmin(dists, axis=1)
    # Create a mask of good points
    good_mask = np.min(dists, axis=1) < th

    # Find global translation (from img to referrence). Note here that the same
    # points are in different positions in each image, and that gives us an
    # idea of how we need to move the img so that it's aligned with the
    # reference
    dxdy_all = corner_ref[img_to_ref] - corner_img
    dxdy = np.mean(dxdy_all[good_mask], axis=0).astype(np.int)

    # Place images side-by-side and draw the patch matching result
    img_side_by_side = np.zeros((max(img_ref.shape[0], img.shape[0]),
                                 img_ref.shape[1] + img.shape[1], 3),
                                dtype=np.uint8)
    # Copy images
    img_side_by_side[:img_ref.shape[0], :img_ref.shape[1]] = img_ref
    img_side_by_side[:img.shape[0], img_ref.shape[1]:img_ref.shape[1]
                     + img.shape[1]] = img
    # Draw patches
    corner_ref_good = corner_ref[img_to_ref[good_mask]]
    corner_img_good = corner_img[good_mask] + np.array([img_ref.shape[1], 0])
    # Draw patches on the reference image and on the img that we
    # are stitching now. Also draw a line between the two patches. The
    # following functions should be used. `draw_patches`, `cv2.line`
    draw_patches(img_side_by_side, corner_ref_good, config)
    draw_patches(img_side_by_side, corner_img_good, config)
    for p1, p2 in zip(corner_ref_good, corner_img_good):
        cv2.line(img_side_by_side, tuple(p1), tuple(p2), (0, 0, 255), 4,
                 cv2.LINE_AA)

    # # Compute start and end points of new image in reference image coordinates
    xs_img = 0 + dxdy[0]
    ys_img = 0 + dxdy[1]
    xe_img = img.shape[1] + dxdy[0]
    ye_img = img.shape[0] + dxdy[1]

    # Create new image that is the size of two images together
    w_new = max(img_ref.shape[1], xe_img) - min(0, xs_img)
    h_new = max(img_ref.shape[0], ye_img) - min(0, ys_img)
    # Create new stack of images for each image to be re-positioned
    img_new = np.zeros((imgs_ref.shape[0] + 1, h_new, w_new, 3),
                       dtype=np.float)
    img_new[:] = np.nan  # We'll initialize to nan

    # If the new image is on the left or the top of the reference image, make
    # new (0,0) according to this.
    xs_pad = -min(0, xs_img)
    ys_pad = -min(0, ys_img)
    # For image that is stitched
    xs_img += xs_pad
    ys_img += ys_pad
    xe_img += xs_pad
    ye_img += ys_pad
    # For reference image
    xs_ref = 0 + xs_pad
    ys_ref = 0 + ys_pad
    xe_ref = img_ref.shape[1] + xs_pad
    ye_ref = img_ref.shape[0] + ys_pad

    # Fill in the new image
    img_new[:-1, ys_ref:ye_ref, xs_ref:xe_ref] = imgs_ref
    img_new[-1, ys_img:ye_img, xs_img:xe_img] = img

    # Merge the all images together
    img_merged = merge_images(img_new, config.method)

    # Show merge result more graphically
    img_disp = img_merged.copy()
    cv2.rectangle(img_disp, (xs_img + 5, ys_img + 5), (xe_img - 5, ye_img - 5),
                  (0, 0, 255), 5)

    # Display all figured
    cv2.imshow("Stitch result", img_disp)
    cv2.imshow("Match result", img_side_by_side)
    cv2.waitKey(-1)

    return img_new


def main(config):

    # Read images
    img_files = os.listdir(config.img_dir)
    imgs = [
        cv2.imread(os.path.join(config.img_dir, _f), 1) for _f in img_files
        if _f.endswith(".jpg")
    ]

    imgs_ref = imgs[0].copy()[None].astype(np.float)
    assert imgs_ref.shape[-1] == 3

    for img in imgs[1:]:
        assert img.shape[-1] == 3
        imgs_ref = stitch_image(imgs_ref, img, config)


if __name__ == "__main__":

    config, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        parser.print_usage()
        exit(1)

    main(config)
    exit(0)

#
# solution.py ends here
