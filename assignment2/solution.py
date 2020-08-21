#!/usr/bin/env python3
# solution.py --- 
# 
# Filename: solution.py
# Description: 
# Author: Kwang Moo Yi
# Maintainer: 
# Created: Sun Sep  9 20:28:55 2018 (-0700)
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
# Connor Schultz
# V00872923
# csc205 A.2
# Due: Oct. 9th 2018

# References:
# Official OpenCV documentation was used to help with this assignment
# https://docs.opencv.org/3.4/dc/da5/tutorial_py_drawing_functions.html

import cv2
import numpy as np

# Modified print_info from A.1
# Draws the desired shape on the canvas as well
# as prints info from A.1
def parse_and_draw(canvas, line):

    # TODO: Modify print_info from assignment 1 to draw things according to

    line_list = []
    curr = ""
    for x in line:
        curr += x
        if (x == " "):
            line_list.append(curr.rstrip(" "))
            curr = ""
        elif (x == "\n"):
            line_list.append(curr.rstrip("\n"))
            break

    vals = []
    for x in line_list[1:]:
        vals.append(int(x))

    # TODO: parse and print!
    # Determines the correct format based on the keyword stored in line_list[0]
    # Once formatted the output is printed

    if (line_list[0] == "canvas"):
        canvas = np.zeros((vals[0], vals[1], 1), np.uint8)
        canvas.fill(255)
        print("Canvas size is %s x %s" % (line_list[1], line_list[2]))

    elif (line_list[0] == "circle"):
        cv2.circle(canvas, (vals[2], vals[3]), vals[1], 0, vals[0])
        print("Circle of thickness %s and radius %s at (%s, %s)" % (
            line_list[1], line_list[2], line_list[3], line_list[4]))

    elif (line_list[0] == "line"):
        cv2.line(canvas, (vals[1], vals[2]), (vals[3], vals[4]), 0, vals[0])
        print("Line of thickness %s from (%s, %s) to (%s, %s)" % (
            line_list[1], line_list[2], line_list[3], line_list[4], line_list[5]))

    elif (line_list[0] == "polyline"):
        pts = np.array([[vals[1], vals[2]], [vals[3], vals[4]], [vals[5],vals[6]],], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], False, 0,vals[0])
        print("Polygon Line of thickness %s passing through, (%s, %s), (%s, %s), (%s, %s)" % (
            line_list[1], line_list[2], line_list[3], line_list[4], line_list[5], line_list[6], line_list[7]))

    elif (line_list[0] == "polyfill"):
        pts = np.array([[vals[0], vals[1]], [vals[2], vals[3]], [vals[4], vals[5]],[vals[6],vals[7]] ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], True, 0)
        print("Polygon Fill for lines passing through, (%s, %s), (%s, %s), (%s, %s), (%s, %s)" % (
            line_list[1], line_list[2], line_list[3], line_list[4], line_list[5], line_list[6], line_list[7],
            line_list[8]))
    else:
        print("Please enter a valid input")

    return canvas


if __name__ == "__main__":

    # Load file (Assignment 1)

    # Open required files
    file_i = open("input.txt", "r")
    file_o = open("output.txt", "w")

    # read file line by line and store in a list
    lines = []
    for line in file_i.readlines():
        lines.append(line)
        file_o.write(line)


    # Initialize canvas to None
    canvas = None
    # Parse and draw information
    for line in lines:
        canvas = parse_and_draw(canvas, line)

    # TODO: display canvas on screen
    cv2.imshow('Canvas',canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Add another line for drawing a circle at the end at (50, 50) with radius
    # of 20 and thickness 5 (Assignment 1)
    file_o.write("circle 5 20 50 50\n")

    # close files
    file_o.close()
    file_i.close()

# 
# solution.py ends here
