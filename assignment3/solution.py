#!/usr/bin/env python3
# solution.py --- 
# 
# Filename: solution.py
# Description: 
# Author: Kwang Moo Yi
# Maintainer: 
# Created: Mon Oct  8 13:09:16 2018 (-0700)
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

# Details:

'''
    Connor Schultz
    V00872923
    csc205 assignment 3
    Due: Oct. 23rd 2018

'''
import cv2
import numpy as np
import sys

# Just a dummy assignment so that TODO lines do not cause errors. Safe to
# ignore.
TODO = None


# ------------------------------------------------------------
# Not nice, but we'll just use some global variables
canvas = None
poly_pt_list = []
curr_poly = []
mouseX = -1
mouseY = -1
click = False
dblclick = False
counter = 0
# ------------------------------------------------------------

def mouse_callback(event, x, y, flags, params):
    """Mouse callback function"""

    global canvas, click, dblclick, mouseX, mouseY

    if event == cv2.EVENT_MOUSEMOVE:
        mouseX = x
        mouseY = y
    if event == cv2.EVENT_LBUTTONDBLCLK:
        dblclick = True
    elif event == cv2.EVENT_LBUTTONUP:
        click = True

'''
    Initialize Polygon Variables
'''
def init():
    global poly_pt_list
    poly_pt_list = []
    curr_poly = []
    counter = 1
    

if __name__ == "__main__":

    # Create named window
    window_name = "Press (r) to reset, (q) to quit"
    cv2.namedWindow(window_name)

    # TODO: Set mouse callback to the window (1 mark)
    TODO
    cv2.setMouseCallback(window_name, mouse_callback)

    #Check var is a check variable that handles single and double click separation
    check_var = False
    # Initialize mouse
    init()
    while True:
        # Deal with mouse input
        if dblclick:
            # TODO: Write code so that double click would move onto the next
            # polygon (1 mark)
            TODO
            poly_pt_list.insert(counter,curr_poly)
            cv2.fillPoly(canvas, [pts_curr],0, 1)
            curr_poly = []
            counter+=1
            check_var = True
            dblclick = False
            click = False
        elif click:
            # TODO: Write code so so that a new point is added to your polygon
            if(check_var == False):
                curr_poly.append([mouseX,mouseY])
            check_var = False
            # (1 mark)
            TODO
            click = False
        else:
            # TODO: Write code so so that last point of the polygon follows
            # your mouse (1 mark)
            if(len(curr_poly) == 0):
                curr_poly.insert(0,[mouseX,mouseY])
            else:
                curr_poly[-1] = [mouseX,mouseY]
            TODO
            print(curr_poly)
        # Draw canvas
        canvas = np.ones((500, 500), dtype=np.float32)

        # TODO: Draw polygon (1 mark)
        TODO
        pts_curr = np.array(curr_poly,np.int32)
        #Draw Polygons in the poly_pt_list
        for x in poly_pt_list:
            pts_polys = np.array(x,np.int32)
            cv2.fillPoly(canvas, [pts_polys],0, 1)

        #Draw current polygon if it exists
        if(len(curr_poly)!= 0):
            cv2.fillPoly(canvas, [pts_curr],0, 1)
        
        # Show canvas
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(10)

        # Deal with keyboard input
        if 113 == key:
            break
        elif 114 == key:
            init()


#
# solution.py ends here
