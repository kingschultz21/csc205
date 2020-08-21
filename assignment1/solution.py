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
# Connor Schultz
# V00872923
# Last edit 09/30

# Change Log:
# 
# 
# 
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:

def print_info(line):
    """Parses the line read from text files and prints.

    Parameters
    ----------

    line: list of string

        Each item of the list contains a string that was read from the txt
        file.

    """

    #Store each line as a list of words
    #Tokenize at spaces and use \n to determine end of line
    line_list = []
    curr = ""
    for x in line:
    	curr += x
    	if(x == " "):
    		line_list.append(curr.rstrip(" "))
    		curr = ""
    	elif(x == "\n"):
    		line_list.append(curr.rstrip("\n"))
    		break
    
    # TODO: parse and print!
    #Determines the correct format based on the keyword stored in line_list[0]
    #Once formatted the output is printed
    if(line_list[0] == "canvas"):
    	print("Canvas size is %s x %s"%(line_list[1],line_list[2]))
    elif(line_list[0] == "circle"):
    	print("Circle of thickness %s and radius %s at (%s, %s)"%(line_list[1],line_list[2],line_list[3],line_list[4]))
    elif(line_list[0] == "line"):
    	print("Line of thickness %s from (%s, %s) to (%s, %s)"%(line_list[1],line_list[2],line_list[3],line_list[4],line_list[5]))
    elif(line_list[0] == "polyline"):
    	print("Polygon Line of thickness %s passing through, (%s, %s), (%s, %s), (%s, %s)"%(line_list[1],line_list[2],line_list[3],line_list[4],line_list[5],line_list[6],line_list[7]))
    elif(line_list[0] == "polyfill"):
    	print("Polygon Fill for lines passing through, (%s, %s), (%s, %s), (%s, %s), (%s, %s)"%(line_list[1],line_list[2],line_list[3],line_list[4],line_list[5],line_list[6],line_list[7],line_list[8]))
    else:
    	print("Please enter a valid input")
   

if __name__ == "__main__":

    # TODO: Read file and turn it into a list of strings
    #Open required files
    file_i = open("input.txt","r")
    file_o = open("output.txt","w")

    #read file line by line and store in a list
    lines = []
    for line in file_i.readlines():
    	lines.append(line)
    	file_o.write(line)

    # Print information
    for line in lines:
        print_info(line)

    # TODO: Add another line for drawing a circle at the end at (50, 50) with
    # radius of 20 and thickness 5, and write this to a file
    file_o.write("circle 5 20 50 50\n")

    #close files
    file_o.close()
    file_i.close()
# 
# solution.py ends here
