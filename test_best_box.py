#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:02:42 2022

@author: max
"""

import os 


from load_POET import pascalET
import numpy as np


DEBUG = False
DEBUG_CLEANUP = False
dset = pascalET()
dset.loadmat()


#debug new format 
BP = dset.convert_eyetracking_data(CLEANUP=True,STATS=True)
dset.load_images_for_class(0)
dset.specific_plot(0,"2009_004601.jpg")
idx = dset.filename.index("2009_004601.jpg") #remember - this only works with current filename structure. Right now, filenames are saved when images are loaded
#for debugging best bbox: following image has two boxes
testImB = dset.etData[0][17].gtbb
testBP = dset.debug_box_BP #WHY IS IT SO BIG? 

print("Check number of detected fixes in box: ")
try:
    print("Number of fixes in general: ",dset.eyeData_stats[0,idx,:,0])
    print("Number of fixes in bbox: ", dset.eyeData_stats[0,idx,:,1])
    #seems right, but IT DOES NOT CHECK OTHER BOX!
except:
    pass