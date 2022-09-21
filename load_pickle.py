#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:40:01 2022

@author: max
"""

import pickle 
import os
import load_POET

filename = os.path.dirname(__file__)+"/Data/POETdataset/PascalImages/Resized/resized_pascalET.obj"
filehandler = open(filename, 'rb') #raw binary, mister
dset = pickle.load(filehandler)


dset.specific_plot(8,dset.filename[0],resized=True)