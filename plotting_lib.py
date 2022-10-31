#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:51:26 2022

@author: max
"""

import matplotlib.pyplot as plt
import torch 
import os 

#-------------------Script flags----------------------#
DEBUG = True

#-----------------------END>--------------------------#
className = "airplanes"
path = os.path.dirname(__file__)+"/Data/POETdataset/"+className


#/home/max/Documents/s194119/Bachelor/Data/POETdataset/airplanes/airplanes_test_testresults.pth"

def load_results(root_dir,className,file_specifier):
    path = root_dir + "/" + className+"_"+file_specifier+".pth"
    if(DEBUG):
        print("path is: ",path)
    try: 
        entrys = torch.load(path)
        print("...Success loading...")
        return entrys
    except: 
        print("Error loading file. Check filename/path: ",path)
        return None
    
data = load_results(path,className,"test_testresults")

       
      