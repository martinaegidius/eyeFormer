#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 07:55:29 2022

@author: max
"""

import numpy as np 

BP = np.array([[np.nan,   np.nan],
       [ 186.9,   83. ],
       [ 295.3,  145.1],
       [ 215.4,  180.7],
       [ 250,  401],
       [ 220. ,  178.7],
       [ 156.1,  159.4],
       [-130.6,    2.8],
       [ 182.1,   81. ],
       [ 284.2,  147.7],
       [ 402.1,  177.6],
       [ 168.4,  168.2],
       [ 177.5,  134.7]])


BP = np.array([[   np.nan,    np.nan],
       [ 281.8, -104.3]])

bbx = [200,100,200,100] #bbox starts in (200,100) and reaches to (250,150)

tmpBP = np.delete(BP,np.where(np.isnan(BP[0,:])),axis=0) #delete NANs
tmpBP = np.delete(tmpBP,np.where(np.isnan(BP[1,:])),axis=0) #delete NANs
tmpBP = np.delete(tmpBP,np.where((tmpBP[:,0]<=bbx[0])),axis=0) #left constraint
tmpBP = np.delete(tmpBP,np.where((tmpBP[:,1]<=bbx[1])),axis=0) #up constraint
tmpBP = np.delete(tmpBP,np.where((tmpBP[:,0]>=bbx[0]+bbx[2])),axis=0) #right constraint #only keep between 200 and 400
tmpBP = np.delete(tmpBP,np.where((tmpBP[:,1]>=bbx[1]+bbx[3])),axis=0) #only keep between 100 and 200

print(tmpBP)