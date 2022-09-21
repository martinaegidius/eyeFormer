#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:58:14 2022

@author: max
"""
import os 


from load_POET import pascalET


DEBUG = True
dset = pascalET()
dset.loadmat_all()
if(DEBUG==True):
    dset.load_images_for_class(8)
    dset.random_sample_plot()

#dset.basic_hists()
#dset.basic_stats()
#dset.specific_plot(9,"2009_003646.jpg") #guy with muddy sofa and TV :-) 