#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 08:38:04 2022

@author: max
"""
import os 
import scipy 
import numpy as np
import tqdm
import random
import matplotlib.pyplot as plt
from matplotlib import image
import cv2

class pascalET():
    classes = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]
    p = os.path.dirname((__file__))
    matfiles = []
    etData = []
    eyeData = None
    filename = None
    im_dims = None
    chosen_class = None
    NUM_TRACKERS = 5
    ratios = None
    class_counts = None
    pixel_num = None
    fix_nums = None
    
    
    
    def loadmat(self):
        for name in self.classes: 
            self.matfiles.append(self.p+"/Data/POETdataset/etData/"+"etData_"+name+".mat")
        
        for file in self.matfiles: 
            A = scipy.io.loadmat(file,squeeze_me=True,struct_as_record=False)
            self.etData.append(A["etData"])
            #etData[0].fixations[0].imgCoord.fixR.pos
        
    
    def load_images_for_class(self,num):
    #0: aeroplane
    #1: bicycle 
    #2: boat
    #3: cat
    #4: cow 
    #5: diningtable
    #6: dog 
    #7: horse
    #8: motorbike
    #9: sofa
        classdict = {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}    
        self.chosen_class = classdict[num]
        print("Loading class instances for class: ",classdict[num])
        self.eyeData = np.empty((self.etData[num].shape[0],self.NUM_TRACKERS,1),dtype=object)
        self.filename = []
        self.im_dims = []
        for i,j in enumerate(self.etData[num]): #i is image, k is patient
            self.filename.append(self.etData[num][i].filename + ".jpg")
            self.im_dims.append(self.etData[num][i].dimensions)
            #print(j)
            
            for k in range(5):
                #w_max = self.im_dims[i][1] #for removing irrelevant points
                #h_max = self.im_dims[i][0]
                LP = self.etData[num][i].fixations[k].imgCoord.fixL.pos[:]
                RP = self.etData[num][i].fixations[k].imgCoord.fixR.pos[:]
                #remove invalid values . if larger than image-dims or outside of image (negative vals)
                #LP = LP[~np.any((LP[:,]),:]
                BP = np.vstack((LP,RP)) #LP ; RP 
                #eyeData[i,k,0] = [LP,RP] #fill with list values 
                self.eyeData[i,k,0] = BP #fill with matrix
        
        print("Loading complete. Loaded ",self.eyeData.shape[0], " images.")                
      
    def random_sample_plot(self):
        num = random.randint(0,self.eyeData.shape[0])
        path = self.p + "/Data/POETdataset/PascalImages/" +self.chosen_class+"_"+self.filename[num]
        im = image.imread(path)
        plt.title("{}:{}".format(self.chosen_class,self.filename[num]))
        #now for eye-tracking-data
        ax = plt.gca()
        for i in range(self.NUM_TRACKERS):
            mylabel = str(i+1)
            num_fix = int(self.eyeData[num,i][0][:,0].shape[0]/2)
            print(num_fix) #number of fixations on img
            #Left eye
            color = next(ax._get_lines.prop_cycler)['color']
            plt.scatter(self.eyeData[num,i][0][:,0],self.eyeData[num,i][0][:,1],alpha=0.8,label=mylabel,color = color) #wrong - you are not plotting pairwise here, you are plotting R as function of L 
            plt.plot(self.eyeData[num,i][0][0:num_fix,0],self.eyeData[num,i][0][0:num_fix,1],label=str(),color= color)
            plt.plot(self.eyeData[num,i][0][num_fix:,0],self.eyeData[num,i][0][num_fix:,1],label=str(),color = color)
        plt.legend()
        plt.imshow(im)
        
    def basic_stats(self):
        from matplotlib.ticker import PercentFormatter
        #goals: 
            #histogram of num of class instances
            #histogram of aspect ratios 
            #histogram of number of pixels in images (complete and per class)
            #sum of number of fixations on image 
        
        classdict = {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}
        self.class_count = {v: k for k, v in classdict.items()}
        self.ratios = []
        self.num_pixels = []
        
        
        ax = plt.gca()
        for i,j in enumerate(self.classes):
            print("Round: {}".format(i))
            self.load_images_for_class(i)
            #number of class instances:
            self.class_count[j] = self.eyeData.shape[0]
            tmp_ratios = []
            tmp_num_pixels = []
            
            plt.suptitle("Ratios, classwise")
            for k in range(len(self.im_dims)):
                #for stats on complete dataset
                self.ratios.append(self.im_dims[k][1]/self.im_dims[k][0]) #weirdly enough, image-format is stored as height, width 
                self.num_pixels.append(self.im_dims[k][1]*self.im_dims[k][0]) #weirdly enough, image-format is stored as height, width 
                #for stats on class: 
                tmp_ratios.append(self.im_dims[k][1]/self.im_dims[k][0]) #weirdly enough, image-format is stored as height, width 
                tmp_num_pixels.append(self.im_dims[k][1]*self.im_dims[k][0]) #weirdly enough, image-format is stored as height, width 
            
            plt.subplot(2,5,i+1)
            color = next(ax._get_lines.prop_cycler)['color']
            print(color)
            plt.hist(tmp_ratios,bins=30,range=(0,4),color = color)
            plt.title("Class: {}".format(j))
                
        
        plt.show()
        print(self.class_count)
        print(len(self.ratios))
        print(len(self.num_pixels))
        plt.figure(200)
        plt.hist(self.ratios,bins=100)
        plt.show()
        
        
            
            
            #self.load_images_for_class()
        
        
      
dset = pascalET()
dset.loadmat()
if("DEBUG"==True):
    dset.load_images_for_class(9)
    dset.random_sample_plot()

dset.basic_stats()