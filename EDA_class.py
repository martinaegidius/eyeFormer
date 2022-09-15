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

DEBUG = True

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
    classwise_num_pixels = None
    classwise_ratios = None
    bbox = None
    
    
    
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
        self.bbox = []
        for i,j in enumerate(self.etData[num]): #i is image, k is patient
            self.filename.append(self.etData[num][i].filename + ".jpg")
            self.im_dims.append(self.etData[num][i].dimensions)
            #print(j)
            self.bbox.append(self.etData[num][i].gtbb)
            
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
        from matplotlib.patches import Rectangle
        #random.seed(18)
        num = random.randint(0,self.eyeData.shape[0])
        path = self.p + "/Data/POETdataset/PascalImages/" +self.chosen_class+"_"+self.filename[num]
        im = image.imread(path)
        plt.title("{}:{}".format(self.chosen_class,self.filename[num]))
        #now for eye-tracking-data
        
        ax = plt.gca()
        global x 
        global y
        global lol
        lol = self.bbox[num]
        xy,w,h = self.get_bounding_box(self.bbox[num])
        rect = Rectangle(xy,w,h,linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        for i in range(self.NUM_TRACKERS):
            color = next(ax._get_lines.prop_cycler)['color']
            mylabel = str(i+1)
            num_fix = int(self.eyeData[num,i][0][:,0].shape[0]/2)
            print(num_fix) #number of fixations on img
            #Left eye
            plt.scatter(self.eyeData[num,i][0][:,0],self.eyeData[num,i][0][:,1],alpha=0.8,label=mylabel,color = color) #wrong - you are not plotting pairwise here, you are plotting R as function of L 
            plt.plot(self.eyeData[num,i][0][0:num_fix,0],self.eyeData[num,i][0][0:num_fix,1],label=str(),color= color)
            plt.plot(self.eyeData[num,i][0][num_fix:,0],self.eyeData[num,i][0][num_fix:,1],label=str(),color = color)
        plt.legend()
        plt.imshow(im)
        
    def specific_plot(self,classOC,filename):
        from matplotlib.patches import Rectangle
        classdict = {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}    
        self.chosen_class = classdict[classOC]
        
        try: 
            idx = self.filename.index(filename) #get index of requested file
        except:
            print("Filename not found in loaded data. Did you type correctly?")
            return
        
        
        path = self.p + "/Data/POETdataset/PascalImages/" +self.chosen_class+"_"+filename
        fig = plt.figure(3201)
        im = image.imread(path)
        plt.title("{}:{}".format(self.chosen_class,filename))
        #now for eye-tracking-data
        
        ax = plt.gca()
        lol = self.bbox[idx]
        xy,w,h = self.get_bounding_box(self.bbox[idx])
        rect = Rectangle(xy,w,h,linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        for i in range(self.NUM_TRACKERS):
            color = next(ax._get_lines.prop_cycler)['color']
            mylabel = str(i+1)
            num_fix = int(self.eyeData[idx,i][0][:,0].shape[0]/2)
            print(num_fix) #number of fixations on img
            #Left eye
            plt.scatter(self.eyeData[idx,i][0][:,0],self.eyeData[idx,i][0][:,1],alpha=0.8,label=mylabel,color = color) #wrong - you are not plotting pairwise here, you are plotting R as function of L 
            plt.plot(self.eyeData[idx,i][0][0:num_fix,0],self.eyeData[idx,i][0][0:num_fix,1],label=str(),color= color)
            plt.plot(self.eyeData[idx,i][0][num_fix:,0],self.eyeData[idx,i][0][num_fix:,1],label=str(),color = color)
        plt.legend()
        plt.imshow(im)
        
        
    def get_bounding_box(self,inClass):
        #convert to format for patches.Rectangle; it wants anchor point (upper left), width, height
        
        # x = np.empty((inClass.shape[-1]//2)+2) #multiple bboxes may be present
        # y = np.empty((inClass.shape[-1]//2)+2) #multiple bboxes may be present
        # assert x.shape[0]==4
        if(inClass.ndim>1):
            inClass = inClass[0]
        # x[:2] = inClass[0::2]
        # x[2:] = inClass[::-1][-3::2]
        # y[:2] = inClass[1]
        # y[2:4] = inClass[-1]
        xy = (inClass[0],inClass[1])
        w = inClass[2]-inClass[0]
        h = inClass[-1]-inClass[1]
        #y = [inClass[1],inClass[-1]]
        return xy,w,h
        
    def basic_hists(self):
        #goals: 
            #histogram of num of class instances[done]
            #histogram of aspect ratios [done]
            #histogram of number of pixels in images (complete and per class) [done]
            #means and variances histograms of both ratio and number of pixels 
            #sum of number of fixations on image 
        
        classdict = {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}
        self.class_count = {v: k for k, v in classdict.items()}
        self.ratios = []
        self.num_pixels = []
        color_list = [] #for saving same color per class as list format
        
        self.classwise_num_pixels = []
        self.classwise_ratios = []
        #RATIOS
        ax = plt.gca()
        plt.figure(figsize=(1920/160, 1080/160), dpi=160)
        for i,j in enumerate(self.classes):
            print("Round: {}".format(i))
            self.load_images_for_class(i)
            #number of class instances:
            self.class_count[j] = self.eyeData.shape[0]
            tmp_ratios = []
            tmp_num_pixels = []
            #self.classwise_ratios.append([]) #list of lists
            #self.classwise_num_pixels.append([])
            plt.suptitle("Ratios, classwise")
            for k in range(len(self.im_dims)):
                #for stats on complete dataset
                self.ratios.append(self.im_dims[k][1]/self.im_dims[k][0]) #weirdly enough, image-format is stored as height, width 
                self.num_pixels.append(self.im_dims[k][1]*self.im_dims[k][0]) #weirdly enough, image-format is stored as height, width 
                #for stats on class: 
                tmp_ratios.append(self.im_dims[k][1]/self.im_dims[k][0]) #weirdly enough, image-format is stored as height, width 
                tmp_num_pixels.append(self.im_dims[k][1]*self.im_dims[k][0]) #weirdly enough, image-format is stored as height, width 
            self.classwise_ratios.append(tmp_ratios)
            self.classwise_num_pixels.append(tmp_num_pixels)
            plt.subplot(2,5,i+1)
            color = next(ax._get_lines.prop_cycler)['color']
            color_list.append(color)
            plt.hist(tmp_ratios,bins=30,range=(0,4),color = color)
            plt.title("Class: {}".format(j))    
        
        plt.savefig("/home/max/Documents/s194119/Bachelor/Graphs/ratios-classwise.pdf",format="pdf",dpi=160)
        
        
        #overall data-set image-ratios
        plt.figure(200)
        plt.hist(self.ratios,bins=100)
        plt.savefig("/home/max/Documents/s194119/Bachelor/Graphs/ratios-overall.pdf",format="pdf",dpi=160)
    
        #number of class-instances:
        plt.figure(num=199,figsize=(1920/160,1080/160),dpi=160)
        plt.bar(self.class_count.keys(),self.class_count.values(),color=color_list)
        plt.savefig("/home/max/Documents/s194119/Bachelor/Graphs/class-balance.pdf",format="pdf",dpi=160)
        
        #total ds pixelnumbers
        plt.figure(201)
        plt.hist(self.num_pixels,bins=15)
        plt.savefig("/home/max/Documents/s194119/Bachelor/Graphs/overall_num_pixels.pdf",format="pdf",dpi=160)
        
        #classwise pixelnumbers (same loop as earlier for ratios)
        ax = plt.gca()
        plt.figure(num=202,figsize=(1920/160, 1080/160), dpi=160)
        plt.suptitle("Total number of pixels, classwise")
        for i,j in enumerate(self.classes):
            plt.subplot(2,5,i+1)
            color = next(ax._get_lines.prop_cycler)['color']
            plt.hist(self.classwise_num_pixels[i],color=color)
            plt.title("Class: {}".format(j))    
        plt.savefig("/home/max/Documents/s194119/Bachelor/Graphs/classwise_numpx.pdf",format="pdf",dpi=160)
            
        
    def basic_stats(self): #cannot run w/o basic_hist()
        #show means and variances of 1: ratios, 2: pixel-numbers
        ratio_means = []
        ratio_var = []
        for i in range(len(self.classwise_ratios)): 
            ratio_means.append(sum(self.classwise_ratios[i])/len(self.classwise_ratios[i])) #(looks a bit funny because list of lists)
            ratio_var.append(np.var(self.classwise_ratios[i]))
        num_px_means = []
        num_px_var = []
        for i in range(len(self.classwise_num_pixels)):
            num_px_means.append(sum(self.classwise_num_pixels[i])/len(self.classwise_num_pixels[i])) #(looks a bit funny because list of lists)
            num_px_var.append(np.var(self.classwise_num_pixels[i])) #values are huge, but when looking at mean it seems to be quite fine
            
        print("Classwise ratio means: \n",ratio_means)
        print("Classwise ratio vars: \n",ratio_var)
        print("Classwise num_px means: \n",num_px_means)
        print("Classwise num_px vars: \n",num_px_var)
        
        
        
        
        

dset = pascalET()
dset.loadmat()
if(DEBUG==True):
    dset.load_images_for_class(9)
    dset.random_sample_plot()

#dset.basic_hists()
#dset.basic_stats()
dset.specific_plot(9,"2009_003646.jpg")