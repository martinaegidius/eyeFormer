#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:51:26 2022

@author: max
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch 
import os 

#-------------------Script flags----------------------#
DEBUG = True

#-----------------------END>--------------------------#
className = "airplanes"
impath = os.path.dirname(__file__)+"/Data/POETdataset/" #path for image-data
dpath = impath + className #path for saved model-results
impath += "/PascalImages/"


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

def rescale_coords(data):
    #takes imdims and bbox data and rescales. [MUTATES]
    #Returns tuple of two tensors (target,preds)
    imdims = data[-1].squeeze(0).clone().detach()
    target = data[3].squeeze(0).clone().detach()
    preds = data[4].squeeze(0).clone().detach()
    
    target[0::2] = target[0::2]*imdims[-1]
    target[1::2] = target[1::2]*imdims[0]
    preds[0::2] = preds[0::2]*imdims[-1]
    preds[1::2] = preds[1::2]*imdims[0]
    
    #print("Rescaled target: ",target)
    #print("Rescaled preds: ",preds)
    
    return target,preds
    
    

def bbox_for_plot(data):
    """Returns bounding box in format ((x0,y0),w,h)
    Args: 
        input: 
            data: saved list-of-list-of-list containing model-data and image-names. 
                Format: [filename,class,IOU,target,output,size]
        returns: 
            Bounding box coordinates in format ((x0,y0),w,h) for preds and target 
            (target,preds)
    """   
    targets,preds = rescale_coords(data)
    
    #for target
    x0 = targets[0]
    y0 = targets[1]
    w = targets[2] - x0
    h = targets[3] - y0
    #print("x0: ",x0,"y0: ",y0,"w: ",w,"h: ",h)
    target = [x0,y0,w,h]
    
    #for preds 
    x0 = preds[0]
    y0 = preds[1]
    w = preds[2] - x0
    h = preds[3] - y0
    #print("x0: ",x0,"y0: ",y0,"w: ",w,"h: ",h)
    preds = [x0,y0,w,h]
    
    return target,preds
    
    
def plot_overfit_set(data,root_dir,classOC=0):
    """Prints data which is overfit upon
    Args: 
        input: 
            data: saved list-of-list-of-list containing model-data and image-names
                Format: [filename,class,IOU,target,output,size]
            root_dir: imagedir
            classOC: descriptor of class which is used
        returns: 
            None
    """                                           
    NSAMPLES = len(data)
    NCOLS = 2
    fig, ax = plt.subplots(NSAMPLES//NCOLS,NCOLS)
    col = 0
    row = 0
    for entry in range(NSAMPLES):
        filename = data[entry][0][0]
        im = plt.imread(root_dir+filename)
        target,preds = bbox_for_plot(data[entry])
        rectT = patches.Rectangle((target[0],target[1]), target[2], target[3], linewidth=1, edgecolor='r', facecolor='none')
        rectP = patches.Rectangle((preds[0],preds[1]), preds[2], preds[3], linewidth=1, edgecolor='m', facecolor='none')
        
        if entry==NCOLS: #NUMBER OF COLS
            row += 1 
            col = 0 
        print("Col: ",col,"Row:",row)
        ax[row,col].imshow(im)
        ax[row,col].add_patch(rectT)
        ax[row,col].add_patch(rectP)
        
        #plt.text(target[0]+target[2]-10,target[1]+10, "IOU", bbox=dict(facecolor='red', alpha=0.5))
        ax[row,col].text(data[entry][4][0][2].item()-0.1,data[entry][4][0][1].item()+0.1,"{:.2f}".format(data[entry][2][0].item()),bbox=dict(facecolor='magenta', alpha=0.5),transform=ax[row,col].transAxes)
        #ax[row,col].text(preds[2],preds[3],"{:.2f}".format(data[entry][2][0].item()),bbox=dict(facecolor='magenta', alpha=0.5),transform=ax[row,col].transAxes)
        
        col += 1
    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1.125
    fig.legend((rectT,rectP),("Target","Prediction"),loc="upper center",ncol=2,framealpha=0.0)
    plt.show()
    return None


            
            

        

    
testdata = load_results(dpath,className,"test_testresults")
testOnTrain = load_results(dpath,className,"overfit_test_results")
plot_overfit_set(testOnTrain,impath)





       
      