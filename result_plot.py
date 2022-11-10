#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:11:55 2022

@author: max
"""

import torch 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from matplotlib import rc,rcParams
"""
#-------------------------Learning-curves-plotting script for all classes---------------------------------#

Usage: first scp all run-results
cp folder-contens of nL_1_nH_2 to inst folder of interest
then simply run

#Todo: cp cat, cow, diningtable, dog, horse, motorbike and sofa as follows
        scp -r s194119@login1.gbar.dtu.dk:BA/eyeFormer/Data/POETdataset/boat/nL_1_nH_2 /home/max/Documents/s194119/Bachelor/Results/1_fold_results/boat/

Additionally: you ran it without dropout -.- 

#---------------------------------------------------------------------------------------------------------#
"""

FSZ = 18
lFSZ = 16
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}

plt.rcParams.update(params)

sns.set_theme()
root_path = os.path.dirname(__file__)+"/Data/POETdataset/"
classes = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]

epochDict = {"aeroplane":[-1,-1],"bicycle":[-1,-1],"boat":[-1,-1],"cat":[-1,-1],
             "cow":[-1,-1],"diningtable":[-1,-1],"dog":[-1,-1],"horse":[-1,-1],
             "motorbike":[-1,-1],"sofa":[-1,-1]} #make a dict for storing data for optimal number of epochs. 

pwds = os.path.dirname(__file__)+"/Results/1_fold_results_nl_6_nh_2/"
if not os.path.exists(pwds):
    os.mkdir(pwds)

#for inst in classes:
for inst in classes:
    path = os.path.dirname(__file__)+"/Results/1_fold_results_nl_6_nh_2/"+inst+"/nL_6_nH_2/"
    train_loss = torch.load(path+inst+"_train_losses.pth")
    val_loss = torch.load(path+inst+"_val_losses.pth")
    minVal = min(val_loss)
    idx = val_loss.index(minVal)
    epochDict[inst][0] = idx
    epochDict[inst][1] = minVal
    
    fig,axes = plt.subplots(1,2,figsize=(16,7),sharex=True)
    axes[0].plot(train_loss)
    axes[0].plot(val_loss)
    axes[0].plot(idx,minVal,marker='x',markersize=11,markerfacecolor="red",markeredgewidth = 2,markeredgecolor="red",linestyle="None")
    axes[0].set_xlabel("Epochs",fontsize=FSZ,fontweight="bold")
    axes[0].set_ylabel("L1-Loss",fontsize=FSZ,fontweight="bold")
    axes[0].tick_params(axis='both', which='major', labelsize=13)
    plt.xlim([0,2000])
    axes[1].semilogy(train_loss)
    axes[1].semilogy(val_loss)
    axes[1].set_xlabel("Epochs",fontsize=FSZ,fontweight="bold")
    axes[1].set_ylabel("Logarithmic L1-Loss",fontsize=FSZ,fontweight="bold")
    axes[1].semilogy(idx,minVal,marker='x',markersize=11,markerfacecolor="red",markeredgewidth = 2,markeredgecolor="red",linestyle="None")
    axes[1].tick_params(axis='both', which='major', labelsize=13)
    plt.xlim([0,2000])
    
    leg = fig.legend(["Train","Validation","Chosen epoch"],loc="upper center",ncol=3,bbox_to_anchor=(0.5, 0.96),
              bbox_transform=fig.transFigure,fontsize=lFSZ) #prop={'size': 13}
    #fig.legend(fontsize=FSZ)
    for line in leg.get_lines():
        line.set_linewidth(6.0)
    
    fig.suptitle(inst,size=FSZ-2)#,y=1.2)
    
    

    plt.savefig(pwds+inst+"_learning_curves.pdf")
np.save(path+"../../"+"optimal_no_epochs.npy",epochDict) #save number of epochs to file 

