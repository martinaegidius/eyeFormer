#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:11:55 2022

@author: max
"""

import torch 
import os 
import matplotlib.pyplot as plt
import math

path = os.path.dirname(__file__)+"/Data/POETdataset/airplanes/experiment1/"

train_loss = torch.load(path+"airplanes_train_losses.pth")
val_loss = torch.load(path+"airplanes_val_losses.pth")

minVal = min(val_loss)
idx = val_loss.index(minVal)

plt.semilogy(train_loss)
plt.semilogy(val_loss)
plt.semilogy(idx,minVal,marker='x',markersize=8,markerfacecolor="red",markeredgecolor="red")
plt.legend(["Train","Validation","Chosen epoch"])



