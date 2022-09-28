#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:13:32 2022

@author: max
"""

from load_POET import pascalET
import os 
from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np

class AirplanesBoatsDataset(Dataset):
    def __init__(self,pascalobj,root_dir,classes):
        self.ABDset = generate_DS(pascalobj,classes)
        self.root_dir = root_dir
    
    def __len__(self): 
        return len(self.ABDset)
    
    
    def __getitem__(self,idx): #for images
        if torch.is_tensor(idx):
            idx = idx.tolist() #for indexing
        
        #PROBABLY THE PROBLEM IS YOU ARE SETTING THE VARS FROM NUMPY "OBJECT"-TYPE
        filename = self.ABDset[idx,0]
        bbox = self.ABDset[idx,1].astype(np.int8) #earlier issue: cant convert type np.uint8
        objClass = self.ABDset[idx,2]
        signals = self.ABDset[idx,3:]
        
        sample = {"filename": filename,"bbox": bbox,"class": objClass,"signals": signals}
        return sample

"""FOR IMAGES
    def __getitem__(self,idx): #for images
        if torch.is_tensor(idx):
            idx = idx.tolist() #for indexing
        
        classdict = {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}
        className = classdict[self.ABDset[idx,2]]+"_"
        img_name = self.root_dir+className+ self.ABDset[idx,0]
        print(img_name)
"""
    
    
    
def generate_DS(dataFrame,classes = [x for x in range(10)]):
    """ Note: as implemented now works for two classes only
    Args
        classes: list of input-classes of interest {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}   
        If none is provided, fetch all 9 classes
        
        returns: 
            A tensor which contains: 1. image-filenames, 2. bounding box, 3. class number, 4:8: a list of processed cleaned both-eye-fixations
    """
    dataFrame.loadmat()
    dataFrame.convert_eyetracking_data(CLEANUP=True,STATS=True)
    
    dslen = 0
    for CN in classes: 
        dslen += len(dataFrame.etData[CN])
    
    df = np.empty((dslen,8),dtype=object) #probably the issue
    
    datalist = []
    for CN in classes:
        datalist.append(dataFrame.etData[CN])
    
    for i in range(dslen):
        if(i<len(datalist[0])):
            df[i,0] = [dataFrame.etData[classes[0]][i].filename+".jpg"]
            df[i,1] = dataFrame.etData[classes[0]][i].gtbb
            df[i,2] = classes[0]
            for j in range(dataFrame.NUM_TRACKERS):
                df[i,3+j] = dataFrame.eyeData[classes[0]][i][j][0] #list items
                
        else: 
            nextIdx = i-len(datalist[0])
            df[i,0] = [dataFrame.etData[classes[1]][nextIdx].filename+".jpg"]
            df[i,1] = dataFrame.etData[classes[1]][nextIdx].gtbb
            df[i,2] = classes[1]
            for j in range(dataFrame.NUM_TRACKERS):
                df[i,3+j] = dataFrame.eyeData[classes[1]][nextIdx][j][0]
            
        
        
    print("Total length is: ",dslen)
    return df
    
dataFrame = pascalET()
root_dir = os.path.dirname(__file__) + "/Data/POETdataset/PascalImages/"

#DF  = generate_DS(dataFrame,[0,9])
    
airplanesBoats = AirplanesBoatsDataset(dataFrame, root_dir, [0,9])


"""For debugging when object-issue is fixed
dataloader = DataLoader(airplanesBoats,batch_size=4,shuffle=True,num_workers=0)
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch,sample_batched)
    
    if i_batch==10:
        break
    
