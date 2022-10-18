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
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import transforms, ops
import math


class AirplanesBoatsDataset(Dataset):
    def __init__(self,pascalobj,root_dir,classes,transform=None):
        """old init: 
            def __init__(self,pascalobj,root_dir,classes):#,eyeData,filenames,targets,classlabels,root_dir,classes):
                self.ABDset = generate_DS(pascalobj,classes)
                self.eyeData = eyeData
                self.filenames = filenames
                self.root_dir = root_dir
                self.targets = targets #bbox
                self.classlabels = classlabels
        """

        self.ABDset,self.filenames,self.eyeData,self.targets,self.classlabels,self.imdims,self.length = generate_DS(pascalobj,classes)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self): 
        return len(self.ABDset)
    
    
    def __getitem__(self,idx): #for images
        if torch.is_tensor(idx):
            idx = idx.tolist() #for indexing
        
        SINGLE_PARTICIPANT = True
        
        #PROBABLY THE PROBLEM IS YOU ARE SETTING THE VARS FROM NUMPY "OBJECT"-TYPE
        #filename = self.ABDset[idx,0]
        filename = self.filenames[idx]
        #target = self.ABDset[idx,1].astype(np.int16) #earlier issue: cant convert type np.uint8
        #objClass = self.ABDset[idx,2]
        #signals = self.ABDset[idx,3:]
       
        if(SINGLE_PARTICIPANT==True):
            signals = self.eyeData[idx][0] #participant zero only
        
        else: 
            signals = self.eyeData[idx,:]
        
        targets = self.targets[idx,:]
        
        sample = {"signal": signals,"target": targets,"file": filename, "index": idx,"size":self.imdims[idx],"class":self.classlabels[idx],"mask":None}
        
        if self.transform: 
            sample = self.transform(sample)
        
        return sample
        
                

def rescale_coordsO(eyeCoords,imdims,bbox):
    """
    Args: 
        eyeCoords: Tensor
        imdims: [1,2]-tensor with [height,width] of image
        imW: int value with width of image 
        imH: int value of height of image 
    Returns: 
        Rescaled eyeCoords (inplace) from [0,1].
    """
    outCoords = torch.zeros_like(eyeCoords)
    outCoords[:,0] = eyeCoords[:,0]/imdims[1]
    outCoords[:,1] = eyeCoords[:,1]/imdims[0]
    
    tbox = torch.zeros_like(bbox)
    #transform bounding-box to [0,1]
    tbox[0] = bbox[0]/imdims[1]
    tbox[1] = bbox[1]/imdims[0]
    tbox[2] = bbox[2]/imdims[1]
    tbox[3] = bbox[3]/imdims[0]
    
    
    return outCoords,tbox
   
class tensor_pad(object):
    """
    Object-function which pads a batch of different sized tensors to all have [bsz,32,2] with value -999 for padding

    Parameters
    ----------
    object : lambda, only used for composing transform.
    sample : with a dict corresponding to dataset __getitem__

    Returns
    -------
    padded_batch : tensor with dimensions [batch_size,32,2]

    """
    
    def __call__(self,sample):
        batch_size = sample[""]
        padded_batch = torch.zeros(batch_size,32,2)
        for batch in range(batch_size): #if batch-first == True
            padded_batch[batch] = torch.nn.functional.pad(sample["signal"][batch],pad=(0,32-sample["signal"][batch].shape[0],0),mode="constant",value=0.0)
        #padded_batch = torch.nn.functional.pad(sample["signal"][batch],pad=(0,32-sample["signal"][batch].shape[0],0),mode="constant",value=0.0)
        paddedSample = m(sample["signal"])

        return sample
class tensorPad(object):
    def __call__(self,sample):
        x = sample["signal"]
        xtmp = torch.zeros(32,2)
        numel = x.shape[0]
        xtmp[:numel,:] = x[:,:]
        
        maskTensor = (xtmp[:,0]==0)    
        sample["signal"] = xtmp
        sample["mask"] = maskTensor
        return sample
        
class rescale_coords(object):
    """
    Args: 
        eyeCoords: Tensor
        imdims: [1,2]-tensor with [height,width] of image
        imW: int value with width of image 
        imH: int value of height of image 
    Returns: 
        Rescaled eyeCoords (inplace) from [0,1].
    """
    
    def __call__(self,sample):
        eyeCoords,imdims,bbox = sample["signal"],sample["size"],sample["target"]
        mask = sample["mask"]
        outCoords = torch.ones(32,2)
        
        #old ineccesary #eyeCoords[inMask] = float('nan') #set nan for next calculation -> division gives nan. Actually ineccesary
        #handle masking
        #inMask = (eyeCoords==0.0) #mask-value
        #calculate scaling
        outCoords[:,0] = eyeCoords[:,0]/imdims[1]
        outCoords[:,1] = eyeCoords[:,1]/imdims[0]
        
        #reapply mask
        outCoords[mask] = 0.0 #reapply mask
        
        tbox = torch.zeros_like(bbox)
        #transform bounding-box to [0,1]
        tbox[0] = bbox[0]/imdims[1]
        tbox[1] = bbox[1]/imdims[0]
        tbox[2] = bbox[2]/imdims[1]
        tbox[3] = bbox[3]/imdims[0]
        
        sample["signal"] = outCoords
        sample["target"] = tbox
        
        return sample
        #return {"signal": outCoords,"size":imdims,"target":tbox}
    
    
    


"""FOR IMAGES
    def __getitem__(self,idx): #for images
        if torch.is_tensor(idx):
            idx = idx.tolist() #for indexing
        
        classdict = {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}
        className = classdict[self.ABDset[idx,2]]+"_"
        img_name = self.root_dir+className+ self.ABDset[idx,0]
        print(img_name)
"""
    
    
    
def generate_DS(dataFrame,classes=[x for x in range(10)]):
    """ Note: as implemented now works for two classes only
    Args
        classes: list of input-classes of interest {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}   
        If none is provided, fetch all 9 classes
        
        returns: 
            A tensor which contains: 1. image-filenames, 2. bounding box, 3. class number, 4:8: a list of processed cleaned both-eye-fixations
            A list of filenames
            A tensor which contains all eyetracking data
            A tensor containing bounding boxes
            A tensor containing numerated classlabels
            A tensor containing image-dimensions
    """
    dataFrame.loadmat()
    dataFrame.convert_eyetracking_data(CLEANUP=True,STATS=True)
    
    dslen = 0
    for CN in classes: 
        dslen += len(dataFrame.etData[CN])
    
    df = np.empty((dslen,8),dtype=object)
    
    #1. fix object type by applying zero-padding
    num_points = 32
    eyes = torch.zeros((dslen,dataFrame.NUM_TRACKERS,num_points,2))
    #2.saving files to list
    filenames = []
    #3: saving target in its own array
    targets = np.empty((dslen,4))
    #classlabels and imdims to own array
    classlabels = np.empty((dslen))
    imdims = np.empty((dslen,2))
    
    
    
    datalist = []
    for CN in classes:
        datalist.append(dataFrame.etData[CN])
    
    for i in range(dslen):
        if(i<len(datalist[0])):
            df[i,0] = [dataFrame.etData[classes[0]][i].filename+".jpg"]
            filenames.append(dataFrame.classes[classes[0]]+"_"+dataFrame.etData[classes[0]][i].filename+".jpg")
            df[i,1] = dataFrame.etData[classes[0]][i].gtbb
            df[i,2] = classes[0]
            targets[i] =  dataFrame.etData[classes[0]][i].gtbb
            classlabels[i] = classes[0]
            imdims[i] = dataFrame.etData[classes[0]][i].dimensions
            for j in range(dataFrame.NUM_TRACKERS):
                sliceShape = dataFrame.eyeData[classes[0]][i][j][0].shape
                df[i,3+j] = dataFrame.eyeData[classes[0]][i][j][0] #list items
                if(sliceShape != (2,0)): #for empty
                    eyes[i,j,:sliceShape[0],:] = torch.from_numpy(dataFrame.eyeData[classes[0]][i][j][0][-32:])
                else: 
                    eyes[i,j,:,:] = 0.0
                    print("error-filled measurement [entryNo,participantNo]: ",i,",",j)
                    print(eyes[i,j])
                
                    
                #old: padding too early #eyes[i,j] = zero_pad(dataFrame.eyeData[classes[0]][i][j][0],num_points,-999) #list items
                
                
        else: 
            nextIdx = i-len(datalist[0])
            df[i,0] = [dataFrame.etData[classes[1]][nextIdx].filename+".jpg"]
            filenames.append(dataFrame.classes[classes[1]]+"_"+dataFrame.etData[classes[1]][nextIdx].filename+".jpg")
            df[i,1] = dataFrame.etData[classes[1]][nextIdx].gtbb
            targets[i] =  dataFrame.etData[classes[1]][nextIdx].gtbb
            df[i,2] = classes[1]
            classlabels[i] = classes[1]
            imdims[i] = dataFrame.etData[classes[1]][nextIdx].dimensions
            for j in range(dataFrame.NUM_TRACKERS):
                sliceShape = dataFrame.eyeData[classes[1]][nextIdx][j][0].shape
                df[i,3+j] = dataFrame.eyeData[classes[1]][nextIdx][j][0]
                if(sliceShape != (2,0)): #for empty
                    eyes[i,j,:sliceShape[0],:] = torch.from_numpy(dataFrame.eyeData[classes[1]][nextIdx][j][0][-32:]) #last 32 points
                else:
                    eyes[i,j,:,:] = 0.0
                
                #old: padding too early #eyes[i,j] = zero_pad(dataFrame.eyeData[classes[1]][nextIdx][j][0],num_points,-999) #list items
                
            
        
        
    print("Total length is: ",dslen)
    
    targetsTensor = torch.from_numpy(targets)
    classlabelsTensor = torch.from_numpy(classlabels)
    imdimsTensor = torch.from_numpy(imdims)
    
    return df,filenames,eyes,targetsTensor,classlabelsTensor,imdimsTensor,dslen
   


def zero_pad(inArr: np.array,padto: int,padding: int):
    """
    Args:
        inputs: 
            inArr: array to pad
            padto: integer value of the array-size used. Uses last padto elements of eye-signal
            padding: integer value of padding value, defaults to zero
            
        output: 
            torch.tensor of dtype torch.float32 with size [padto,2]
        
    """
    if(inArr.shape[0]>padto):
        return torch.from_numpy(inArr[:padto]).type(torch.float32)
        
    else: 
        outTensor = torch.ones((padto,2)).type(torch.float32)*padding #[32,2]
        if(inArr.shape[1]==0): #Case: no valid entries in eyeData
            return outTensor
        
        numEntries = inArr.shape[0]
        outTensor[:numEntries,:] = torch.from_numpy(inArr[-numEntries:,:]) #get all
        return outTensor
    
    
dataFrame = pascalET()
root_dir = os.path.dirname(__file__) + "/Data/POETdataset/PascalImages/"

"""Old implementation, corresponding to commented part in dataset-init:
#DF,FILES,EYES,TARGETS,CLASSLABELS, IMDIMS = generate_DS(dataFrame,[0,9]) #init DSET, extract interesting stuff as tensors. Aim for making DS obsolete and delete
#airplanesBoats = AirplanesBoatsDataset(dataFrame, EYES, FILES, TARGETS,CLASSLABELS, root_dir, [0,9]) #init dataset as torch.Dataset.
"""

classesOC = [0,9]
composed = transforms.Compose([transforms.Lambda(tensorPad()),
                               transforms.Lambda(rescale_coords())]) #transforms.Lambda(tensor_pad() also a possibility, but is probably unecc.

airplanesBoats = AirplanesBoatsDataset(dataFrame, root_dir, classesOC,transform=composed) #init dataset as torch.Dataset.



#"""
#check that transforms work 

#define split ratio
test_size = int(0.2*airplanesBoats.length)
train_size = airplanesBoats.length-test_size

#set seed 
torch.manual_seed(1)
CHECK_BALANCE = False

#get split 
train,test = torch.utils.data.random_split(airplanesBoats,[train_size,test_size])
###load split from data

split_root_dir = os.path.dirname(__file__)

def get_split(root_dir,classesOC):
    """Args: 
        root_dir, ie. path to main-project-folder
        classesOC: List of classes of interest
        
       Returns:
        list of filenames for training
        list of filenames for testing
    """
        
    split_dir = root_dir +"/Data/POETdataset/split/"
    classlist = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]

    training_data = []
    for classes in classesOC: #loop saves in list of lists format
        train_filename = split_dir + classlist[classes]+"_Rbbfix.txt"
        with open(train_filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        for i in range(len(lines)): 
            lines[i] = classlist[classes]+"_"+lines[i]+".jpg"
            
        training_data.append(lines)
        
    test_data = []
    for classes in classesOC: 
        train_filename = split_dir + classlist[classes]+"_Rfix.txt"
        with open(train_filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        for i in range(len(lines)): 
            lines[i] = classlist[classes]+"_"+lines[i]+".jpg"
            
        test_data.append(lines)
    
    #unpack lists of lists to single list for train and for test
    train = []
    test = []
    for i in range(len(training_data)):
        train = [*train,*training_data[i]]
        test = [*test,*test_data[i]]        
         
    return train,test

trainLi,testLi = get_split(split_root_dir,classesOC) #get Dimitrios' list of train/test-split.
trainIDX = [airplanesBoats.filenames.index(i) for i in trainLi] #get indices of corresponding entries in data-structure
testIDX = [airplanesBoats.filenames.index(i) for i in testLi]


#subsample based on IDX
train = torch.utils.data.Subset(airplanesBoats,trainIDX)
test = torch.utils.data.Subset(airplanesBoats,testIDX)



#make dataloaders of chosen split
BATCH_SZ = 1

trainloader = DataLoader(train,batch_size=BATCH_SZ,shuffle=True,num_workers=0)
testloader = DataLoader(test,batch_size=BATCH_SZ,shuffle=True,num_workers=0)

#check class-balance (only necessary for finding appropriate seed ONCE): 
if(CHECK_BALANCE==True):
    countDict = {"trainplane":0,"trainsofa":0,"testplane":0,"testsofa":0}
    for i_batch, sample_batched in enumerate(trainloader):
        for j in range(sample_batched["class"].shape[0]):
            if sample_batched["class"][j]==0:
                countDict["trainplane"] += 1 
            if sample_batched["class"][j]==9:
                countDict["trainsofa"] += 1
            
    for i_batch, sample_batched in enumerate(testloader):
        for j in range(sample_batched["class"].shape[0]):
            if sample_batched["class"][j]==0:
                countDict["testplane"] += 1 
            if sample_batched["class"][j]==9:
                countDict["testsofa"] += 1
        
    
    print("Train-set: (airplanes, sofas) (",countDict["trainplane"]/train_size,"),(",countDict["trainsofa"]/train_size,")")
    print("Test-set: (airplanes, sofas) (",countDict["testplane"]/test_size,"),(",countDict["testsofa"]/test_size,")")
    print("Seems quite balanced :-)")


"""
Model definitions: 
and build

"""    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_square_subsequent_mask(sz: int) -> torch.Tensor: 
    return torch.triu(torch.ones(sz,sz)*float('-inf'),diagonal=1)

#define transformer model. Goal: overfitting

class eyeFormer_baseline(nn.Module):
        def __init__(self,input_dim=2,hidden_dim=2048,output_dim=4,dropout=0.1):
            self.d_model = input_dim
            super(eyeFormer_baseline,self).__init__() #get class instance
            self.pos_encoder = PositionalEncoding(input_dim,dropout)
            encoderLayers = nn.TransformerEncoderLayer(d_model=input_dim,nhead=2,dim_feedforward=hidden_dim,dropout=dropout,activation="relu",batch_first=True) #False due to positional encoding made on batch in middle
            #make encoder - 3 pieces
            self.cls_token = nn.Parameter(torch.zeros(1,1))
            self.transformer_encoder = nn.TransformerEncoder(encoderLayers,num_layers = 3)
            #self.decoder = nn.Linear(64,4,bias=True) 
            self.clsdecoder = nn.Linear(2,4,bias=True)
        
            
        def forward(self,x,src_padding_mask):
            """#src_key_padding_mask: 
            if torch.cuda.is_available==True: #if batchsize 1 we need to unsqueeze
                src_padding_mask = (x[:,:,0]==0).cuda()#.reshape(x.shape[1],x.shape[0]) #- src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
            else:
                src_padding_mask = (x[:,:,0]==0)#.reshape(x.shape[1],x.shape[0])
            
            print("src_mask: \n",src_padding_mask)
            
            """
            if x.dim()==1: #fix for batch-size 1 
                x = x.unsqueeze(0)
                
            bs = x.shape[0]
            #print("key-mask\n",src_padding_mask)
            clsmask = torch.zeros(bs,1).to(dtype=torch.bool)
            mask = torch.cat((clsmask,src_padding_mask[:,:].reshape(bs,32)),1) #unmask cls-token
            #print("reshaped mask\n",src_padding_mask)
            
            
            #src-mask needs be [batch-size,32]. It is solely calculated based on if x is -999, as both x,y will -999 pairwise.
            #src_padding_mask = src_padding_mask.swapaxes(0,1)
            """
            If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged. If a FloatTensor is provided, it will be added to the attention weight.
            https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask
            """
            
            #src_padding_mask = src_padding_mask.transpose(0,1) #
            
            x = x* math.sqrt(self.d_model) #as this in torch tutorial but dont know why
            x = torch.cat((self.cls_token.expand(x.shape[0],-1,2),x),1)
            x = self.pos_encoder(x)
            
            #print("Src_padding mask is: ",src_padding_mask)
            #print("pos encoding shape: ",x.shape)
            output = self.transformer_encoder(x,src_key_padding_mask=mask)
            #print("encoder output:\n",output)
            #print("Encoder output shape:\n",output.shape)
            #print("Same as input :-)")
           
            output = self.clsdecoder(output[:,0,:])
            
            return output
        
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout = 0.0,max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len,1,d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            
        Returns: 
            PosEnc(x): Tensor, shape [batchsize,seq_len,embedding_dim]
        """
        x = x.swapaxes(1,0) #[seq_len,bs,embedding_dim]
        x = x + self.pe[:x.size(0)] #[seq_len x bs x embedding dim]
        x = x.swapaxes(1,0) #[bs x seq_len x embedding_dim]
        return self.dropout(x)

            
   
model = eyeFormer_baseline()


#lossF = torch.nn.CrossEntropyLoss() #Kristian undrer sig over hvordan man kan bruge den
#lossF = ops.complete_box_iou_loss()

activation = {}
def getActivation(name):
    def hook(model,input,output):
        activation[name] = output.detach()
    return hook

#register forward hooks
h1 = model.transformer_encoder.register_forward_hook(getActivation('encoder'))
h2 = model.clsdecoder.register_forward_hook(getActivation('linear'))


def RMSELoss(output,target):
    """
    Get mean square error averaged on batch
    Calculates ((sqrt(x_true)- sqrt(x_pred))ˆ2 + (sqrt(y_true)- sqrt(y_pred))ˆ2) for all points and makes sequential sum
    Parameters
    ----------
    output : TYPE
        DESCRIPTION.
    target : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    loss = 0 
    if target.shape[0]==4: 
        target = target.unsqueeze(0)
    for i in range(target.shape[0]):
        loss += torch.pow(torch.sqrt(output[i])-torch.sqrt(target[i]),2)
    
    return torch.mean(loss) #average per batch loss

def MSELoss(output,target):
    """
    Get mean square error averaged on batch
    Calculates ((sqrt(x_true)- sqrt(x_pred))ˆ2 + (sqrt(y_true)- sqrt(y_pred))ˆ2) for all points and makes sequential sum
    Parameters
    ----------
    output : TYPE
        DESCRIPTION.
    target : TYPE
        DESCRIPTION.

    Returns
    -------
    Loss
    #Todo: consider adding additional penalties for out-of-image guesses (ie x<0 and x>1)

    """
    loss = 0 
    if target.shape[0]==4: 
        target = target.unsqueeze(0)
    for i in range(target.shape[0]):
        loss += torch.pow(output[i]-target[i],2)
    
    return torch.mean(loss) #average per batch loss


optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = nn.SmoothL1Loss() #default: mean and beta=1.0


encoder_list,linear_list = [], []
 

def train_one_epoch(model,loss) -> float: 
    running_loss = 0.
    #src_mask = generate_square_subsequent_mask(32).to(device)
    correct_count = 0
    false_count = 0
    batchwiseLoss = []
    
    
    for i, data in enumerate(trainloader):
        optimizer.zero_grad() #reset grads
        target = data["target"]
        mask = data["mask"]
        #print("Mask:\n",data["mask"])
        #print("Input: \n",data["signal"])
        #print("Goal is: \n",data["target"])
        outputs = model(data["signal"],mask)
        
        #register activations 
        encoder_list.append(activation["encoder"])
        linear_list.append(activation["linear"])
        
        #print("Prediction: \n",outputs)
        loss = loss_fn(outputs,target)
        #loss = MSELoss(outputs,target)
        #loss = ops.complete_box_iou_loss(outputs,data["target"].type(torch.LongTensor),reduction='mean') #reduction for calculating loss over whole batch and not just single-image
        #print("Loss is: ",loss)
        loss.backward()
        
        IOU = ops.box_iou(outputs,data["target"])
        correct_count += torch.sum(IOU>0.5) #PASCAL CRITERIUM
        false_count += data["signal"].shape[0]-correct_count #signal.shape[0] = batch_size 
        
        optimizer.step()
        running_loss += loss.item() #is complete EPOCHLOSS
        """
        if i%4==3: #report loss for every 4 batches (ie every 16 images)
            last_loss = running_loss/4
            print(' batch {} loss: {}'.format(i+1,last_loss))
            running_loss = 0.
         """
     
    epochLoss = running_loss/i
    return epochLoss,correct_count,false_count,target,data["signal"],mask

#train 
epoch_number = 0
EPOCHS = 5
epochLoss = 0 
model.train(True)
epochLossLI = []
torch.autograd.set_detect_anomaly(True)

for epoch in range(EPOCHS):
    model.train(True)
    print("EPOCH {}:".format(epoch_number+1))    
    epochLoss, correct_count, false_count,target,signal,mask = train_one_epoch(model,loss_fn)
    print("epoch loss {}".format(epochLoss))
    
    epochLossLI.append(epochLoss)
        
    
    epoch_number += 1 
    
h1.remove()
h2.remove()   



#TEST-LOOP
testlosses = []
with torch.no_grad():
    for i, data in enumerate(testloader):
        signal = data["signal"]
        target = data["target"]
        mask = data["mask"]
        output = model(signal,mask)
        batchloss = loss_fn(target,output)
        testlosses.append(batchloss)
        if i%100==0:
            print("L1-loss in batch {}: {}\n".format(i,batchloss))
        
        
    
    


    
import matplotlib.pyplot as plt
epochPlot = [x+1 for x in range(len(epochLossLI))]
plt.plot(epochPlot,epochLossLI)   
plt.ylabel("L1-LOSS") 
    

#TODO: 
    #IMPLEMENT 
        #PADDING-MASK  (CHECK)
        #PASCAL-CRITERIUM IMPLEMENTATION (ACTIVE)
            #Issue: seems nn.linear outputs (batch_sz,32,4) instead of just (batch_sz,4). Figure out why
        #MULTIPLE-EPOCH
        #FILENAMES (CHECK)
        
        


"""
for i_batch, sample_batched in enumerate(trainloader):
    print(i_batch,sample_batched)
    batch = sample_batched
    if i_batch==1:
        break
"""


# airplanesBoats = AirplanesBoatsDataset(dataFrame, root_dir, [0,9],transform=None) #init dataset as torch.Dataset with NO transforms. 
# #For debugging class instantiation with transforms:
# dataloader = DataLoader(airplanesBoats,batch_size=4,shuffle=True,num_workers=0)
# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch,sample_batched)
    
#     if i_batch==4:
#         break



# test_coords = sample_batched["signal"][0]
# test_dims = sample_batched["size"][0]
# test_box = sample_batched["target"][0]
# test_file = sample_batched["file"][0]
# print(rescale_coordsO(test_coords,test_dims,test_box))
# print("Seems to work fine! :-)")

# """
    