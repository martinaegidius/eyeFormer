#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 08:13:52 2022

@author: max
"""

import torch 
import math 
import torch.nn as nn 

#define transformer model. Goal: overfitting
class eyeFormer_baseline(nn.Module):
        def __init__(self,input_dim=2,hidden_dim=2048,output_dim=4,dropout=0.1):
            self.d_model = input_dim
            super(eyeFormer_baseline,self).__init__() #get class instance
            self.embedding = nn.embedding(32,512)
            self.pos_encoder = PositionalEncoding(input_dim,dropout)
            encoderLayers = nn.TransformerEncoderLayer(d_model=input_dim,nhead=1,dim_feedforward=hidden_dim,dropout=dropout,activation="relu",batch_first=True) #False due to positional encoding made on batch in middle
            #make encoder - 3 pieces
            self.cls_token = nn.Parameter(torch.zeros(1,1,hidden_dim),requires_grad=True)
            self.transformer_encoder = nn.TransformerEncoder(encoderLayers,num_layers = 1)
            #self.decoder = nn.Linear(64,4,bias=True) 
            self.clsdecoder = nn.Linear(2,4,bias=True)
            self.DEBUG = False
        
        def switch_debug(self):
            self.DEBUG = not self.DEBUG
            if(self.DEBUG==True):
                string = "on"
            else: string = "off"
            print("Debugging mode turned "+string)
            
            
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
            if self.DEBUG==True:
                print("1: reshaped mask\n",mask)
            
            
            #src-mask needs be [batch-size,32]. It is solely calculated based on if x is -999, as both x,y will -999 pairwise.
            #src_padding_mask = src_padding_mask.swapaxes(0,1)
            """
            If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged. If a FloatTensor is provided, it will be added to the attention weight.
            https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask
            """
            
            #src_padding_mask = src_padding_mask.transpose(0,1) #
            
            x = self.embedding(x)*math.sqrt(self.d_model) #as this in torch tutorial but dont know why
            x = torch.cat((self.cls_token.expand(x.shape[0],-1,2),x),1)
            if self.DEBUG==True:
                print("2: scaled and cat with CLS:\n",x,x.shape)
            x = self.pos_encoder(x)
            if self.DEBUG==True:
                print("3: positionally encoded: \n",x,x.shape)
            
            #print("Src_padding mask is: ",src_padding_mask)
            #print("pos encoding shape: ",x.shape)
            output = self.transformer_encoder(x,src_key_padding_mask=mask)
            if self.DEBUG==True:
                print("4: Transformer encoder output:\n",output)
            #print("encoder output:\n",output)
            #print("Encoder output shape:\n",output.shape)
            #print("Same as input :-)")
           
            output = self.clsdecoder(output[:,0,:])
            if self.DEBUG==True:
                print("5: linear layer based on CLS-token output: \n",output)
            
            return output
        
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout = 0.1,max_len = 5000):
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


xin = torch.cat((torch.arange(0,32,1),torch.arange(0,32,1)),0).reshape(32,2).unsqueeze(0)
yin = torch.cat((torch.arange(32,0,-1),torch.arange(0,32,1)),0).reshape(32,2).unsqueeze(0)

x = torch.cat((xin,xin,xin,xin),0) #[bs x seq len x attributes]
y = torch.cat((yin,yin,yin,yin),0)





