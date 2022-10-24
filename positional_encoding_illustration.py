#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:07:27 2022

@author: max
"""

import torch 
import math 

max_len = 32
d_model = 4

pe_matrix = torch.zeros(max_len,d_model)
position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1) #unsqueeze to make it a column-vector

posIdx = torch.arange(0,d_model,2).float() #(start,end,step). Possible values of i
argTerm = posIdx/torch.pow(10000,posIdx*d_model)

pe_matrix[:,0::2] = torch.sin(position*argTerm)
pe_matrix[:,1::2] = torch.cos(position*argTerm)


#their way 
div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
