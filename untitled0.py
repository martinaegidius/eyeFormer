#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 20:50:07 2022

@author: max
"""

seq = 1

#nums = [x for x in range()]
nums = [1,2,3,4,5,6]
for num in nums: 
    seq = 1
    for i in range(num):
        seq *= (1-i/6)
    print("p(",num,"): ",seq)
