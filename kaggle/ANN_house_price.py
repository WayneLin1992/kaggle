# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:32:43 2020

@author: Orange
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('train.csv')
x = dataset.iloc[:, 1:80].values
y = dataset.iloc[:, -1].values
df = dataset
a = df.isnull().sum()
droplist = []
for i in a:
    if i != 0:
        droplist.append(i)
    