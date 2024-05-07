#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'
# os.chdir(r'D:\Desktop\Quantum\待绘图\SAPO') #数据文件存放文件夹位置


def filt(y):
    sy = savgol_filter(y,40,2)
    return sy

df = pd.read_csv('data_combined_min.csv')


colors_bar = ['#FF799C', '#00EE7D', '#009ade', '#af58ba', '#ffc61e', '#a0b1ba']
colors_line = ['#ff1f5b', '#00E276', '#1470A8', '#6e005f', '#f28522', '#899EA9']


fig,ax = plt.subplots(figsize=(6,4),facecolor='white')
for i in range(5):
    bins,tick = np.histogram(df.iloc[:,i],bins=np.arange(0,5,1/30))
    bins = bins/sum(bins)
    tick = tick[:-1]+1/30
    ax.bar(tick,bins,width=1/30,align='edge',color=colors_bar[i],alpha=0.6,ec='w',lw=0.8)
    cs = CubicSpline(tick,bins)
    x = np.linspace(tick[0],tick[-1],200)
    y = cs(x) #样条插值
    y = filt(y) #平滑
    ax.plot(x,y,color=colors_line[i],label=df.columns[i],alpha=0.6)
ax.set_ylim(0,0.12)
ax.set_xlim(0,5)
ax.tick_params(direction='in')

ax.legend()
ax.set_xlim(0,5)
ax.set_xlabel('Distribution')
plt.show()
