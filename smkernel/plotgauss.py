# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 01:56:58 2021

@author: tom9m
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


n=0 #次元数-1
r=0 #動画番号-1
name='aist4'
mu=np.loadtxt('c:/Users/tom/src/input/smkdata/{}_{}d.txt'.format(name,n+1))[r,:]
v=np.loadtxt('c:/Users/tom/src/input/smkdata/{}_v{}d.txt'.format(name,n+1))[r,:]
w=np.loadtxt('c:/Users/tom/src/input/smkdata/{}_w{}d.txt'.format(name,n+1))[r,:]        
K=4
"""
x = np.arange(-50., 100, 0.01)     #-8から８まで0.01刻みの配列

for i in zip(v,mu,w):     #zipは同時にループしてくれます

    y = (1 / np.sqrt(2 * np.pi * i[0] ) ) * np.exp(-(x-i[1])**2/(2 * i[0]))*i[2]    #ガウス分布の公式

    plt.plot(x, y)     #x, yをplotします
    plt.grid()     #グリット線
    plt.xlabel('x')     #x軸のラベル
    plt.ylabel('y')     #y軸のラベル

plt.show()
"""
n = 301
xx = np.linspace(-4, 60, n)
d=2
m =mu

sigma = v

pi = w

# Density function
pdfs = np.zeros((n, K))
for k in range(K):
    
    pdfs[:, k] = pi[k]*stats.norm.pdf(xx, loc=m[k], scale=sigma[k])

# =======================================
# Visualization

plt.figure(figsize=(14, 6))
for k in range(K):
    plt.plot(xx, pdfs[:, k],label='topic{}'.format(k+1))
plt.title("pdfs")
plt.xlim(-4,60)
plt.legend(  bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0,fontsize=18)

plt.show()

plt.figure(figsize=(14, 6))
plt.stackplot(xx, pdfs[:, 0], pdfs[:, 1], pdfs[:, 2], pdfs[:, 3])#, pdfs[:, 4], pdfs[:, 5], pdfs[:, 6], pdfs[:, 7], pdfs[:, 8], pdfs[:, 9])
plt.title("stacked")
plt.xlim(-4,60)
plt.show()