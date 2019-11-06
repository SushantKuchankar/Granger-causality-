# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:21:51 2018

@author: Sushant
"""
import scipy.io as sci
import statsmodels.tsa.stattools as stat
import numpy as np
import matplotlib.pyplot as plt

#Alogrithms written by me
from Algorithms import *


#loading Daata for one subject
path2 = "C:/Users/Gaurav/Desktop"
#change file name to change the subject
x = sci.loadmat(path2 +"/"+"eeg.mat")
data = x["x"]

y = data

#Augmented Dickey-Fuller unit root test for stationaty 
for i in range(0,61)
    print(statsmodels.tsa.stattools.adfuller(data[i,:,1].T))



electrod = [1,2,3,4,5,22,23,24,25,26,27,28,29,30]
y = data[electrod,:,:30]
order = orderselection(y,50)
plt.plot(order[0])
plt.plot(np.arange(1,len(order[2])+1),order[2])
plt.xlabel("Model lag")
plt.ylabel("BIC index")
plt.title("VAR Model Order Selection")
plt.plot(order[0])


g  = GrangerTest(y2,7,5)



####################GC for synthetic data####################################
#Genarating syntheic dataset for evalution of granger causlity algorithm
# d(t) = c0y(t-1) +  .....   + clag*y(t-lag) +b0d(t-1) +  .....   + bminusinfi*d(minus infi)
alpha = 0.7
y = np.random.rand(1000)
lag = 5
c1 = np.exp(-np.arange(0,10,10/lag) )
c2 = np.cos(-np.arange(0,10,10/lag) )
d = np.empty(np.shape(y))
for i in range(0,len(y)):
    if(i<lag):
        d[i]=0.09
    else:
        d[i]= (1-alpha)*(np.dot(y[i-lag:i],c1))+alpha*np.dot(d[i-lag:i],c2)
d5 = np.vstack((d,y))
d6 = np.vstack((y,d))

gv_d  = GrangerTest(d5,5,1,0.01)
gd_v= GrangerTest(d6,5,1,0.01)





