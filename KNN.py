#encoding=utf-8
import os, sys

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#assert
datas=np.array([[1,1.1],[0,0],[1,1],[0,0.1]])
labels=np.array(['A','B','A','B'])
x=np.array([2,2])
distance=np.sum((x - datas)**2,axis=1)**0.5
dist_labels=np.vstack((distance,labels))
classCount={}
n_sample=len(datas)
for i in range(n_sample):
    classLabel=dist_labels[1][i]
    classCount[classLabel]=classCount.get(classLabel,0)+1
print classCount
max(classCount,key=classCount.get)

#TODO
# so=dist_labels.T(dist_labels.T[:,0].argsort()).T
# -np.sort(-dist_labels,axis=1)
# np.concatenate((np.sum(np.square(x - datas),axis=1)**0.5,np.array(labels)),axis=0)
# sorted([1,2],reverse=True)

