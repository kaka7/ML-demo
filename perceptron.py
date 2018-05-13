#encoding=utf-8
import os, sys
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#os.path.join/isdir/isfile/exists/os.makedirs

data_dir = '/home/naruto/PycharmProjects/data/'
output_dir = "output"
ckpt_dir = "ckpt_dir"

FLAG = None

#assert

def get_iter_data(data):
    pass

data=np.array([(3,3,1),
               (4,3,1),
               (1,1,-1)])
n_samples=len(data)
x=data[0:n_samples,0:-1]
y=data[0:n_samples,-1]
dim_samples=np.shape(x)[1]

# 1
w=np.zeros((dim_samples,1))
b=0

# x=iter(x)
# y=iter(y)
# # while True:
# for i in range(10):
#     x_i=next(x)
#     y_i=next(y)
#     print x_i,y_i

lr=0.1
# while True:
for i in range(50):
    index=np.random.randint(0,3,1)
    # loss=y[index]*(x[index]*w+b)
    loss=y[index]*np.sum(x[index]*w.T+b)
    if loss<=0:
        w+=lr*y[index]*x[index].T
        b+=lr*y[index]
print w,b





