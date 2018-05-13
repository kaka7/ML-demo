#encoding=utf-8
import os, sys

from datetime import datetime
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

x=np.arange(10).reshape((10,1))
y=np.array([1,1,1,-1,-1,-1,1,1,1,-1]).reshape((10,1))
datas=np.hstack((x,y))
x_min=np.min(x)
x_max=np.max(x)
n_sample=len(x)
step=10

step_length=1.0*(x_max-x_min)/step
feature_dim=len(datas[0])-1
compare_list=['lt','gt']


def cal_error_rate(data,feature_dim,val,compare_cls,D):
    pred=np.ones((n_sample,1))
    error_list=np.ones((n_sample,1))
    if compare_list=='lt':
        pred[data[:,feature_dim]<=int(val)]=-1
    else:
        pred[data[:,feature_dim]>int(val)]=-1
    error_list[pred==y]=0
    error_rate=D.T.dot(error_list)#加权
    print"the feature_dim is :{},val:{},compare_cls:{},error_rate:{}".format(feature_dim,val,compare_cls,error_rate)
    # return np.sum(error_list)/n_sample,pred
    return error_rate,pred
# cal_error(datas,0,2.2,'lt')

def find_base_func(D):
    base_G_m={}
    best_pred=None
    minError=np.inf
    for i in range(feature_dim):
        for j in range(-1,step+2):
            val=x_min+j*step_length
            for k in compare_list:
                e_m,pred=cal_error_rate(datas,i,val,k,D)
                if e_m<minError:
                    minError=e_m
                    best_pred=pred
                    base_G_m["feature_dim"]=i
                    base_G_m["value"]=val
                    base_G_m["compare_cls"]=k
    return base_G_m,minError,best_pred
# find_base_func()
def adaboost(iter_epoch=100):
    G_x=np.matrix(np.zeros((n_sample,1)))
    # (1)初始化权重
    base_G_m_list=dict
    D_init=1.0*np.ones((n_sample,1))/n_sample
    # （2）循环
    D_m_plus_1=D_init
    for i in range(iter_epoch):
        # (a,b)
        D_m=D_m_plus_1
        base_G_m,e_m,G_m_x=find_base_func(D_m)
        # (c)
        alpha_m=0.5*np.log((1-e_m)/e_m)
        base_G_m["alpha_m"]=alpha_m
        base_G_m_list.update(base_G_m)
        # (d)
        f_x_m=alpha_m*G_m_x
        weight_coef=-f_x_m*y
        Z_m=D_m.T.dot(np.exp(weight_coef))
        D_m_plus_1=D_m*np.exp(weight_coef)/Z_m
        print "the D_m_plus_1:{} ".format(D_m_plus_1)
        a = np.arange(10).reshape(10, 1)
        print G_x
        # G_x=G_x+a
        G_x=G_x+np.matrix(f_x_m)
        # test = np.equal(np.sign(G_x), y).flatten().tolist()
        # b= np.equal(np.sign(G_x), y).flatten().tolist().count(False)
        error=1.0*np.equal(np.sign(np.array(G_x)), y).flatten().tolist().count(False)/n_sample
        # error=np.sum(np.equal(np.sign(G_x),y))/n_sample
        print "error:{}".format(error)
        if error==0.0:
            print base_G_m_list
            print D_m
            break
adaboost()

        
        
        







