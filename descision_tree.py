#encoding=utf-8
import os, sys
import numpy as np
import matplotlib.pyplot as plt


feature_datas=np.array([[1,1],
              [1,1],
              [1,0],
              [0,1],
              [0,1]])
label_datas=np.array(['yes','yes','no','no','no']).reshape((5,1))
datas=np.hstack((feature_datas,label_datas))
sample_number=len(datas)
# inputs[:,-1]
def cal_entropy(sub_datas):
    labels=sub_datas[:,-1]
    label_factors=set(sub_datas[:,-1])
    label_factors_dict={}
    n_sample=len(sub_datas)
    for i in range(n_sample):
        current_label=labels[i]
        label_factors_dict[current_label]=label_factors_dict.get(current_label,0)+1
    entropy=0
    for factor in label_factors:
        p_x=1.0*label_factors_dict[factor]/n_sample
        entropy+=-p_x*np.log2(p_x)
    return entropy
# cal_entropy(datas)
def spilt_datas(datas,feature_cloumn_index,feature_val):
    n_sample=len(datas)
    feature_cloumn_data=datas[:,feature_cloumn_index]
    # feature_val=set(feature_cloumn_data)
    need_datas=[]
    for i in range(n_sample):
        if str(feature_cloumn_data[i])==str(feature_val):
            need_list=[]
            need_list=datas[i,0:feature_cloumn_index].tolist()
            need_list.extend(datas[i,feature_cloumn_index+1:])
            need_datas.append(need_list)
    return np.array(need_datas)
spilt_datas(datas,0,0)
def choose_best_feature_index(datas):
    # 计算H(D)
    H_D=cal_entropy(datas)
    g_D_A_list=[]
    # 对每个dim的feature计算H(Y|X)=sum(p_i*(H(Y|X=x_i)))
    feature_dim=len(datas[0])-1
    for i in range(feature_dim):
        feature_dim_data=datas[:,i].tolist()
        feature_dim_val=set(feature_dim_data)
        feature_dim_p=[feature_dim_data.count(val)*1.0/sample_number for val in feature_dim_val]
        feature_dim_p=np.array(feature_dim_p).reshape((len(feature_dim_p),1))
        new_entropy=0
        for j,value in enumerate(feature_dim_val):
            spilt_data=spilt_datas(datas,i,value)
            post_p=cal_entropy(spilt_data)
            # 计算p_i * (H(Y | X=x_i))
            new_entropy+=feature_dim_p[j]*post_p
        #计算g（D,A）
        g_D_A=H_D-new_entropy
        g_D_A_list.extend(g_D_A)
    print "the information gain:{}".format(g_D_A_list)
    print "the best feature is:{}".format(g_D_A_list.index(max(g_D_A_list)))

choose_best_feature_index(datas)







