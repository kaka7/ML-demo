from PIL import Image
import numpy as np
img1=Image.open('/home/naruto/PycharmProjects/data/marason.png')
img1.show()
img1.size#712,1184
img1.getdata()
img1_data=np.array(img1.getdata())
print img1_data
print np.matrix(img1_data,dtype='float')
data1= np.matrix(img1_data,dtype='float')/255.0
print data1

img1_L = img1.convert("L")
print img1_L
img1_L_data = img1_L.getdata()
print img1_L_data
data2 = np.matrix(img1_L_data,dtype='float')
print data2.shape
print data2
Image.fromarray(np.reshape(data2,[712,1184])).show()
Image.fromarray(np.reshape(data2,[1184,712])).show()
Image.fromarray(np.reshape(np.matrix(img1_L_data,dtype='float')/255.0,[712,1184])).show()#/255 就看不到啦
# data = data*255#data [0,1]
# new_im = Image.fromarray(data.astype(np.uint8))
# plt.imshow(data, cmap=plt.cm.gray, interpolation=‘nearest‘)
# new_im.show()
# new_im.save(‘lena_1.bmp‘)

import numpy as np
import matplotlib.pylab as plt
# 加载图像
im = plt.imread('/home/naruto/PycharmProjects/data/marason.png') # 加载当前文件夹中名为BTD.jpg的图片
print(im.shape) # 输出图像尺寸(1184, 712, 4) why????????????????????????????

# 裁剪图像
#
# def plti(im, **kwargs):
#     """
#     画图的辅助函数
#     """
#     plt.imshow(im, interpolation="none", **kwargs)
#     plt.axis('off') # 去掉坐标轴
#     plt.show() # 弹窗显示图像
#
# im = im[400:3800,:2000,:]  # 直接切片对图像进行裁剪
# plti(im)

# 分离各通道的图像
#
# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
# # 将一张图分为1x3个子图，axs为各子图对象构成的列表。figsize为显示窗口的横纵比。
#
# for c, ax in zip(range(3), axs): # 使用zip来同时循环3通道和3个子图对象
#     tmp_im = np.zeros(im.shape) # 初始化一个和原图像大小相同的三维数组
#     # 注意 tmp_im 仍然是三通道
#     tmp_im[:,:,c] = im[:,:,c] # 只复制某一通道
#     one_channel = im[:,:,c].flatten() # 索引该通道并展平至一维
#     print("channel", c, " max = ", max(one_channel), "min = ", min(one_channel)) # 输出该通道最大最小的像素值
#     ax.imshow(tmp_im) # 在子图上绘制
#     ax.set_axis_off() # 去掉子图坐标轴
# # 注意以上 tmp_im 采用的是切片复制
# plt.show()
#
# #输出：
# #channel 0  max =  220 min =  11
# #channel 1  max =  203 min =  10
# # #channel 2  max =  185 min =  0
# https://www.jianshu.com/p/6dcb1c1af2a7
# https://www.jianshu.com/p/f8811ecf14f7