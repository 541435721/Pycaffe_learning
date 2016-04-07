# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/07_13:56 '

from numpy import *
import caffe
from caffe import layers as L
from caffe import params as P
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model

caffe.set_device(0)
caffe.set_mode_gpu()

x = random.randint(0, 100, (10, 2))
y = array(map(lambda x: int(x[-1] > x[0]), x))

with h5py.File('testdata.h5', 'w') as f:
    f['data'] = x
    f['label'] = y.astype(float32)
with open('testdata.txt', 'w') as f:
    f.write('testdata.h5' + '\n')

# 让caffe以测试模式读取网络参数
net = caffe.Net('Net.proto', 'snap_iter_10000.caffemodel', caffe.TEST)
out = net.forward()
#
# print net.blobs['data'].data  # 输出数据
# print net.blobs['label'].data  # 输出标签
# print net.blobs['loss'].data  # 输出预测结果
# # print out['loss']
# for k, v in net.blobs.items():
#     print (k, v.data.shape)
# net = caffe.Classifier('Net.proto', 'snap_iter_10000.caffemodel')
