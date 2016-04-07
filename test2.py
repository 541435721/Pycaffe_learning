# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/07_15:08 '

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

x = array([[1, 0], [1, 1], [0, 0], [0, 1]])
y = array([0, 1, 1, 0])

with h5py.File('ul_train.h5', 'w') as f:
    f['data'] = x
    f['label'] = y.astype(float32)
with open('ul_train.txt', 'w') as f:
    f.write('ul_train.h5' + '\n')

with h5py.File('ul_test.h5', 'w') as f:
    f['data'] = x
    f['label'] = y.astype(float32)
with open('ul_test.txt', 'w') as f:
    f.write('ul_test.h5' + '\n')

solver = caffe.get_solver('ul_solver.proto')  # 设置优化器配置文件
solver.solve()

print
