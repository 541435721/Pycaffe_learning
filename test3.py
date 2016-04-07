# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/07_15:26 '

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

net = caffe.Classifier('ul_net.proto', 'ul_snap_iter_10000.caffemodel')


predictions = net.predict([])
# out = net.forward()
# print net.blobs['data'].data  # 输出数据
# print net.blobs['label'].data  # 输出标签
# print net.blobs['loss'].data  # 输出预测结果
