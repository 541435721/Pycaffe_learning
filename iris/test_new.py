# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/08_12:37 '

from numpy import *
from numpy import *
import caffe
from caffe import layers as L
from caffe import params as P
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.datasets import load_iris

if __name__ == '__main__':
    x = random.randint(10, 100, (10, 4))
    x = x * 1.0 / 10
    new_data = {}
    new_data['input'] = reshape(x, (len(x), 1, 1, 4))
    # print new_data['input']
    # 调用训练好的网络预测数据
    Net = caffe.Net('deploy.prototxt', 'iris__iter_1000000.caffemodel', caffe.TEST)
    out = Net.forward(data=new_data['input'])  # 调用测试数据
    # i 表示数据结果索引
    print out['ip3']
    for i in xrange(10):
        print "output:" + str(out['ip3'][i].argmax())
