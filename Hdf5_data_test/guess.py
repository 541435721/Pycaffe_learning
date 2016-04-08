# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/08_09:27 '

from numpy import *
import caffe
from caffe import layers as L
from caffe import params as P
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model

if __name__ == '__main__':
    net = caffe.Classifier('deploy.prototxt', 'snap_iter_10000.caffemodel')
    # net.predict([array([1, 1])])
    Net = caffe.Net('deploy.prototxt', 'snap_iter_10000.caffemodel', caffe.TEST)

    Net.blobs['data'].data[...] = array([1, 5])
    out = Net.forward()
    # out = Net.forward(data=array([1, 5]))

    print 'lebel:' + str(out['prob'][0].argmax())

    t = random.randint(0, 100, (10, 2))
    for i in t:
        print i
        Net.blobs['data'].data[...] = i
        out = Net.forward()
        print 'lebel:' + str(out['ip3'][0].argmax())
