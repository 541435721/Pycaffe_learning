# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/08_08:52 '


from numpy import *
import caffe
from caffe import layers as L
from caffe import params as P
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model

if __name__ == '__main__':
    x = random.randint(0, 10, (1000, 2))  # 创建1000个样本，特征维度为2
    y = array(map(lambda x: int(2 * x[0] < x[1]), x))  # 创建标签

    x, xt, y, yt = sklearn.cross_validation.train_test_split(x, y)  # 分割训练集和数据集
'''
    with h5py.File('train_data.h5', 'w') as f:
        f['data'] = x
        f['label'] = y.astype(float32)
    with open('train.txt', 'w') as f:
        f.write('train_data.h5' + '\n')

    with h5py.File('test_data.h5', 'w') as f:
        f['data'] = xt
        f['label'] = yt.astype(float32)
    with open('test.txt', 'w') as f:
        f.write('test_data.h5' + '\n')

    solver = caffe.get_solver('solver.prototxt')  # 设置优化器配置文件
    solver.solve()
'''
