# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/08_10:34 '

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
    data = load_iris()  # 获取训练集
    x = data['data']  # 获取样本特征值
    target = data['target']  # 获取分类标签
    label = data['target_names']  # 获取分类标签名

    # 分类标签离散化
    y = zeros((len(target), 3))
    for count, target in enumerate(target):
        y[count][target] = 1

    x, xt, y, yt = sklearn.cross_validation.train_test_split(x, y)  # 分割训练集和数据集

    # 保存训练数据集，以便生成hdf5文件
    new_data = {}
    new_data['input'] = reshape(x, (len(x), 1, 1, 4))
    new_data['output'] = y

    # 保存测试数据集，以便生成hdf5文件
    test_data = {}
    test_data['input'] = reshape(xt, (len(xt), 1, 1, 4))
    test_data['output'] = yt

    # 训练集保存到hdf5文件
    with h5py.File('train_data.h5', 'w') as f:
        f['data'] = new_data['input'].astype(float32)
        f['label'] = new_data['output'].astype(float32)
    with open('train_data.txt', 'w') as f:
        f.write('train_data.h5' + '\n')

    # 测试集保存到hdf5文件
    with h5py.File('test_data.h5', 'w') as f:
        f['data'] = test_data['input'].astype(float32)
        f['label'] = test_data['output'].astype(float32)
    with open('test_data.txt', 'w') as f:
        f.write('test_data.h5' + '\n')

    # 设置GPU模式
    caffe.set_mode_cpu()
    # 训练神经网络
    solver = caffe.get_solver('solver.prototxt')
    solver.solve()
