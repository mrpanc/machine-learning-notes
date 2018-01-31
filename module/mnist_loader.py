# coding: utf-8

import numpy as np

from mnist import MNIST
from sklearn.preprocessing import MinMaxScaler


def vectorized_result(j):
    """
    将数字(0...9)变为one hot向量
    输入：
        j: int，数字(0...9)
    输出：
        e: np.ndarray, 10维的向量，其中第j位为1，其他位都为0。
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data_wrapper(dirpath, scale=True):
    """
    载入mnist数字识别数据集
    输入
        dirpath: str, 数据所在文件夹路径
    输出：
        training_data: list, 包含了60000个训练数据集，其中每一个数据由一个tuple '(x, y)'组成，
                        x是训练的数字图像，类型是np.ndarray, 维度是(784,1)
                        y表示训练的图像所属的标签，是一个10维的one hot向量
        test_data: list, 包含了10000个测试数据集，其中每一个数据由一个tuple '(x, y)'组成，
                        x是测试的数字图像，类型是np.ndarray, 维度是(784,1)
                        y表示测试的图像所属标签，int类型，是一个(0...9)的数字
    """
    mndata = MNIST(dirpath)
    tr_i, tr_o = mndata.load_training()
    te_i, te_o = mndata.load_testing()
    if scale:
        min_max_scaler = MinMaxScaler()
        tr_i = min_max_scaler.fit_transform(tr_i)
        te_i = min_max_scaler.transform(te_i)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_i]
    training_outputs = [vectorized_result(y) for y in tr_o]
    training_data = list(zip(training_inputs, training_outputs))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_i]
    test_data = list(zip(test_inputs, te_o))
    return training_data, test_data
