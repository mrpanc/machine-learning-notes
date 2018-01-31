# coding: utf-8

# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def plot_mnist_image(matrix):
    """展示MNIST数据集中的数字"""
    assert(len(matrix) == 784)
    plt.imshow(np.reshape(matrix, [28, 28]))
