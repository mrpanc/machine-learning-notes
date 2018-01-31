# coding: utf-8

# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from single_neural_network import sigmoid, sigmoid_prime


def save_imgae(filename, fig, dpi=300):
    """保存图片"""
    fig.savefig(filename, dpi=dpi)


def plot_training_cost(training_cost, axe=None):
    """绘制训练损失随Epoch变化的曲线"""
    if axe is None:
        axe = plt.subplot(1, 1, 1)
    axe.set_xlabel('Epoch')
    axe.set_ylabel('Cost')
    axe.plot(training_cost)
    axe.grid(True)


def plot_output(outputs, axe=None):
    """绘制输出随epoch变化的曲线"""
    if axe is None:
        axe = plt.subplot(1, 1, 1)
    axe.set_xlabel('Epoch')
    axe.set_ylabel('Output')
    axe.plot(outputs)
    axe.grid(True)


def plot_sigmoid(x=np.linspace(-10, 10, 1000), axe=None):
    """绘制sigmoid函数图像"""
    if axe is None:
        axe = plt.subplot(1, 1, 1)
    axe.set_xlabel('z')
    axe.set_ylabel('Value of sigmoid function')
    axe.set_title('Figure of sigmoid function')
    axe.plot(x, sigmoid(x))
    axe.grid(True)


def plot_sigmoid_prime(x=np.linspace(-10, 10, 1000), axe=None):
    """绘制sigmoid_prime函数图像"""
    if axe is None:
        axe = plt.subplot(1, 1, 1)
    axe.set_xlabel('z')
    axe.set_ylabel('Value of sigmoid function')
    axe.set_title('Figure of sigmoid prime function')
    axe.plot(x, sigmoid_prime(x))
    axe.grid(True)
