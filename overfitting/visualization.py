# coding: utf-8

# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def save_imgae(filename, fig, dpi=300):
    """保存图片"""
    fig.savefig(filename, dpi=dpi)


def plot_training_accuracy(training_accuracy, ax=None, epoches=400, xmin=0):
    """打印训练集准确率随epoch变化曲线图"""
    if ax is None:
        ax = plt.subplot(111)
    ax.plot(np.arange(xmin, epoches),
            training_accuracy[xmin:epoches],
            color='#2A6EA6')
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title("Accuracy on training data")


def plot_test_accuracy(test_accuracy, ax=None, epoches=400, xmin=0):
    """打印测试集准确率随epoch变化曲线图"""
    if ax is None:
        ax = plt.subplot(111)
    ax.plot(np.arange(xmin, epoches),
            test_accuracy[xmin:epoches],
            color='#FFA933')
    ax.grid(True)
    ax.set_title("Accuracy on test data")
    ax.set_xlabel('Epoch')


def plot_training_evaluation_accuracy(training_accuracy, test_accuracy,
                                      ax=None, epoches=400, xmin=0):
    """打印训练集准确率曲线和测试集准确率曲线对比图"""
    if ax is None:
        ax = plt.subplot(111)
    ax.plot(np.arange(xmin, epoches),
            training_accuracy[xmin:epoches],
            color='#2A6EA6',
            label="Accuracy on the training data")
    ax.plot(np.arange(xmin, epoches),
            test_accuracy[xmin:epoches],
            color='#FFA933',
            label="Accuracy on the test data")
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.legend(loc=0)


def plot_training_cost(training_cost, ax=None, epoches=400, xmin=0):
    """打印训练集成本随epoch变化曲线图"""
    if ax is None:
        ax = plt.subplot(111)
    ax.plot(np.arange(xmin, epoches),
            training_cost[xmin:epoches],
            color='#2A6EA6')
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title("Cost on training data")


def plot_test_cost(test_cost, ax=None, epoches=400, xmin=0):
    """打印测试集成本随epoch变化曲线图"""
    if ax is None:
        ax = plt.subplot(111)
    ax.plot(np.arange(xmin, epoches),
            test_cost[xmin:epoches],
            color='#FFA933')
    ax.grid(True)
    ax.set_title("Cost on test data")
    ax.set_xlabel('Epoch')


def plot_training_evaluation_cost(training_cost, evaluation_cost,
                                  ax=None, epoches=400, xmin=0):
    """打印训练集代价函数曲线和验证集代价函数曲线对比图"""
    if ax is None:
        ax = plt.subplot(111)
    ax.plot(np.arange(xmin, epoches),
            training_cost[xmin:epoches],
            color='#2A6EA6',
            label="Cost on the training data")
    ax.plot(np.arange(xmin, epoches),
            evaluation_cost[xmin:epoches],
            color='#FFA933',
            label="Cost on the test data")
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.legend(loc=0)
