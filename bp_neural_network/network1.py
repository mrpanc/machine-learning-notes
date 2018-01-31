# coding: utf-8

import numpy as np
import random


class Network(object):
    def __init__(self, sizes):
        """初始化神经网络
        1. 根据输入，得到神经网络的结构
        2. 根据神经网络的结构使用均值为0，方差为1的高斯分布初始化参数权值w和偏差b。
        输入：
        sizes: list, 表示神经网络各个layer的数目，例如[784, 30, 10]表示3层的神经网络。
                    输入层784个神经元，隐藏层只有1层，有30个神经元，输出层有10个神经元。
        """
        np.random.seed(41)
        random.seed(41)
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def feed_forward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self,
            training_data,
            epochs,
            mini_batch_size,
            alpha,
            test_data=None):
        """随机梯度下降
        输入：
        training_data：是由tuples ``(x, y)``组成的list，x表示输入，y表示预计输出
        epoches：int, 表示训练整个数据集的次数
        mini_batch_size: int, 在SGD过程中每次迭代使用训练集的数目
        alpha: float, 学习速率
        test_data: 是由tuples ``(x, y)``组成的list，x表示输入，y表示预计输出。
                    如果提供了``test_data``，则每经过一次epoch，都计算并输出当前网络训练结果在测试集上的准确率。
                    虽然可以检测网络训练效果，但是会降低网络训练的速度。
        """
        if test_data:
            n_test = len(test_data)
        m = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, m, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, alpha):
        """每迭代一次mini_batch，根据梯度下降方法，使用反向传播得到的结果更新权值``w``和偏差``b``
        输入：
        mini_batch: 由tuples ``(x, y)``组成的list
        alpha: int，学习速率
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nable_w = self.back_prop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nable_w)]
        self.weights = [
            w - (alpha / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (alpha / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def back_prop(self, x, y):
        """反向传播
        1. 前向传播，获得每一层的激活值
        2. 根据输出值计算得到输出层的误差``delta``
        3. 根据``delta``计算输出层C_x对参数``w``, ``b``的偏导
        4. 反向传播得到每一层的误差，并根据误差计算当前层C_x对参数``w``, ``b``的偏导
        输入：
        x: np.ndarray, 单个训练数据
        y: np.ndarray, 训练数据对应的预计输出值
        输出：
        nabla_b: list, C_x对``b``的偏导
        nabla_w: list, C_x对``w``的偏导
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward prop
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward prop
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(
            zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """计算准确率，将测试集中的x带入训练后的网络计算得到输出值，
            并得到最终的分类结果，与预期的结果进行比对，最终得到测试集中被正确分类的数目
        输入：
        test_data: 由tuples ``(x, y)``组成的list
        输出：
        int, 测试集中正确分类的数据个数
        """
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for x, y in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """代价函数对a的偏导
        输入：
        output_activations： np.ndarray, 输出层的激活值，即a^L
        y: np.ndarray, 预计输出值
        输出：
        output_activations-y: list, 偏导值
        """
        return (output_activations - y)


# 激活函数及其导数
def sigmoid(z):
    """The sigmoid function"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))
