# coding: utf-8
import numpy as np


# 定义 quatratic cost 函数和 cross-entropy cost 函数
class QuatraticCost(object):

    @staticmethod
    def fn(a, y):
        """返回损失"""
        return 0.5 * np.linalg.norm(a - y)**2

    @staticmethod
    def delta(z, a, y):
        """返回输出层的损失delta"""
        return (a - y) * sigmoid_prime(z)


class Network(object):
    def __init__(self, w, b, cost=QuatraticCost):
        self.w = w
        self.b = b
        self.cost = cost

    def feed_forward(self, x):
        """根据当前的参数得到预计输出"""
        return sigmoid(self.w * x + self.b)

    def gradiant_decent(self, training_data, epoches, alpha):
        """梯度下降"""
        training_cost, outputs = [], []
        for j in range(epoches):
            self.update_params(training_data, alpha)
            cost = self.compute_cost(training_data)
            output = self.feed_forward(training_data[0])
            outputs.append(output)
            training_cost.append(cost)
        return training_cost, outputs

    def update_params(self, training_data, alpha):
        """更新参数w, b"""
        x, y = training_data
        z = self.w * x + self.b
        a = sigmoid(z)
        delta = self.cost.delta(z, a, y)
        nabla_w = delta * x
        nabla_b = delta
        self.w = self.w - alpha * nabla_w
        self.b = self.b - alpha * nabla_b

    def compute_cost(self, training_data):
        """计算损失"""
        x, y = training_data
        a = self.feed_forward(x)
        return self.cost.fn(a, y)


def sigmoid(z):
    """激活函数"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """激活函数的导数"""
    return sigmoid(z) * (1.0 - sigmoid(z))
