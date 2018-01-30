# coding: utf-8

import json
import random

import numpy as np


# 激活函数和激活函数的导数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


# 向量化输出
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


# 定义 quatratic cost 函数和 cross-entropy cost 函数
class QuatraticCost(object):
    @staticmethod
    def activate(z):
        return sigmoid(z)

    @staticmethod
    def predict(a, weights, biases):
        """
        Forward Propagation: 对于每一层网络的输入a，返回激活函数sigmoid(w.T*a+b)的输出
        """
        for w, b in zip(weights, biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    @staticmethod
    def fn(a, y):
        """返回损失"""
        return 0.5 * np.linalg.norm(a - y)**2

    @staticmethod
    def delta(z, a, y):
        """返回输出层的损失delta"""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def activate(z):
        return sigmoid(z)

    @staticmethod
    def predict(a, weights, biases):
        """
        Forward Propagation: 对于每一层网络的输入a，返回激活函数sigmoid(w.T*a+b)的输出
        """
        for w, b in zip(weights, biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    @staticmethod
    def fn(a, y):
        """返回损失"""
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """返回输出层的损失delta"""
        return a - y


def softmax(z):
    """偏移C=max(z)，避免z过大或过小导致上溢出或下溢出"""
    c = np.max(z)
    exp_z = np.exp(z - c)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


class SoftmaxCost(object):
    @staticmethod
    def activate(z):
        return softmax(z)

    @staticmethod
    def predict(a, weights, biases):
        """根据输入得到预测的输出"""
        for i, w, b in zip(range(len(weights)), weights, biases):
            z = np.dot(w, a) + b
            if i == len(weights) - 1:
                a = softmax(z)
            else:
                a = sigmoid(z)
        return a

    @staticmethod
    def fn(a, y):
        """返回损失"""
        return -np.sum(np.nan_to_num(np.log(a) * y))

    @staticmethod
    def delta(z, a, y):
        """返回输出层的损失delta"""
        return a - y


# 神经网络
class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        初始化网络
        1. sizes:类型为list,[2,3,4]表示3层神经网络，
            第一层2个神经元，第二层3个神经元，第三层4个神经元
        2. cost: 默认为Cross-Entropy cost
        3. weights和biases通过default_weight_initializer随机化
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """
        weights:随机化为均值0标准差1的高斯分布/sqrt(x)，
                x为连接同一神经元的权重数目，保证所有的输出即w.T*x近似为同一分布
        biases:随机化为均值0标准差1的高斯分布
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def large_weight_initializer(self):
        """
        weights,biases:随机化为均值0标准差1的高斯分布，var(w.T*x)会随着x规模的增大而增大
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def SGD(self,
            training_data,
            epoches,
            mini_batch_size,
            alpha,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_accuracy=False,
            monitor_evaluation_cost=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """
        1. 将训练集打乱后根据mini_batch_size分为多个子训练集(mini_batch)
        2. 对于每一个mini_batch进行训练，不断更新weights和biases
        3. 根据需要计算training_cost, training_accuracy, test_cost, test_accuracy
        """
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(0, epoches):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha, lmbda,
                                       len(training_data))
            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {}".format(accuracy))

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {}".format(accuracy))
        return (training_cost, training_accuracy, evaluation_cost,
                evaluation_accuracy)

    def update_mini_batch(self, mini_batch, alpha, lmbda, n):
        '''
        1. 通过BP算法得到损失函数对w,b的梯度
        2. 更新weights和biases
        '''
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backword_prop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [
            (1 - alpha * lmbda / n) * w - alpha / len(mini_batch) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - alpha / len(mini_batch) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backword_prop(self, x, y):
        '''
        1. 前向传播，得到每一层的激活值
        2. 计算输出层产生的错误，即delta_L
        3. 反向传播错误，得到每一层的delta
        4. 根据delta计算损失函数C对w,b的偏导
        '''
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # forward prop
        activation = x
        activations = [x]
        zs = []
        for i, w, b in zip(range(len(self.weights)),
                           self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            if i == len(self.weights) - 1:
                activation = self.cost.activate(z)
            else:
                activation = sigmoid(z)
            activations.append(activation)
        # backward
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            nabla_b[-l] = delta
        return (nabla_w, nabla_b)

    def accuracy(self, data, convert=False):
        '''
        若data来自于training_set，则y是一组10维的向量，计算时需要设置convert为True
        若data来自于test_set或evaluation_set，则不需要convert，直接把y拿来用即可
        '''
        if convert:
            results = [(np.argmax(self.cost.predict(x,
                        self.weights, self.biases)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.cost.predict(x,
                        self.weights, self.biases)), y)
                       for (x, y) in data]

        return 1.0 * sum(int(x == y) for (x, y) in results) / len(data)

    def total_cost(self, data, lmbda, convert=False):
        """
        根据不同的代价函数计算总损失
        """
        cost = 0.0
        for (x, y) in data:
            a = self.cost.predict(x, self.weights, self.biases)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w)**2 for w in self.weights)

        return cost

    def predict(self, a):
        return self.cost.predict(a, self.weights, self.biases)

    def save(self, filename):
        """
        将神经网络的结构、学习后的参数以及使用的成本函数通过json格式存储，便于之后使用
        """
        data = {
            "sizes": self.sizes,
            "wights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__)
        }
        f = open(filename, 'r')
        json.dump(data, f)
        f.close()
