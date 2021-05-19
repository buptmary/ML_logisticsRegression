# -*- coding: utf-8 -*
# @File: logistics.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


# 读入文件
def load_data(filename):
    dataset = np.loadtxt(filename, encoding='utf-8', skiprows=1)
    temp = np.mat(dataset[:, 1:3])
    train_X = np.c_[temp, np.ones(len(temp))]  # 增加一列1，用于计算β
    labels = dataset[:, -1]
    return train_X, labels


def sigmoid(x, W):
    return 1.0 / (1.0 + np.exp(-x * W))


# logistics回归，返回W权重
def logistic_regression(train_X, labels, alpha=0.01, max_iter=1001):
    X = np.mat(train_X)
    Y = np.mat(labels).T
    m, n = np.shape(X)
    # 随机初始化W
    W = np.mat(np.random.randn(n, 1))
    w_save = []
    # 更新W
    for i in range(max_iter):
        H = sigmoid(X, W)
        dW = X.T * (H - Y)  # dW:(3,1）根据梯度下降算法，需要先求得dCost/dW，此处用dW代替
        W -= alpha * dW  # 梯度下降 W = W - alpha*dCost/dW

    return W


# 数据可视化
def show_diagram(train_X, labels, W):
    w1 = W[0, 0]
    w2 = W[1, 0]
    b = W[2, 0]
    plot_x1 = np.arange(0, 1, 0.01)
    plot_x2 = -w1 / w2 * plot_x1 - b / w2
    plt.plot(plot_x1, plot_x2, c='r', label='decision boundary')
    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.scatter(train_X[labels == 0, 0].A, train_X[labels == 0, 1].A, marker='^', color='r', s=80, label='bad')
    plt.scatter(train_X[labels == 1, 0].A, train_X[labels == 1, 1].A, marker='o', color='g', s=80, label='good')
    plt.legend(loc='upper right')
    plt.show()


# 结果输出函数
def predict(X, W):
    m = len(X)
    pred = np.zeros(m)
    for i in range(m):
        if sigmoid(X[i], W) > 0.5:      # 使用sigmoid判断，大于0.5label为1，否则为0
            pred[i] = 1

    return pred


def main():
    # 加载数据集
    train_X, labels = load_data('watermelon_3a.txt')
    # 对数据进行logistics回归分类
    W = logistic_regression(train_X, labels)
    print('W:', W.T)
    # 将数据集带入模型预测
    y_pred = predict(train_X, W)

    print(y_pred)
    print(labels)
    # 输出二分类的性能评估
    print(metrics.classification_report(labels, y_pred))
    # 可视化
    show_diagram(train_X, labels, W)


if __name__ == '__main__':
    main()
