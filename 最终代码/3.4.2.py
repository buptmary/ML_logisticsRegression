# -*- coding: utf-8 -*
# @File: 3.4.2.py

import logistics
import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings('ignore')


# 读入文件
def load_data(filename):
    df = pd.read_excel(filename)
    dataset = df.values
    print(np.shape(dataset))
    temp = np.mat(dataset[:, 1:])
    train_X = np.c_[temp, np.ones(len(temp))]  # 增加一列1，用于计算β
    labels = dataset[:, 0]
    return train_X, labels


def leave_one(train_X, labels):
    loo = LeaveOneOut()
    total_acc = 0
    loo.get_n_splits(train_X)
    num = 0
    for train_index, test_index in loo.split(train_X, labels):
        x_train, x_test = train_X[train_index], train_X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # 开始进行logistics回归分类训练
        W = logistics.logistic_regression(x_train, y_train, alpha=0.01, max_iter=1000)  # 计算权重W
        y_pred = logistics.predict(x_test, W)  # 根据训练好的模型进行预测并输出预测值

        acc = accuracy_score(y_test, y_pred)
        total_acc += acc
        num += 1
        if num % 100 == 0:
            print("前", num, "组错误率为", 1 - (total_acc / num))
    print("留一法的平均错误率为：", 1 - (total_acc / num))


def K_fold(train_X, labels, splits=10):
    order_id = []
    total_acc = 0
    sfolder = StratifiedKFold(n_splits=splits, shuffle=True)  # 十折交叉验证划分数据集
    for num, (train, test) in enumerate(sfolder.split(train_X, labels)):
        x_train = train_X[train, :]
        y_train = labels[train]
        x_test = train_X[test, :]
        y_test = labels[test]
        order_id.extend(test)

        # 开始进行logistics回归分类训练
        W = logistics.logistic_regression(x_train, y_train, alpha=0.01, max_iter=1001)  # 计算权重W
        y_pred = logistics.predict(x_test, W)   # 根据训练好的模型进行预测并输出预测值

        acc = accuracy_score(y_test, y_pred)
        total_acc += acc
        print('第', num + 1, '折验证的错误率', 1 - acc)
    print("十折交叉验证的平均错误率为：", 1 - (total_acc / splits))


def main():
    x, y = load_data('Diabetes.xls')  # 读取文件
    K_fold(x, y)
    print("------------------------------------------")
    leave_one(x, y)


if __name__ == '__main__':
    print("diabetes数据集logistics回归训练结果")
    main()
