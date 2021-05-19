from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


# 读入文件
def load_data(filename):
    df = pd.read_csv(filename)
    dataset = df.values
    print(np.shape(dataset))
    temp = np.mat(dataset[:, 0:30])
    print(temp, np.shape(temp))
    train_X = np.c_[temp, np.ones(len(temp))]  # 增加一列1，用于计算β
    labels = dataset[:, -1]
    print(train_X, np.shape(train_X))
    # print(labels)
    return train_X, labels


x_train, y_train = load_data('breast_cancer.csv')

logreg = LogisticRegression().fit(x_train, y_train)
print("Training set score:{:.3f}".format(logreg.score(x_train, y_train)))
