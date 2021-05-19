import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate(sw, x):
    sw[0, 0] += x[0] * x[0]
    sw[0, 1] += x[0] * x[1]
    sw[1, 0] += x[1] * x[0]
    sw[1, 1] += x[1] * x[1]
    return sw


def LDA_w(df):
    df_labels = df.iloc[:, -1]
    labels = list(set(df_labels.values))
    index_1 = []
    index_2 = []
    for i in range(df.shape[0]):
        if df.iloc[i, -1] == labels[0]:
            index_1.append(i)
        else:
            index_2.append(i)
    df1 = df.iloc[index_1, :]
    df2 = df.iloc[index_2, :]
    x1 = df1.values[:, 1:3]
    x2 = df2.values[:, 1:3]
    mean1 = np.array([np.mean(x1[:, 0]), np.mean(x1[:, 1])])
    mean2 = np.array([np.mean(x2[:, 0]), np.mean(x2[:, 1])])
    Sw = np.zeros((2, 2))
    for i in range(x1.shape[0]):
        Sw = calculate(Sw, x1[i, :] - mean1)
    for i in range(x2.shape[0]):
        Sw = calculate(Sw, x2[i, :] - mean2)
    w = np.linalg.inv(Sw) @ (mean1 - mean2).transpose()
    return w


def LDA_plot(df, w):
    df_labels = df.iloc[:, -1]
    labels = list(set(df_labels.values))
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(df.shape[0]):
        if df.iloc[i, -1] == labels[0]:
            x1.append(df.iloc[i, 1])
            y1.append(df.iloc[i, 2])
        else:
            x2.append(df.iloc[i, 1])
            y2.append(df.iloc[i, 2])
    plt.plot(x1, y1, 'gs', label="first kind")
    plt.plot(x2, y2, 'r*', label="second kind")
    x = np.arange(0, 1, 0.01)
    y = np.array((-w[0] * x) / w[1])
    plt.plot(x, y, label="LDA")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel(df.columns[1])
    plt.ylabel(df.columns[2])
    plt.title('LDA')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    io = 'D://西瓜数据集3.0a.xlsx'
    dataframe = pd.read_excel(io)
    w = LDA_w(dataframe)
    LDA_plot(dataframe, w)
    print('LDA曲线为：', w[0], '*x ', w[1], '*y = 0')
