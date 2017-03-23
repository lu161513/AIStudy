# -*- coding:utf-8 -*-


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


def runPCA():
    df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)  # 加载葡萄酒数据集
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values  # 把数据与标签拆分开来
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # 把整个数据集的70%分为训练集，30%为测试集
    # 下面3行代码把数据集标准化为单位方差和0均值
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.fit_transform(X_test)

    pca = PCA(n_components=2)  # 保留2个主成分
    lr = LogisticRegression()  # 创建逻辑回归对象
    X_train_pca = pca.fit_transform(X_train_std)  # 把原始训练集映射到主成分组成的子空间中
    X_test_pca = pca.transform(X_test_std)  # 把原始测试集映射到主成分组成的子空间中
    lr.fit(X_train_pca, y_train)  # 用逻辑回归拟合数据
    plot_decision_regions(X_train_pca, y_train, clf=lr, legend=2)
    print (lr.score(X_test_pca, y_test)) # 0.98 在测试集上的平均正确率为0.98
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='lower left')
    plt.show()

    return

if __name__ == "__main__":

    runPCA()