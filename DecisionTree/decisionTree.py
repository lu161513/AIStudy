# -*- coding:utf-8 -*-

import numpy as np
import urllib
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# 分类与回归树(Classification and Regression Trees ,CART)算法常用于特征含有类别信息的分类或者回归问题，这种方法非常适用于多分类情况。



def run():
    # url with dataset
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    # download the file
    raw_data = urllib.urlopen(url)
    # load the CSV file as a numpy matrix
    dataset = np.loadtxt(raw_data, delimiter=",")
    # separate the data from the target attributes
    X = dataset[:, 0:7]
    y = dataset[:, 8]

    # 数据归一化
    # normalize the data attributes
    normalized_X = preprocessing.normalize(X)
    # standardize the data attributes
    standardized_X = preprocessing.scale(X)

    # fit a CART model to the data
    model = DecisionTreeClassifier()
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


if __name__ == "__main__":
    run()
