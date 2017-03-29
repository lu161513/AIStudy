#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np



# 从原始数据集的CSV里面读取数据，并且加载到Tensorflow当中
# 构建一个基于神经网络的分类器
# 使用训练数据进行模型训练
# 使用测试数据进行模型评估
# 使用训练好的模型对新的数据进行分类

# 设定数据集的位置
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# 使用Tensorflow内置的方法进行数据加载，target_type是最终的label的类型，这里只有012三个取值，所以用int
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int) # 训练集
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int) # 测试集

# 每行数据4个特征，都是real-value的
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# 构建一个DNN分类器，3层，其中每个隐含层的节点数量分别为10，20，10，目标的分类3个，并且指定了保存位置
# n_classes=3 表明是3分类问题
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")
											
# 指定数据，以及训练的步数
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)
			   
# 模型评估
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# 直接创建数据来进行预测
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))