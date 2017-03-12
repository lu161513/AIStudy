# Tensorflow Project2 对评论分类

###### 着重以下两点
- 训练好的模型的保存和使用
- 训练方法的选择
    - feedforward神经网络
    - CNN卷积神经网络

###### 数据集

[我是数据集](http://help.sentiment140.com/for-students/)

training.1600000.processed.noemoticon.csv文件的**字段**如下：

- 0 – the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
- 1 – the id of the tweet (2087)
- 2 – the date of the tweet (Sat May 16 23:58:44 UTC 2009)
- 3 – the query (lyx). If there is no query, then this value is NO_QUERY.
- 4 – the user that tweeted (robotickilldozr)
- 5 – the text of the tweet (Lyx is cool)


---
### 预处理
使用process.py进行数据预处理

process.py中的代码把原始数据转为training.csv、和tesing.csv，里面只包含label和tweet。lexcion.pickle文件保存了词汇表。

### 训练（两种方式）
- 使用train-feedforward.py进行训练
- 使用train-cnn.py进行训练

训练模型保存为model.ckpt。

### 使用模型
使用guess.py进行猜测