# TensorFlow Project1: 对评论进行分类

###### 需要以下资料
- neg.txt：[5331条负面电影评论](http://blog.topspeedsnail.com/wp-content/uploads/2016/11/neg.txt) 
- pos.txt：[5331条正面电影评论](http://blog.topspeedsnail.com/wp-content/uploads/2016/11/pos.txt)

###### 核心想法
- 处理的是字符串，要把字符串转换为向量/数字表示。将出现的单词映射为数字ID
- 每行评论字数不同，而神经网络需要一致的输入(其实有些神经网络不需要，至少本帖需要)，这可以使用词汇表解决。 python的nltk库