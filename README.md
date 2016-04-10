UFLDL exercises
===============

UFLDL有两个版的教程页面, [新版](http://ufldl.stanford.edu/tutorial/), [旧版(有中文版)](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial), 这两个的内容是一样的, 第一个是更新的一个版本, 刚开始本人follow的是第一个, 但目前没有整理完善, 组织顺序也很奇怪, 后来才转到第二个.

ex1和nn是follow第一个时完成的, 后面的是follow第二个时完成的.

第一版中提供的[start code](http://ufldl.stanford.edu/tutorial/StarterCode)将所有UFLDL练习代码进行了整合, 用这个start code感觉更好些, 用这个代码的结构follow旧版的练习可以减少一些冗余, 只是导入数据的方式有一点点不一样.

已完成的有:

+ ex1, 单层回归:
	+ [线性回归](http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/)
	+ [逻辑回归](http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression/)
	+ [向量化编程](http://ufldl.stanford.edu/tutorial/supervised/Vectorization/)
	+ [Softmax回归(MNIST)](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)
+ nn, [两层神经网络(MNIST)](http://ufldl.stanford.edu/tutorial/supervised/ExerciseSupervisedNeuralNetwork/)
+ sparse_autoencoder, [稀疏自编码器](http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder)
+ pca, [主成分分析](http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening), 这一节内容还包括pca白化, zca白化
	+ [2d](http://ufldl.stanford.edu/wiki/index.php/Exercise:PCA_in_2D)
	+ [pca和白化](http://ufldl.stanford.edu/wiki/index.php/Exercise:PCA_and_Whitening)
+ [softmax回归](http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression), 这一节与ex1中略有有重复, 这一节代码要为下一节stl服务, 需要完成
+ [Self-Taught Learning](http://ufldl.stanford.edu/wiki/index.php/Exercise:Self-Taught_Learning)
+ [自编码线性解码器](http://ufldl.stanford.edu/wiki/index.php/Exercise:_Implement_deep_networks_for_digit_classification), (this case is NOT done well.)

以上除了nn是用python编写, 其它均为matlab.

略去各数据集, minfun等部分.