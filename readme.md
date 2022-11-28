# Original GraphSage
最初的graphsage的对照组
包含``origin_sage.py``一个文件，简单实现了graphsage的training。

# Prototype
**研究问题**：当相同数据进行两次training后，能否通过添加label representation regularizer的方法保证下游frozen classifier能够复用？

三个文件是最初的原型代码：

- ``1aided_train.py`` 把label放入training集中训练sage，并且保存label representation和node representation
- ``2aided_classifier.py``训练下游logistic regression并且保存模型
- ``3aided_train.py``freeze label representation，再次训练sage，并且在frozen的下游logistic regression上做测试
这三个文件的储存下来的文件都放在了``model_data``里面。

# Label Representation Stability Exp
**研究问题**：多次训练的label representation正负两个label representation的距离是否足够稳定？。

包含以下两个文件：
- ``label_stability_train.py``训练100次label并且保存100次训练的label representation
- ``label_stability_calc_dist.py``计算100次label representation的距离
这2个文件的储存下来的文件都放在了``model_data``里面。

结论如下：
| 距离种类      | 正负label 距离 mean std     |
| ----------- | ----------- | 
| euclidean      | 16.83(0.27)      |
| cosine    | -0.02(0.02)       |
| dot   | -3.04(2.86)       | 

**结论**：label representation是稳定的，而且可以看到在 cos距离上的数据是最稳定而且最正交的。（正负两个label应当相反或正交）

# 全流程 full 实验
**研究问题**：当相同数据进行两次training后，能否通过添加label representation regularizer的方法保证下游frozen classifier能够复用？

- ``full100.py`` 把Prototype的三个步骤合并到一个里面，并且加入distance regularizer, 并且多轮化，查看平均数和方差获得实验数据。

regularizer只做了0.9、 0.7、0.5 几个实验，发现0.5效果最好，**下表结果是0.5 main loss和0.5 regularizer的performance**.

regularizer分为euclidean regularizer和cosine regularizer，50次的最后实验结果如下：

| 次      | exp settings      | logits (valid) predictor |  LR (test) classifier | gap mean std| note|
| ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- |
| 第一次      | 原sage (epoch ~200)      | 0.86   | 0.86 | -|
|     |        |
| 第二次   | 原sage (epoch ~200)       | 0.86 | 0.76| 0.18(0.07)
|    | label sage cosine (epoch ~5.5k, coef 0.5)      | 0.92+ | 0.826| 
|    | label sage euclidean  (epoch ~7k, coef 0.5)     | ~0.85 | 0.8244|



这个py文件没有储存数据，全部是在内存中的。

# 数据动态变化实验
**研究问题**：当动态数据变换时，添加label representation regularizer的方法还是否奏效？