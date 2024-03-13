# 文章

[CLIP-KD: An Empirical Study of Distilling CLIP Models](https://arxiv.org/pdf/2307.12732.pdf)

## 这篇论文研究了一个什么样的问题

研究了用不同的蒸馏策略对CLIP进行蒸馏，得到一个更小的CLIP模型

## 关于这个问题，论文提出了什么样的解决思路或方法？

采用了不同的蒸馏策略，包括：relation paradigm，feature paradigm，gradient paradigm，contrastive paradigm来对教师CLIP模型进行蒸馏。在实验部分做了单个策略和组合策略的实验，来分析不同策略对蒸馏的效果以及策略之间是否存在互补性。

### relation paradigm：关系范式

对应论文中的Contrastive Relational Distillation方法。CRD的contrastive distribution能够更好的提取特征embedding之间的结构化关系，使得学生能更好地对教师的上下文关系进行模仿，提高表达特征的质量。

这个部分是将image embedding和text embedding先做对比学习，得到相似性矩阵，然后让学生模型的矩阵模仿教师的矩阵，这可能更多地涉及到对比学习的相关知识。

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/CRD.png" alt="CRD.png" width="450" data-align="center">

和CLIP相同，所有蒸馏策略下的loss都由image2text和text2image组成，对于CRD，首先求Contrastive Distribution，两个方向的学生模型和教师模型总共四个参数

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/image2text%20contrastive%20distribution.png" alt="image2text contrastive distribution.png" width="227" data-align="center">

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/text2image%20contrastive%20distributino.png" alt="text2image contrastive distributino.png" width="237" data-align="center">

然后根据下面的公式求分别的loss和总loss，

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/respective%20loss.png" alt="respective loss.png" width="325" data-align="center">

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/CRD%20loss.png" alt="CRD loss.png" width="419" data-align="center">

### Feature Distillation：特征范式

对应论文中的特征蒸馏（FD）和掩码特征蒸馏（MFD），也是常见的蒸馏方式（应该和soft target差不多吧）。

#### FD

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/FD.png" alt="FD.png" width="489" data-align="center">

就是让embeddings去拟合，直接降低教师和学生的知识差（论文里的说法，说人话就是拟合），使用MSE loss实现拟合，loss计算方式如下

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/FD%20loss.png" alt="FD loss.png" width="368" data-align="center">

#### MFD

在FD的基础上，在学生的输入图像中加入了掩码机制，将图像部分区域进行遮盖，即加入了自监督的机制（在后续实验中提到，MFD和FD的结果基本一致，所以个人认为自监督机制在这个任务中可能不是很重要，涉及到不同任务能提供不同的增益）。

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/MFD.png" alt="MFD.png" width="502" data-align="center">

计算loss和FD是一样的，公式如下

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/MFD%20loss.png" alt="MFD loss.png" width="440" data-align="center">

### Gradient Distillaiton：梯度范式

对应论文中的梯度蒸馏，论文中提到，梯度信息通常表现出模型对输入数据的响应。让学生拟合梯度信息，能够让其更好的理解输出应该如何根据输入而变化。

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/GD.png" alt="GD.png" width="440" data-align="center">

首先求出image gradient和text gradient，公式如下

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/GD%20i2t%20gradient.png" alt="GD i2t gradient.png" width="319">    <img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/GD%20t2i%20gradient.png" alt="" width="225">

其中**1**是指示函数，当k==b时，为1；反之为0。$p_k$和关系蒸馏中相同。

然后计算CLIP的Gradient Loss，公式如下

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/GD%20clip%20gradient.png" alt="GD clip gradient.png" width="334" data-align="center">

最后计算总loss

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/GD%20loss.png" alt="GD loss.png" width="335" data-align="center">

### Contrastive paradigm： 对比范式

根据GPT的说法，对比范式通过比较大型模型和小型模型的输出，强调他们在相似实例上的相似性，同时引入差异以提高不同实例的区分度。**不是很看得懂说实话**

论文中有两种用到了对比范式的蒸馏方式，分别为Interactive Contrastive Learning和Augmented Feature Distillation

#### Interactive Contrastive Learning：互动式对比学习

这个确实有点新奇，其他策略中都是学生的image embedding和教师的image embedding比，text同理。ICL是学生的image embedding和教师的text embedding比，示意图如下。整体的逻辑是互动的，计算方式和关系蒸馏一样。先计算contrastive contribution，然后用学生的contribution去拟合教师的contribution，以实现蒸馏。

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/ICL.png" alt="ICL.png" width="447" data-align="center">

loss的计算逻辑和FD相同，公式也基本一致

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/ICL%20I2T%20loss.png" alt="ICL I2T loss.png" width="487" data-align="center">

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/ICL%20T2I%20loss.png" alt="ICL T2I loss.png" width="499" data-align="center">

总ICL-loss就是half-half

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/ICL%20loss.png" alt="ICL loss.png" width="497" data-align="center">

#### Augmented Feature Distillation

在FD和ICL的基础上，加了一个Fusion Encoder，示意图如下

![AFD.png](X:\graduate\paper_study\G1\model%20compression\large%20model%20distillation\AFD.png)

Fusion Encoder的计算公式如下

<img title="" src="file:///X:/graduate/paper_study/G1/model%20compression/large%20model%20distillation/03181ea694e1dc44262ce78e2783266.png" alt="03181ea694e1dc44262ce78e2783266.png" width="362" data-align="center">

其中 **||** 是concatenation operator，就是拼接，就这样想想，真实缝合啊，这个可能就没有长度不一样的烦恼了。loss和其他逻辑一致

### 为什么这个思路或方法可以用于解决这个问题

因为蒸馏是模型压缩的其中一种方式。这篇论文创新性的采用了多种蒸馏策略，对CLIP进行了蒸馏。相较于对CLIP做的其他工作，本篇论文侧重于用KD对CLIP进行压缩而不是采用新的CLIP方法。

## 你认为论文的思路或方法可能存在什么问题？

<img title="" src="file:///C:/Users/mings/AppData/Roaming/marktext/images/2024-03-13-18-52-25-image.png" alt="" width="486" data-align="center">

总loss是不是可以再设计一下，感觉有点草率

还有一个就是超参数，老生常谈的话题

## 实验

<img src="file:///C:/Users/mings/AppData/Roaming/marktext/images/2024-03-13-18-56-51-image.png" title="" alt="" width="354">总共参与实验的有这么一些

首先是各种蒸馏策略的消融实验

![蒸馏策略消融实验.png](X:\graduate\paper_study\G1\model%20compression\large%20model%20distillation\蒸馏策略消融实验.png)

首先是CLIP-KD策略本身对学生模型的性能提升实验

![CLIP-KD性能实验.png](X:\graduate\paper_study\G1\model%20compression\large%20model%20distillation\CLIP-KD性能实验.png)


