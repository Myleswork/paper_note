# 二看transformer

为了学习ViT和swin transformer这些视觉架构，对attention和transformer有过两周时间的学习，零零散散地也算是能看懂了，但是就像笔记里所展示的那样，很散，串不起来，所以准备用几天时间把transformer的这些知识串起来，有一个整体的理解。

本人想要搞清楚的问题

- [x] embedding究竟是什么（暂时不打算**深究**了，毕竟是纯NLP的内容）

- [x] Q、K、V究竟是什么（01.11）

- [x] multi-head attention究竟是什么

- [x] encoder和decoder究竟是什么

- [ ] encoder和decoder之间传递的信息是什么（half-half）

## embedding

首先在NLP中，上来就是embedding，自从听说不同词的embedding是train出来的，我就一直处于深深的不理解当中。后来我才知道，不是embedding是train出来的，而是这个产生embedding的转换矩阵是train出来的。

这个转换矩阵用于将稀疏矩阵，通过线性变换（在CNN中用全连结层进行转换），变成一个密集矩阵。

这个线性变换其实就是矩阵乘法呗，这个词嵌入转换矩阵是对语料库进行抽象，提取一些共有的特征，形成的矩阵。可以看看这个[例子](https://www.cnblogs.com/USTC-ZCC/p/11068791.html)，例子对语料提取的特征就是“皇帝”，“宫里”和“女”。当然，这也仅仅只是一个例子，实际对语料库进行训练的时候，肯定没有这么具象。按照我搞NLP的同学的说法，在word2vec之前，还有一个word2id的过程，比如按照词频，对语料库中的词（对于英文来说一般是一个个单词；对于中文来说可能是字，也可能是词，一般会有分词的过程）进行排序，然后按顺序编号，该编号组成的系数矩阵，和转换矩阵相乘（或者说就是线性变换），产生一个存在内在联系（对于抽象出的特征来说）的密集矩阵，也就是embedding。最后肯定要来看看维度，既然是对语料库进行抽象，那么势必会减少存储需求。举个例子，假如语料库中有3000个词，删除一些停用词，加入unknown编号，最后剩下来2800个词，这个word2id还要看一看， 怎么编码的。

这个词嵌入矩阵是怎么训练出来的？

按照目前的理解，首先要有语料库和训练算法。语料库中有用的是上下文，通过上下文的**某些信息**，来进行聚类（其实就是抽象出一些特征，将one-hot组成的稀疏矩阵降维成密集矩阵），那么上面提到的某些信息，就称为了聚类的依据。

在[李宏毅的视频](https://www.youtube.com/watch?v=X7PH3NuYW0Q&t=655s)中，提到了两种训练理念，分别是count-based和prediction-based。

**count-based**，顾名思义，就是两个词之间共同出现的频率决定他们的距离（聚类嘛），假设在二维坐标上就是靠的近和远。但是好像很少看见这种方法？可能是我了解不多，视频中提到了哈佛大学的[GloVe](https://nlp.stanford.edu/projects/glove/)。

感觉比较常见的是**prediction based**，就是用上下文来进行预测，有用前文预测后文的，有用上下文预测中间文的，有用中间文预测上下文的。大概就是这么个意思。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/b4b4e8e3a60f49e3a93dc98e8887721e.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/c361b823eae84043a32c57159731e981.png)

总之，最后训练得到一个维度更小的词嵌入转换矩阵（或者说是线性变换啦），和one-hot组成的矩阵相乘，得到embedding。

上面这部分在[论文](http://arxiv.org/abs/1706.03762)中就是*input embedding*层

## position encoding

$NLP$的$pe$我个人认为还是好理解的，$ViT$的$pe$我是真觉得挺神奇的哈哈哈，可能理论上是一样的吧。首先$pe$有很多种，包括绝对位置编码，相对位置编码；又有固定编码和可学习编码。transformer里用的应该是三角函数式位置编码，是一种绝对位置编码（没记错的话），具体的可以看一看[这篇文章](https://0809zheng.github.io/2022/07/01/posencode.html)。反正就是解决attention机制无法获取位置信息的问题。

但是这是直接加上去（add，不是concat）的，确实神奇。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/786dedde3f2d4d03b06c3446d282f069.png)

## Attention机制

先放两篇个人认为讲的比较好的blog，从两方面讲解了attention机制，合起来就比较全面了。[blog1](https://blog.csdn.net/weixin_42392454/article/details/122478544?spm=1001.2014.3001.5502)和[blog2](https://zhuanlan.zhihu.com/p/410776234)

我个人认为，理解transformer最核心的部分，也就是attention，核心在于理解其中矩阵变换的集合意义，通俗来说就是向量内积。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/e04be2ed207d4cb8b18d7a8cfa4a4340.png)

那对向量求内积的几何意义是什么，为什么通过矩阵（就是由向量组成的嘛，一样的）相乘就能获得我们想要的词与词之间的相关性强弱呢？

对向量求内积，几何意义是表征两个向量的夹角，表征一个向量在另一个向量上的投影[2]，**在embedding处理后**，所有词都通过抽象出来的特征产生**内在联系**（个人认为，有了这个前提才能用下面的结论，没有内在联系的向量之间作内积是没有意义的，得到的数值也说明不了任何问题），从数据上来说，不同词的embedding的相同列表示的是和某个特征的相关性，换句话说就是他们可比较（和one-hot不同）。这就意味着不同词的embedding向量可以通过求内积判断他们的相关性，内积越大，说明他们的相关性越高；反之则越小。这个相关性，就是我们想要的attention，也就是说在处理当前词时，应该要给予和它相关性高的词更多的关注。

![](https://pic2.zhimg.com/80/v2-f6973006b0ca2b67f452439698e6aacd_1440w.webp)

上面这部分内容其实就解释了$QK^T$的集合意义，也就是求出了某一个word和所有word的相关性。可以用一张图来形象地展示。

<img title="" src="https://pic3.zhimg.com/80/v2-f85c81cbb259b80c3644a16e005679be_1440w.webp" alt="" width="334" data-align="center">

那再来看和*V*相乘，这个我感觉不是那么好理解，我们将$QK^T$记作P，可以理解为每个字符之间两两相似度，可以理解为权重矩阵（softmax后）。而V其实就可以理解为所有字符embedding构成的矩阵P和V相乘，得到加权求和后的embedding，因为对于字符A来说，其和语料库中的其他字符的相关性是不同的，那么对于字符A更关注的字符来说，他们的embedding就应该更突出，可以理解一下这里的“突出”，应为embedding本质上来说是某一个字符相对词嵌入矩阵抽象出来的特征对应的稀疏，如果字符A和字符B的相关性更高，即我在endoding时需要更加关注字符B，那么我就需要去提高他的相应的权重，而字符B相对于语料库特征的权重是固定的，我们只能通过字符A相对于其他字符（可以以字符B为例）的相关性权重强弱来改变embedding，从而达到上面我们所提到的更关注某一些字符的能力。这就解释了为什么需要$QK^T$乘V。

不知道我有没有表达清楚，也不知道我理解的对不对，但至少我把我自己给说服了哈哈哈。那么最终我们获得的b1，其实就是某个字符对其他所有字符在self-attention机制下的embedding，这个embedding就能去匹配输出（当然我记得实际的transformer里有好多层self-attention）。

那么写到这里为止，其实也大概能悟到Q，K，V究竟是啥了，从本质上来说，他们其实是一样的，就是语料库中所有word对应的embedding；从维度上来说，他们也是一样的，是A分别和三个相同维度的可训练的矩阵相乘得到的。而$QK^T$的是为了求embedding之间的内积，几何意义就是角度，实际意义就是两个embedding的相关性。

$(QK^T)V$就是让某一个字符相对于其他字符的相关性系数（或者说权重）加诸于其他字符的embedding之上，之所以能够这样做的原因，是因为embedding就是字符对应语料库特征的系数！

![sum](https://img-blog.csdnimg.cn/b901cb864e444f60b980863b8be2e89c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA54Ot6KGA5Y6o5biI6ZW_,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

微观（bushi）的计算过程就如上图所示了，还是比较清晰的。

## multi-head attention

multi-head attention其实和cnn中的多个filter是一个道理，用n个filter去提取特征，然后将得到的抽象特征进行concat，至于说为什么不同的filter能够提取到不同的特征（不知道这么说是否严谨，可能也有不同的filter提提取到大致相同的特征），有点黑盒，后续在深挖一下。

那multi-head其实就是一个逻辑，多个head能够抽取不同的信息，将抽取到的信息concat，能够提升模型性能。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/369c5c9deaeb4be2a30b03e11f0401ff.png)

## Encoder＆Decoder

首先需要明确的是，Encoder和Decoder并不是特定的模型或者网络结构的名称，而是一种通用的框架。只要是符合“编码-解码”的，都可以称之为Encoder-Decoder结构，比如CNN，卷积操作就是encoder，将图像转换成一些数字；最后的dense部分就是decoder，将卷积部分得到的一系列数字vector，转换成我们想要的输出，比如说分类任务那就是input对应的类别。Encoder和Decoder部分可以是针对不同数据类型的不同模型，像现在的多模态，是吧。

Encoder，也就是编码，就是把不定长的输入序列转化成一个固定长度向量。如果我没记错的话，transformer中会插入一些冗余序列来实现得到一个固定长度向量。

Decoder，也就是解码，即将Encoder生成的固定长度向量转换成不定长的输出序列。

上面那么大篇幅讲的也都是Encoder，很少看到文章仔细来讲transformer的decoder，也就是怎么把encoder生成的固定长度向量转换成不定长的输出的（当然也不排除我看得少）。

encoder部分可并行（这也是attention机制的一个优势），decoder是不可并行（是不是也分训练的时候和推理的时候）的，这也是非常显然的，因为下一个输出需要依赖前文。

那回到transformer，先来看下面这张经典的结构图

<img title="" src="https://img-blog.csdnimg.cn/direct/f3caa21f71ae4f299adc821f963f2209.png" alt="在这里插入图片描述" width="517" data-align="center">

首先要说明的是，这张图是训练过程的示意图，如果拿推理的逻辑去看这张图的话，那问题就多了，主要集中在Decoder部分。比如，Outputs为什么是去输入的？Output为什么会能embedding？Output为什么能$pe$？所以先按训练的逻辑来看这张图，拿一个实际的例子来解释，会更清楚。Decoder部分也会单独讲。

以翻译任务为例，我们想要把依据英文，例如“Hello world”翻译为“你好世界”，那么在Encoder部分，Inputs就是"Hello world"的one-hot矩阵(当然是经过一些处理的，理解上没有差异)，经过embedding以及pe操作后，送入灰块块进行self-attention计算，最后得到一个hidden state。这个hidden state就包含了句子中词与词之间的隐藏信息，将这个隐藏信息送入decoder，用于“你好世界”上下文之间的训练。

也就是什么意思呢，在训练过程中，encoder的输入是原始数据，比如就是“hello world”；而decoder部分的输入，是"pos"+翻译之后的内容，比如就是"pos 你好 世界"，当然，在训练时，当前字符后面的字符是被mask掉的，代码实现我记得就是把当前code(或者叫token？)后面的code的权重设置为0（也就是不让它们更新），而decoder部分的GT是什么呢，是翻译之后的内容+"EOS"。这样的话训练过程基本上就清晰了，这部分论文里我记得并没有提到（也可能是我漏看了），所以云里雾里。

那么推理部分和训练部分的区别就在于，我的模型现在有原始数据的语料库hidden state和对应翻译结果的语料库hidden state，然后有**原始数据**和**POS**，然后将原始数据输入进encoder进行相同的处理，在当前字符走完一遍流程之后将hidden state送入decoder，作为query和key，decoder部分将当前字符（一开始是POS，对吧，那么后面就有生成的字符了）送入multi-head attention（这里就不需要mask了，因为本身也不知道后面的字符是啥，是要预测的）和残差结构，输出的q,k,v作为下一个多投结构的输入，当然要和encoder过来的q和k一起（可能是concat），走完剩下的流程就能输出当前字符的下一预测字符，然后接上预测字符作为下一个循环的输出，直到预测输出为"EOS"，那么预测就结束了。

其实这部分逻辑和seq2seq是一样的，无非就是transformer用的是self-attention，seq2seq用的RNN罢了。可以先去理解一下seq2seq，对理解transformer有很大的帮助。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/5c920cb8f2e447f282c6ad2dbd44324b.png) 

## encoder和decoder之间传递的信息是什么

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/6b9e4f0e3e26420c8cb78bd1f16d14ba.png)

传递的是当前已有sentence的hidden state，这个还可以探究一下，先放一放。

## 参考材料

[1][Attention is all you need](http://arxiv.org/abs/1706.03762)

[2][超详细图解Self-Attention](https://zhuanlan.zhihu.com/p/410776234)

[3][X都能看懂的Self-Attention讲解](https://blog.csdn.net/weixin_42392454/article/details/122478544?spm=1001.2014.3001.5502)

[4][Transformer中的位置编码(Position Encoding)](https://0809zheng.github.io/2022/07/01/posencode.html)

[5][Transformer源码详解（Pytorch版本）](https://zhuanlan.zhihu.com/p/398039366)

[序列到序列学习（seq2seq）【动手学深度学习v2】](https://www.bilibili.com/video/BV16g411L7FG/?spm_id_from=333.999.0.0&vd_source=9491ac033fe23c13918015db536d84db)

[Transformer【动手学深度学习v2】](https://www.bilibili.com/video/BV1Kq4y1H7FL/?spm_id_from=333.999.0.0&vd_source=9491ac033fe23c13918015db536d84db)
