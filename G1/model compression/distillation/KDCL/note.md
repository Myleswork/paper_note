KDCL方法，打破了传统知识蒸馏思想中的“强弱关系”，但与DML方法有所不同的是，KDCL对不同学生网络生成的软标签进行整合，作为所有学生模型的“额外信息”。对于不同的学生模型，数据集被随机打乱作为输入，这也能显著提高模型的泛化性。

<img title="" src="file:///C:/Users/mings/AppData/Roaming/marktext/images/2024-01-09-11-01-48-image.png" alt="" width="327" data-align="center">

KDCL旨在研究一种能够获得高质量集成软目标的方法，使其能够指导具有显著性能差距的学生模型始终以更高的泛化能力。而难点就在于如何有效地集成软目标。为此，本文采用了一些新机制以及新方法，概括如下：

1. 设计了一种全新的基于协作学习的知识蒸馏流程，这使得不同学习能力的模型均能够在协作学习中收益；
2. 设计了四种模型集成方法以在一阶段知识蒸馏框架中动态地生成高质量的软目标；
3. 对每一个学生模型的输入进行数据增强以及输出进行扰动，以增强模型的泛化性。

作者在文中提到，知识蒸馏实际上是通过教师和学生模型的软目标之间的KL散度来定义的，具体工作可见[5]。所以本文提出的四种方法，主要是通过优化logit来实现上述提到的工作。
作者进行了一系列实验来评估本文提出的训练方法，包括在基准图像分类上的实验以及在COCO数据集上的迁移实验。
在ImageNet数据集上进行的实验旨在评估本文提出的四种软目标集成方式对图像分类任务的影响。为了实现公平比较，所有模型都用相同的框架重新实现。结果如表4所示。
Table 4 Top-1 accuracy rate (%) on ImageNet.
表4 ImageNet上Top-1标准结果

<img title="" src="file:///C:/Users/mings/Desktop/5.jpg" alt="5.jpg" data-align="center" width="399">

文中对这部分实验的部分总结如下：DML可以为紧凑模型生成适当的软目标，但当存在显著的性能差距时，会损害复杂模型，因为紧凑模型的预测与复杂模型和基本事实相冲突。但按照我的理解，这部分实验并不足以支撑这一结论，虽然这一结论是正确的。
第二个实验是研究了不同网络架构的组合在ImageNet上的Top-1指标，结果如下表5所示。MBV2是MobileNetV2的缩写。MBV2x0.5表示宽度乘数为0.5。ResNet-50和ResNet-18被训练了100个时期。MBV2和MBV2x0.5被训练了200个时期。
Table 5  The comparative result of different sub-network on ImageNet validation set.
表5 不同子网在ImageNet验证集上的比较结果

<img title="" src="file:///C:/Users/mings/Desktop/6.jpg" alt="6.jpg" width="340" data-align="center">

从该实验的结果中我们可以看出，更紧凑的模型，例如MobileNetV2×0.5可以为MobileNetV2、ResNet-18甚至ResNet-50提供“知识”，因为紧凑型模型可以在一些样本上击败复杂模型。具有1.9M参数的MobileNetV2×0.5有助于提高具有25.6M参数的ResNet-50。事实证明，我们的方法适用于模型之间存在显著性能差距的情况。长时间的训练可以通过随机扭曲训练图像来提供更多不同的软目标，从而提高准确性。因此，对于另外100个训练时期，ResNet50和ResNet-18的Top-1进一步提高了0.8%和1.0%。
上述实验主要证明了使用两个子网络获得了两个结果。接下来作者希望证明集合更多的模型通常会提供更好的准确性，同时，实验过程中存在增益随着网络的增加而降低的问题，作者猜测是由于强集成网络和附加网络之间的相互信息随着集成规模的增加而增加。于是设计了下面的实验，在ImageNet上利用不同容量的神经网络进一步进行了实验，结果如表6所示，利用三个紧凑型模型的知识，ResNet-50达到了78.2%的Top-1。
Table 6  Top-1 accuracy rate (%) on ImageNet.
表6  ImageNet数据集的Top-1准确率

<img title="" src="file:///C:/Users/mings/Desktop/7.jpg" alt="7.jpg" width="468" data-align="center">

在CIFAR-10上进行的实验大致相同，说明KDCL在不同的任务中都能起到很好的辅助作用。
此外，作者还对通过KDCL策略训练得到的ResNet-18进行了迁移学习，结果如表7所示。
Table 7  Average precision (AP) on COCO 2017 validation set with pre-trained ResNet-18.
表7  在使用预训练的ResNet-18模型进行的COCO 2017验证集上的平均精度（Average Precision，AP）

<img title="" src="file:///C:/Users/mings/Desktop/8.jpg" alt="8.jpg" width="368" data-align="center">

从实验结果中可以得出，KDCL学习机制所带来的改进可以在广泛的任务和数据集中实现。
文章中的实验基本能够说明KDCL机制对写作学习的在线知识蒸馏技术带来的提升，但似乎缺少对四种方法改造部分的定量分析。此外，实验一似乎难以得出关于DML的相关结论。如果可以对这部分内容进行完善，实验部分的逻辑性会更强。
文章的结论部分主要概括了文章做出的贡献以及所做的实验等工作，并对该技术的应用领域进行了展望。KDCL主要解决了不同规模模型在进行在线蒸馏时的软目标冲撞问题，能够提高在线蒸馏框架的易用性和泛化性，属于比较有价值的改进型工作。
文献部分引用充分。
