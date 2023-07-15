# **连续提示**

## 简介

回答含有隐含决策的复杂问题是一项困难的任务，特别是当训练数据有限时。最近的研究利用大型语言模型在少样本学习设置中执行复杂问题回答，生成中间推理步骤并解决。[[Dua 等人., 2022]](https://arxiv.org/abs/2212.04092) 提出了一种名为“连续提示”的新方法，该方法迭代地将复杂问题分解为简单问题并解决，然后重复该过程，直到得出最终解决方案。连续提示将问题分解和问题回答的监督分离，使我们能够：(1)在每个推理步骤中有多次查询上下文示例的机会；(2)分别学习问题分解和问题回答，包括使用合成数据；(3)在大型语言模型表现不佳的推理步骤中使用专门的微调组件。

作者还提出了一种生成合成数据的方法，以提高模型分解和回答中间问题的能力。与具有相同监督的最先进模型相比，所提出的模型在少样本版本的DROP数据集上F1绝对值提高了约5％。


## 原理

连续提示提出了一种将复杂问题分解为简单问题、分别回答并重复此过程直至完整回答复杂问题的方法。我们将每个潜在的步骤表示为一对简单的问答，z_k=(q_k,a_k)，而不像CoT将每个潜在步骤表示为陈述句。这种方法给了我们多次提示L的机会，每个步骤都可能有不同的语境例子，更适合简单的问题。这也使得我们可以在给定中间状态z_k的情况下重新编码上下文。如图我们将第一类输出称为问题分解(QD)，第二类输出称为问答(QA)。我们将最终答案预测视为问题分解的一个特例，其中模型决定不再需要分解并输出最终答案，因此我们在问题分解和问答之间迭代交替，直到模型终止。

在语境学习过程中，在测试输入之前，少量的训练样本直接提供给大模型LM。这些例子是根据它们与测试输入的相似性从一个指标中选择的。为了连续提示，我们创建了两个指标：I_D，用于查询QD的相关演示；I_A，用于查询QA的相关演示。索引I_D每一步k都包含部分分解的链，表明对于训练数据中的每一个复杂问题都将产生下一个问题q_k。

在QD阶段，索引I_D询问复杂的测试问题q和当前步数k，以选择如何为举出的示例生成下一个问题的演示。在QA阶段，利用QD过程中产生的简单问题q_k对指标I_A进行查询，选择相关的简单QA对。


![](pictures\1.png)



## Prompt 示例

### *Prompt*

```
Q: 谁投出了最长的触地得分传球？

Q1: 有哪些触地得分传球？
A1: 22码，8码。

Q2: 在22码和8码中，哪个值最大？
A2: 22码。

Q3: 谁投出了22码的触地得分传球？
A3: Peyton Manning。

Q: 没有更多问题需要问了。
```

### *Output* 

```
Peyton Manning.
```

## 数据集

### [DROP](https://aclanthology.org/N19-1246/)
一个广泛用于问答和阅读理解任务的数据集，旨在测试模型对于多步骤推理和数值推理的能力。数据集的来源是维基百科的文章和问题回答网站中的问题。每个样本由一个段落和一个与段落相关的问题组成 ，一些问题需要多步骤的推理和分析才能得到答案。

### [QQ-P](https://aclanthology.org/D19-1410/)
一个用于问题对匹配任务的常用数据集，旨在帮助判断两个问题是否具有相似的语义意义， 数据来源于Quora社区，包含了大量的非重复问题对和一部分重复问题对。


## 参考文献

[1] Aishwarya Agrawal, Dhruv Batra, Devi Parikh, and Aniruddha Kembhavi. 2018. [Don’t just assume; look and answer: Overcoming priors for visual question answering.](https://ieeexplore.ieee.org/document/8578620) In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages 4971–4980. IEEE Computer Society.

[2] Daniel Andor, Luheng He, Kenton Lee, and Emily Pitler. 2019. [Giving BERT a calculator: Finding operations and arguments with reading comprehension.](https://aclanthology.org/D19-1609/) In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5947– 5952, Hong Kong, China. Association for Computational Linguistics.

[3] Xinyun Chen, Chen Liang, Adams Wei Yu, Denny Zhou, Dawn Song, and Quoc V. Le. 2020.[ Neural symbolic reader: Scalable integration of distributed and symbolic representations for reading comprehension.](https://aclanthology.org/D19-1609/) In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net










