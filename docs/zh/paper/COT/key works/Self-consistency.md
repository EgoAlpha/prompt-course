﻿# Prompting extension->Self-consistency

## Self-Consistency Improves Chain of Thought Reasoning in Language Models

## 介绍

[\[Wang et al. (2023)\]](https://arxiv.org/abs/2203.11171) 提出了一种新的解码策略——自我一致性解码，以取代在思维链提示中使用的简单贪婪解码。它首先采样一组不同的推理路径，而不是只选择贪婪的一条，然后通过边缘化采样的推理路径来选择最一致的答案。大量的实证评估表明，自我一致性提高了思维链提示的性能，在一系列流行的算术和常识推理基准上有惊人的优势，包括GSM8K (+17.9%)、SVAMP (+11.0%)、AQuA (+12.2%)、StrategyQA (+6.4%)和ARC-challenge (+3.9%)。

## 它是如何工作的？

已经观察到，在各种多步骤推理任务中，思维链提示显著提高了模型性能。在本文中，我们引入了一种新的解码策略，称为自我一致性，以取代思维链提示中使用的贪婪解码策略 (Wei et al., 2022)，进一步提高了语言模型的推理性能。自我一致性利用了这样的直觉，即复杂的推理任务通常允许多条推理路径达到正确的答案 (Stanovich & West, 2000)。一个问题越需要深思熟虑的思考和分析(Evans，2010)，能够找到答案的推理路径的多样性就越大。

我们首先用思维链提示语言模型，然后不采用贪婪地解码最优推理路径，而是提出一个“采样-边缘化”解码过程:我们首先从语言模型的解码器中采样，以生成一组不同的推理路径；每条推理路径可能导致不同的最终答案，因此我们通过边缘化采样的推理路径来确定最佳答案，以在最终答案集中找到最一致的答案。这种方法类似于人类的经验，即如果多种不同的思维方式导致相同的答案，那么一个人就更有信心最终的答案是正确的。与其他解码方法相比，自我一致性避免了困扰贪婪解码的重复性和局部最优性，同时减轻了单个采样带来的随机性。

![](..\images\Self_consistency.png)

提示示例：

*提示:*

*问:小树林里有15棵树。格罗夫工人今天将在格罗夫植树。完成后，将会有21棵树。格罗夫工人今天种了多少树？*

*我们从15棵树开始。后来我们有了21棵树。差别一定是他们种植的树的数量。所以，他们一定种了21 - 15 = 6棵树。答案是6。*

*问:如果停车场有3辆车，又有2辆车到达，停车场有多少辆车？*

*停车场已经有三辆车了。又来了两个。现在有3 + 2 = 5辆车。答案是5。*

*问:我6岁的时候，姐姐只有我一半大。现在我70了，我妹妹多大了？*

*答:*

*⭐Sample的一套多样化的推理路径如下:*

*输出1:当我6岁的时候，我姐姐只有我一半大，所以她3岁。现在我70，所以她70 - 3 = 67。答案是67。*

*输出2:当叙述者6岁时，他的妹妹是他的一半年龄，也就是3岁。现在叙述者已经70岁了，他的妹妹应该是70 - 3 = 67岁。答案是67。*

*输出3:当我6岁的时候，我姐姐只有我一半大，所以她3岁。现在我70，所以她70/2 = 35。*

*答案是35。*

*⭐majority投票:*

*答案是67。(最终答案)*
