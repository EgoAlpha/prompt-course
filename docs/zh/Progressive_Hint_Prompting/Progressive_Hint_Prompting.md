# **Progressive-Hint Prompting Improves Reasoning in Large Language Models**

## Introduction

[Chuanyang Zheng et al., 2023](https://arxiv.org/abs/2304.09797)认为类似于人类的认知，对于一个问题，LLM可以重新评估并调整其生成的推理路径，以纠正错误并提高整体性能。

虽然大型语言模型（LLMs）在各种NLP任务中有着显著的性能，但在思维推理方面，目前的大语言模型都有着一定的局限性，并且无法只通过增加模型的规模来克服。"思维链"（CoT）和自我一致性是目前解决这种缺陷的两种方法。思考链(CoT)通过用户进行一系列的线索提示促使LLM来进行逐步推理。与专注于设计更好的提示语的CoT相比，自我一致性则是从LLM那里获得多个答案，最终通过多数投票得出正确答案。

但之前的研究并没有探索利用LLM的输出来迭代完善推理路径的潜力。因此本文提出了一种名为Progressive-Hint Prompting（PHP）的新方法。PHP通过使用先前生成的答案作为提示，逐步引导LLM得出正确的答案，实现了用户和LLM之间的自动多次互动。



## How it Works?

在逻辑推理方方面，人类有着一个突出的优点，在推理过程中，我们不仅有能力通过思考得出问题的答案，而且在得出答案后还可以反复检查，确保答案是正确的。因此，作者认为这个过程可以通过依次采用以前的答案在LLM中进行模拟。

这个过程可以归纳为以下三个步骤： (1）给定一个问题，我们向LLM提问，得到一个基本的答案；（2）结合问题和答案，我们重新询问LLM并获得后续的答案；（3）我们重复（2）中的操作，直到答案趋于一致，并且在最后两个答案上没有变化。

此外，这个程序可以与现有的方法（CoT、自我一致性等）结合起来，进一步提高大型模型的性能。

主要过程如下图所示：

![](picture\PHP.png)



## Prompt 示例

### *Prompt*
```
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? (Hint: The answer is near to 9, 5). 

A: We know the Answer Hints: 9, 5. With the Answer Hints: 9, 5, we will answer the question. Shawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 * 2 = 4 more toys. Now he has 5 + 4 = 9 toys. The answer is 9. 

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? (Hint: The answer is near to 20). 

A: We know the Answer Hints: 20. With the Answer Hints: 20, we will answer the question. There were originally 9 computers. For each day from monday to thursday, 5 more computers were installed. So 4 * 5 = 20 computers were added. Now 9 + 20 = 29 computers are now in the server room. The answer is 29. 

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? (Hint: The answer is near to 45). 

A: We know the Answer Hints: 45. With the Answer Hints: 45, we will answer the question. Michael started with 58 golf balls. He lost 23 on Tuesday, and lost 2 more on wednesday. So he had 58 - 23 = 35 at the end of Tuesday, and 35 - 2 = 33 at the end of wednesday. The answer is 33. 

Q: Olivia has \$23. She bought five bagels for \$3 each. How much money does she have left? (Hint: The answer is near to 8). 

A: We know the Answer Hints: 8. With the Answer Hints: 8, we will answer the question. Olivia had 23 dollars. She bought 5 bagels for 3 dollars each. So she spent 5 * 3 = 15 dollars. Now she has 23 - 15 = 8 dollars left. The answer is 8. 

Q:Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?

```

### *Output*

```

If Wendi feeds each chicken three cups of feed in each meal and she has a flock of 20 chickens, then the total amount of feed needed for each meal is 3 cups/chicken * 20 chickens = 60 cups of feed.

In the morning, she gives 15 cups of feed, and in the afternoon, she gives another 25 cups of feed. Therefore, the total amount of feed given in the first two meals is 15 cups + 25 cups = 40 cups of feed.

To find the remaining amount of feed needed for the final meal of the day, we subtract the amount already given from the total feed needed: 60 cups - 40 cups = 20 cups of feed.

Therefore, Wendi needs to give her chickens 20 cups of feed in the final meal of the day.
```

### *Prompt*

```
Q:Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?***(Hint: The answer is near to 20)***.
```

### *Output*

```
We know the Answer Hint: 20. With the Answer Hint: 20, we will answer the question.

Wendi gives each chicken three cups of feed in each meal, and she has a flock of 20 chickens. So the total amount of feed needed for each meal is 3 cups/chicken * 20 chickens = 60 cups of feed.

In the morning, she gives 15 cups of feed, and in the afternoon, she gives another 25 cups of feed. So the total amount of feed given in the first two meals is 15 cups + 25 cups = 40 cups of feed.

To find the remaining amount of feed needed for the final meal of the day, we subtract the amount already given from the total feed needed: 60 cups - 40 cups = 20 cups of feed.

Therefore, Wendi needs to give her chickens 20 cups of feed in the final meal of the day. The answer is 20.

```

### *Prompt*

```
Q:Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?***(Hint: The answer is near to 240)***.
```

### *Output*

```
We know the Answer Hint: 240. With the Answer Hint: 240, we will answer the question.

Let's start with the information given: Seattle has 20 sheep.

According to the given information, Charleston has 4 times as many sheep as Seattle. So Charleston has 4 * 20 = 80 sheep.

And Toulouse has twice as many sheep as Charleston. Therefore, Toulouse has 2 * 80 = 160 sheep.

To find the total number of sheep Toulouse, Charleston, and Seattle have together, we add up the number of sheep for each person: 20 + 80 + 160 = 260 sheep.

Therefore, Toulouse, Charleston, and Seattle have a total of 260 sheep together. The answer is 260.
```
### *Prompt*

```
Q:Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?***(Hint: The answer is near to 240,260)***.
```
### *Output*

```
We know the Answer Hints: 240, 260. With the Answer Hints: 240, 260, we will answer the question.

According to the given information, Charleston has 4 times as many sheep as Seattle. Since Seattle has 20 sheep, Charleston has 4 * 20 = 80 sheep.

Toulouse has twice as many sheep as Charleston. Therefore, Toulouse has 2 * 80 = 160 sheep.

To find the total number of sheep Toulouse, Charleston, and Seattle have together, we add up the number of sheep for each person: 20 + 80 + 160 = 260 sheep.

Therefore, Toulouse, Charleston, and Seattle have a total of 260 sheep together. The answer is 260.
```


## 数据集

### [AddSub](https://aclanthology.org/D14-1058/)
AddSub数据集是一个用于测试算术电路的数据集，包含了加减法器的测试用例。

### [MultiArith](https://arxiv.org/pdf/1608.01413/)
MultiArith数据集是一个用于测试算术推理的数据集，包含了多种算术运算，如加、减、乘、除等。

### [SingleEQ](https://aclanthology.org/Q15-1042)
SingleEQ数据集是一个用于测试单方程代数问题的数据集，包含了对非负有理数和一个变量进行多项数学运算的单方程代数单词问题。

### [SVAMP](https://arxiv.org/abs/2103.07191)
SVAMP是一个用于测试硬件验证的数据集，包含了各种硬件验证问题，如等价性检查、模型检查等。

### [GSM8K](http://export.arxiv.org/abs/2110.14168)
GSM8K是一个用于测试汉语音素转换的数据集，包含了各种汉字和拼音之间的映射关系。

### [AQuA](https://arxiv.org/pdf/1705.04146/)
AQuA是一个用于测试汇编代码理解的数据集，包含了各种汇编代码和问题之间的映射关系。


## 参考文献

[1] Daniel W Otter, Julian R Medina, and Jugal K Kalita. [A survey of the usages of deep learning for natural language processing.](https://arxiv.org/pdf/1807.10854.pdf) IEEE transactions on neural networks and learning systems, 32(2):604–624, 2020.

[2] Xipeng Qiu, Tianxiang Sun, Yige Xu, Yunfan Shao, Ning Dai, and Xuanjing Huang. [Pre-trained models for natural language processing: A survey.](https://arxiv.org/abs/2003.08271) Science China Technological Sciences, 63(10):1872–1897, 2020.

[3] KR1442 Chowdhary and KR Chowdhary. [Natural language processing.](https://link.springer.com/chapter/10.1007/978-81-322-3972-7_19) Fundamentals of artificial intelligence, pages 603–649, 2020.









