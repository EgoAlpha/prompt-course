# **反射**

## 简介

[[Noah Shinn et al., 2023\]](https://arxiv.org/abs/2303.11366)提出了一种新颖的框架 Reflexion，通过语言反馈来加强语言代理。

通过语言方式对任务反馈信号进行反思，并将自己的反思文本保留在一个情节性记忆缓冲区中，以在后续的试验中促进更好的决策。

整合各种类型（标量值或自由形式语言）和来源（外部或内部模拟）的反馈信号，并在不同任务（顺序决策、编码、语言推理）中相比基线代理取得显著的改进。


## 原理

Reflexion将环境中的二进制或标量反馈转化为文本摘要的语言反馈，然后将其作为附加上下文添加到下一个片段中的LLM代理中。这种自我反思的反馈通过为代理提供具体的改进方向，充当了“语义”梯度信号，帮助代理从之前的错误中学习，以在任务中表现更好。如下图，Reflexion代理通过试验、错误和自我反思学习优化其自身行为，以解决决策、编程和推理任务。

![](pictures\1.png)



## Prompt 示例

### *Prompt*

```
You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 2, a diningtable 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: heat some mug and put it in coffeemachine.
```

### *Model Output*

```
> think: To solve the task, I need to find and take a mug, then heat it with stoveburner, then put it in coffeemachine.
Ok.
> look
You are in the middle of a room. Looking quickly around you, you see nothing.
> think: First I need to find a mug. A mug is more likely to appear in countertop (1-2), diningtable (1-2), drawer (1-3), fridge (1), sinkbasin (1), stoveburner (1-4), toaster (1). I can check one by one, starting with countertop 1.
Ok.
> look
You are in the middle of a room. Looking quickly around you, you see nothing.
> go to countertop 1
On the countertop 1, you see a butterknife 1, a cup 1, a fork 2, and a mug 1.
> think: Now I find a mug (1). Next, I need to take it.
Ok.
> take mug 1 from countertop 1
You pick up the mug 1 from the countertop 1.
> think: Now I take a mug (1). Next, I need to heat it with stoveburner (1-4).
Ok.
> go to stoveburner 1
On the stoveburner 1, you see a pan 2.
> examine stoveburner 1
On the stoveburner 1, you see a pan 2.
> examine stoveburner 1
On the stoveburner 1, you see a pan 2.
> examine stoveburner 1
On the stoveburner 1, you see a pan 2.
> examine stoveburner 1
On the stoveburner 1, you see a pan 2.
```

### *Output* 

```
STATUS: FAIL
New plan: I was stuck in a loop in which I continually examined stoveburner 1 instead of heating mug 1 with stoveburner 1. I should have taken mug 1 from countertop 1, then heated it with stoveburner 1, then put it in coffeemachine 1. It did not help to execute two identical actions in a row. I will try to execute a different action if I am stuck in a loop again. 
```

## 数据集

### [HotPotQA](https://arxiv.org/abs/1809.09600)
HotPotQA是一个基于维基百科的数据集，包含11.3万个问题和答案对，挑战代理对内容进行解析和推理。


## 参考文献

[1] Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., et al. (2022). [Do as i can, not as i say: Grounding language in robotic affordances. ](https://arxiv.org/abs/2204.01691)

[2] Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. (2021). [Program synthesis with large language models.](https://arxiv.org/abs/2108.07732)

[3] Chen, B., Zhang, F., Nguyen, A., Zan, D., Lin, Z., Lou, J.-G., and Chen, W. (2022). [Codet: Code generation with generated tests.](https://arxiv.org/abs/2108.07732)

