# **ReAct Prompting**

## 简介

[Yao et al., 2022(opens in a new tab)](https://arxiv.org/abs/2210.03629)引入了一个名为ReAct的框架，其中LLM用于以交错的方式生成**推理跟踪**和**任务特定动作**。
生成推理跟踪允许模型归纳、跟踪和更新行动计划，甚至处理异常。行动步骤允许与外部来源（如知识库或环境）对接并从外部来源收集信息。

ReAct框架可以允许LLM与外部工具交互，以检索更多信息，从而获得更可靠、更真实的响应。

React框架是一种通用的范式，它将推理和动作与语言模型相结合，以解决各种语言推理和决策任务。

结果表明，ReAct在语言和决策任务方面可以优于几个最先进的基线。ReAct还提高了LLM的人类可解释性和可信度。总体而言，作者发现，最佳方法使用ReAct与思维链（CoT）相结合，允许使用推理过程中获得的内部知识和外部信息。

实验结果表明，COT和React的组合明显优于单独使用任何一种方法。


## 原理

ReAct提示LLM以交错的方式生成与任务相关的言语推理痕迹和动作，这允许模型执行动态推理，以创建、维护和调整高级行动计划（行动的理由），同时还与外部环境（如维基百科）交互，将额外信息纳入推理（行动的原因）。

思维链（CoT）提示显示了LLM进行推理追踪的能力，以生成涉及算术和常识推理的问题的答案，以及其他任务[（Wei et al.，2022）（在新的选项卡中打开）](https://arxiv.org/abs/2201.11903)。但它无法访问外部世界或无法更新其知识，可能会导致事实幻觉和错误传播等问题。

ReAct是一种将推理和行动与LLM相结合的通用范式。ReAct提示LLM为任务生成言语推理痕迹和动作。这允许系统执行动态推理，以创建、维护和调整行动计划，同时还允许与外部环境（例如维基百科）进行交互，将额外信息纳入推理中。下图显示了ReAct的示例以及执行问题回答所涉及的不同步骤。

![react.png](img/react.png)



## Prompt 示例

### *Prompt*

```
3687 Claim: Paramore is not from Tennessee.
Thought 1: I should search for Paramore, and see if it is from Tennessee.
Action 1: Search[Paramore]
Observation 1: Pages for logged out editors learn more. Paramore is an American rock band from Franklin, Tennessee, formed in 2004. The band currently consists of lead vocalist Hayley Williams, guitarist Taylor York and drummer Zac Farro. Williams and Farro are founding members of the group, while York, a high school friend of the original lineup, joined in 2007. The band is signed to Fueled by Ramen, a subsidiary of Atlantic Records, both owned by Warner Music Group.

Thought 2: The observation says that the band is "from Franklin, Tennessee", so the claim is false.
Action 2: Finish[REFUTES]
Observation 2: Episode finished, reward = 1


{'steps': 2, 'answer': 'REFUTES', 'gt_answer': 'REFUTES', 'question_idx': 3687, 'em': 1, 'reward': 1, 'f1': 1} 

1 1 1.0 4.081380128860474
-----------

6238 Claim: Eric Church is a sound mixer.
Thought 1: I need to search Eric Church and see if it says he is a sound mixer.
Action 1: Search[Eric Church]
Observation 1: Pages for logged out editors learn more. Kenneth Eric Church (born May 3, 1977) is an American country music singer-songwriter. He has released nine studio albums through Capitol Nashville since 2005. His debut album, 2006's Sinners Like Me, produced three singles on the Billboard country charts including the top 20 hits "How 'Bout You", "Two Pink Lines", and "Guys Like Me".. His second album, 2009's Carolina, produced three more singles: "Smoke a Little Smoke" and his first top 10 hits, "Love Your Love the Most" and "Hell on the Heart".

Thought 2: The observation does not say if he is a sound mixer, so I cannot say if the claim is true or false.
Action 2: Finish[NOT ENOUGH INFO]
Observation 2: Episode finished, reward = 1


{'steps': 2, 'answer': 'NOT ENOUGH INFO', 'gt_answer': 'NOT ENOUGH INFO', 'question_idx': 6238, 'em': 1, 'reward': 1, 'f1': 1} 

2 2 1.0 3.367874503135681
-----------

5388 Claim: Victoria Palace Theatre is barely in the West End.
Thought 1: I should search for the Victoria Palace Theatre and see if it is in the West End.
Action 1: Search[Victoria Palace Theatre]
Observation 1: Pages for logged out editors learn more. The Victoria Palace Theatre is a West End theatre in Victoria Street, in the City of Westminster, opposite Victoria Station. The structure is categorised as a Grade II* listed building.. The theatre began life as a small concert room above the stables of the Royal Standard Hotel, a small hotel and tavern built in 1832 at what was then 522 Stockbridge Terrace, on the site of the present theatre – not, as sometimes stated, on land where the train station now stands. The proprietor, John Moy, enlarged the building, and by 1850 it became known as Moy's Music Hall.

Thought 2: The observation says that the theatre is in the "West End", so the claim is true.
Action 2: Finish[SUPPORTS]
Observation 2: Episode finished, reward = 0


{'steps': 2, 'answer': 'SUPPORTS', 'gt_answer': 'REFUTES', 'question_idx': 5388, 'em': 0, 'reward': 0, 'f1': 0} 

2 3 0.6666666666666666 3.55368701616923
```

### *Output* 

```
Q：Aside from the Apple Remote, what other devicecan control the program Apple Remote was originallydesigned to interact with?
Thought 1: I need to search Apple Remote and find theprogram it was originally designed to interact with.
Act 1: Search[Apple Remote]
Obs 1: The Apple Remote is a remote control introduced inOctober 2005 by Apple .originally designed to control theFront Row media centerProgram ..
Thought 2: Apple Remote was originally designed to controlthe Front Row media center program, I need to searchFrontRow next and find what other device can control it.
Act 2: Search[Front RowlObs 2: Could not find [Front Row]. Similar:['FrontRowSeat to Earth','Front Row Motorsports'，Front Row(software)，
Thought 3: Front Row is not found. I need to searchRow (software）
Act 3:SearchlFront Row (software)
Obs 3: Front Row is a discontinued media center software 
Thought 4: Front Row (goftware) is controlled by an AppleRemote or the keyboard function keys .So the answer iskeyboard function keys.
Act 4: Finishlkeyboard function keys
```

## 数据集

### MultiWOZ 2.4
一个大规模的面向任务的会话数据集，包含多个域和复杂的用户目标。

### Persona-Cha
用于闲聊对话的数据集，为每个说话者分配一个个人档案，包括兴趣和爱好等个人信息。



