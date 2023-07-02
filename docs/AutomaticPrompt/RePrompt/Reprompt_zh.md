# *Automatic Prompt*

## *iPrompt: Explaining Data Patterns in Natural Language via Interpretable Autoprompting*

[\[Wang et al., 2023\]](https://doi.org/10.1145/3544548.3581402)通过探索人工智能生成图像的情感表达能力，开发了RePrompt，RePrompt是一种基于XAI的Automatic Prompt Engineering 方法，可以细化文本提示（prompt），从而优化图像生成，实现精确的情感表达。

RePrompt中，作者基于外行编辑（用户可解释）策略策划了直观的文本特征，设计了图像质量度量标准，训练了机器学习模型来预测文本特征的图像质量分数，然后将模型解释应用于训练的模型，以生成用于自动编辑文本提示的量规。

模拟研究和用户研究结果表明，相对于IEA，RePrompt使用AI模型改进了图像生成，尤其是对于负面情绪。在我们的评估用户研究中，验证者无法感知不同条件下积极情绪表达的差异，这可能是由于人类对积极刺激的敏感性较低，以及CLIP对积极情绪建模的能力较弱。ITA的结果喜忧参半。

## How it Works?

**RePrompt工作过程**

1）通过文本到图像生成模型了解哪些提示特征可以导致更好的输出图像，以及2）自动修改文本提示以实现更好的输出图片。

`对于1）第一步`选择特征，设计单词级别的特征，这些特征满足了易于理解和调整的要求；

`第二步`设置图像质量度量标准，采用图像情感比对（IEA）和图像文本比对（ITA）的CLIP评分作为图像质量的衡量标准。

`第三步` 特征分析，了解策划特征如何影响图像生成；1）利用SHAP计算全局特征值的重要性，根据特征的重要性和调整值的容易程度选择突出特征；2）识别特征值范围，使用PDP
来分析特征在特征值分布上对模型输出的影响。通过识别最优特征值范围，我们最终策划了特征值调整的规则。

`对与2）第一步`给定一个文本，首先标记每个单词的词性（POS），并丢弃非名词、动词或形容词的单词，接着使用单词的CLIP分数和附加情感标签的全文来计算单词显著性，根据显著性确定删除和添加单词；

`第二步`从ConceptNet中检索前若干显著词的相关词，仅保留文本中的形容词；

`第三步`计算单词显著性并查找单词具体性，然后保留若干最显著的单词；

`第四步`添加情感标签，并最终确定了RePrompt的输出。

*Prompt:*

`原始输入:`"My best friend will be going to school in another country for 4 years".Emotion:"Sad"

然后找出句子中的名词、动词和形容词，计算它们的显著性，根据它们的显著性对它们进行排序，你就会得到下面的表格。

根据已有规则在表中添加或删除单词，本例中删除单词“years”，并从ConceptNet中检索前3个突出词(即“friend”、“going”和“school”)的相关单词(即“current”、“cold”、“advance”)。根据标题，我们只从检索到的单词中保留形容词。我们添加了情感标签，完成了提示修改的输出。

|      Word       | best  | friend  |  going  |  school  |  country  |  years  |
|:---------------:|:-----:|:-------:|:-------:|:--------:|:---------:|:-------:|
|       POS       |  ADJ  |  NOUN   |  VERB   |   NOUN   |   NOUN    |  NOUN   |
| Saliency Order  |   6   |    1    |    3    |    2     |     4     |    5    |

&darr;

|Relevant Words from ConceptNet|
|------------------------------|

&darr;

|elementary,boring,advance,cold,current,intimate...|
|--------------------------------------------------|

&darr;

|Add ADJs:current,cold,advance|
|-----------------------------|

&darr;

`最终输出prompt:`"best,friend,going,school,country,current,cold,advance,sad"
