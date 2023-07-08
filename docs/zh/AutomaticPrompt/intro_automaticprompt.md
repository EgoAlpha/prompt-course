# 🗻 Automatic Prompt

**背景**

提示已经成为一种很有前途的方法，可以使用大型预训练语言模型（LM）来解决广泛的NLP问题，包括从左到右的模型，如GPT（[\[Radford等人 , 2019\]](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)）和Masked-LM，如BERT（[\[Devlin等人, 2019\]](https://doi.org/10.48550/arXiv.2210.01848)）、RoBERTa（[\[Liu等人, 2019\]](https://arxiv.org/pdf/1907.11692.pdf)）等。与为每个下游任务昂贵地更新大量LM参数的传统微调相比，提示将输入与引导LM产生所需输出的附加文本连接起来。提示的一个关键问题是如何找到最佳提示来提高LM在各种任务上的表现。然而，书写提示不仅耗时，而且不清楚相同的措辞是否对每种模式都有效，也不清楚是什么标准决定了特定的措辞是否最能引发所需信息。为了解决上述问题，研究人员提出了自动生成提示这一概念。

**概述**

Automatic Prompt 是使用先进的机器学习等技术为语言模型LM/LLM生成提示。它训练模型生成高质量的、上下文相关的准确提示，引导LM/LLM产生更连贯、更精确、更富有语义和上下文相关的结果。

其工作过程可以概括为：使用某种方法对prompt进行优化（这里的优化是相对于人工编写的prompt的过程），选择更好的prompt，再验证通过某种方法生成的prompt 的优劣（这里的优劣有多样评判标准，如，连贯性、可理解性、与上下文相关性等），最后更新prompt。



**prompt类型**

通常来说，prompt有两种形式：离散型prompt discrete prompt和连续型prompt soft prompt。

Automatic Prompt对不同的类型的prompt有不同的处理方法。

对于离散型的prompt，可以分为两种处理方法，第一种是人工设计的prompt（human-designed prompt）来引导LM/LLM生成所需要的内容，但使用该方法生成的prompt极有可能并非最佳；第二种是非人工设计prompt，即自动生成prompt，该方法通过离散的改变单个prompt token再验证改变后的prompt 
token带来的收益，确定该prompt是好是坏。

对于连续型的prompt，用连续的vectors来代替由token representation组成的token-level prompt，注意!!不微调模型，而是训练这些连续的vectors，以寻找一组最优的vectors来代替显性的离散的prompt。然而，就其性质而言连续型prompt由于其连续的形式而难以被人类理解。




**工作方法**

Automtic Prompt通过两种主要的方法搜索/调优prompt。



其一，通过搜索当前最优prompt进行替换，例如，AutoPrompt[\[Shin等人, 2020\]](https://arxiv.org/pdf/2010.15980.pdf)建议AutoPrompt执行梯度引导搜索，以在提示中找到最佳令牌；RLPrompt[\[（Deng 等人, 2022\]](https://doi.org/10.48550/arXiv.2210.01848)使用强化学习搜索这样的提示。这种方法一般应用于discrete prompt生成。




其二，优化当前prompt达到自动提示的目的，例如，[\[Zhong等人, 2021\]](https://www.aclweb.org/anthology/2021.naacl-main.398.pdf)提出了OptiPrompt，它直接优化输入嵌入空间中的提示，用于事实探究。该方法一般多应用于soft prompt 生成。

下面是这个板块内容的目录，您可以直接点击访问：

## 🌉 目录
- [Automatic Prompt Optimization with Gradient Descent and Beam Search](./optim/autooptim.md#automatic-prompt-optimization-with-gradient-descent-and-beam-search)
- [GPS Genetic Prompt Search for Efficient Few-shot Learning](./GPSPrompt/GPSPrompt.md#gps-genetic-prompt-search-for-efficient-few-shot-learning)
- [iPrompt: Explaining Data Patterns in Natural Language via Interpretable Autoprompting](./IPrompt/AutoiPrompt.md#iprompt-explaining-data-patterns-in-natural-language-via-interpretable-autoprompting)
- [PromptGen Automatically Generate Prompts using Generative Models](./PromptGen/PromptGen.md#promptgen-automatically-generate-prompts-using-generative-models)
- [RePrompt: Automatic Prompt Editing to Refine AI-Generative Art Towards Precise Expressions](./RePrompt/Reprompt.md#reprompt-automatic-prompt-editing-to-refine-ai-generative-art-towards-precise-expressions)








