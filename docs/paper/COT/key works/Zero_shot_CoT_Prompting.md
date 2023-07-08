# zero-shot CoT prompting

## *Large Language Models are Zero-Shot Reasoners*

[\[Kojima et al.,2022\]](https://arxiv.org/abs/2205.11916) shows that LLMs are decent zero-shot reasoners by simply adding “Let’s think step by step” before each answer in prompting. Experimental results demonstrate that this so called Zero-shot-CoT, using the same single prompt template,significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks including arithmetics and other logical reasoning tasks without any hand-crafted few-shot examples.

## How it Works?

The success of large language models (LLMs) is often attributed to (in-context) few-shot or zero-shot learning. It can solve various tasks by simply conditioning the models on a few examples (few-shot) or instructions describing the task (zero-shot).The method of conditioning the language model is called ”prompting”,and designing prompts either manually or automatically has become a hot topic in NLP. While the successes of CoT prompting along those of many other task-specific prompting work are often attributed to LLMs’ ability for few-shot learning,we show that LLMs are decent zero-shot reasoners by adding a simple prompt, “Let’s think step by step”, to facilitate step-by-step thinking before answering each question.Despite the simplicity, Zero-shot-CoT successfully generates a plausible reasoning path in a zero-shot manner and reaches the correct answer in a problem where the standard zero-shot approach fails.

Importantly, Zero-shot-CoT is versatile and task-agnostic, unlike most prior task-specific prompt engineering in the forms of examples (few-shot) or templates (zero-shot): it can facilitate step-by-step answers across various reasoning tasks, including arithmetic,symbolic reasoning,commonsense reasoning and other logical reasoning tasks without modifying the prompt per task. See the figure below.

<img src="../images/Zero_shot_CoT_prompting.png" width="100%">

Prompt示例：

```
*Prompt:*

*Q: A juggler can juggle 16 balls. Half of the balls are golf balls,and half of the golf balls are blue. How many blue golf balls are there?*

*A: Let's think step by step.*
```

*Output:*

```
*There are 16 balls in total. Half of the balls are golf balls. That means that there are 8 golf balls. Half of the golf balls are blue. That means that there are 4 blue golf balls.*
```
