# In-Context Learning(ICL)

## What Is In-context Learning(ICL)?

With the scaling of model size and corpus size, large language models (LLMs) demonstrate an in-context learning (ICL) ability, that is, learning from a few examples in the context. Many studies have shown that LLMs can perform a series of complex tasks through ICL, such as solving mathematical reasoning problems. These strong abilities have been widely verified as emerging abilities for large language models.

## What Are the Manifestations of ICL？

The key idea of in-context learning is to learn from analogy. First, ICL requires a few examples to form a demonstration context. These examples are usually written in natural language templates. Then, ICL concatenates a query question and a piece of demonstration context together to form a prompt, which is then fed into the language model for prediction. Different from supervised learning requiring a training stage that uses backward gradients to update model parameters, ICL does not conduct parameter updates and directly performs predictions on the pretrained language models. The model is expected to learn the pattern hidden in the demonstration and accordingly make the right prediction.

## What Are the Advantages of ICL?

ICL has multiple attractive advantages. First, since the demonstration is written in natural language, it provides an interpretable interface to communicate with LLMs. Second, in-context learning is similar to the decision process of human beings by learning from analogy. Third, compared with supervised training, ICL is a training-free learning framework.

## What Is Included in ICL Research?

The strong performance of ICL relies on two stages: 

(1) the training stage that cultivates the ICL ability of LLMs;

(2) the inference stage where LLMs predict according to task-specific demonstrations.

Although LLMs have shown promising ICL capability, many studies also show that the ICL capability can be further improved through a continual training stage between pretraining and ICL inference, which is called model warmup for short. Warmup is an optional procedure for ICL, which adjusts LLMs before ICL inference, including modifying the parameters of the LLMs or adding additional parameters.

Towards the inference stage, as the input and output labels are all represented in interpretable natural language templates, there are multiple directions for improving ICL performance. Many studies have shown that the performance of ICL strongly relies on the demonstration surface, including demonstration format, the order of demonstration examples, and so on. Demonstration design is often divided into two groups: demonstration organization and demonstration formatting.![](./Screenshot 2023-07-03 150505.png)
