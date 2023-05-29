# Chain-Of-Thought(COT)

## **What Is Chain of Thought Prompting？**

Consider one’s own thought process when solving a complicated reasoning task such as a multi-step math word problem. It is typical to decompose the problem into intermediate steps and solve each before giving the final answer.Based on this idea, [\[Wei et al., 2022a\] ](https://arxiv.org/abs/2201.11903)try to prompt the large language models to generate a chain of thought , rather than just giving the answer directly,when solving a reasoning-heavy problem.

Specificly,a chain of thought(CoT) is a series of intermediate natural language

reasoning steps that lead to the final output, and this approach is named as chain-of-thought prompting(CoT Prompting).

## **What Are the Manifestations of CoT Prompting？**

In order to prompt the LLMs to complete the generation of reasoning chains before giving the final answer，some methods have already been explored and proven to be effective. The pioneering work is few-shot CoT prompting([Wei et al., 2022a]),in which we need to provide examples with chain of thoughts ahead of the final answer(triples: < input, chain of thought, output>). 

Considering the high labor cost of manually constructing examples, some people are beginning to explore using non-manual methods to build examples. Among them, the most excellent methods are Auto-CoT([\[Zhang et al., 2022\]](https://arxiv.org/abs/2210.03493) ), Automate-CoT([\[Shum, et al., 2023\]](https://arxiv.org/abs/2302.12822)) and so on.

Another approach is to optimize prompts, without providing examples, and encourage the model to generate a chain of thought only through prompts, representative work is zero-shot CoT prompting([\[Kojima et al.,2022\]](https://arxiv.org/abs/2205.11916)).

There are also some other methods that utilize other algorithms, such as bootstrapping, reinforcement learning, etc., and have made many improvements on the basis of the initial CoT prompting method.

The ideas and specific details of the above-mentioned methods will be described in detail in subsequent chapters.

# **How to Design Innovative CoT Prompting？**

If you are interested in CoT prompting and willing to research and propose your own innovative methods, you can start from the following aspects:

1. Prompt itself design & prompting workflow & both
2. Demonstrations selection method
3. Post process of prompting results
4. Combining with algorithms and ideas in other fields

Firstly, you can research and modify the prompt itself to achieve better results based on existing paradigms, or propose new paradigms. In addition, the quality of example selection in prompt greatly affects the quality of the final inference result. You may need to research new methods for selecting examples, such as increasing the diversity of examples, in order to improve inference performance

Finally,the post-processing of the prompting results is also worth considering after the interaction with LLMs. For example, multiple reasonging paths(CoT with answer) can be sampled and the answer with the most occurrences can be selected as the final answer. Alternatively, it is possible to consider using algorithms from other fields in this study, which may have unexpected effects.