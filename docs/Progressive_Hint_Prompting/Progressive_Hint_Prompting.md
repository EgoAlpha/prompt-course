# **Progressive-Hint Prompting Improves Reasoning in Large Language Models**

## Introduction

[Chuanyang Zheng et al., 2023](https://arxiv.org/abs/2304.09797) argues that similar to human cognition, LLM can re-evaluate and adapt its generated inference paths to correct errors and improve overall performance.

While Large Language Models (LLMs) have demonstrated remarkable performance across various NLP tasks, their ability to reason is often perceived as a limitation that cannot be overcome merely by increasing the scale of the model.The Chain-of-Thought(CoT) and self-consistency are the two methods to this limitation. CoT motivates LLM to perform stepwise reasoning by elaborating a series of cues. In contrast to CoT  that focuses on designing better prompts, self-consistency suggests taking multiple answers from the LLM and arriving at the correct answer by majority voting. 

But prior research has not explored the potential of leveraging the outputs of LLM to refine reasoning paths iteratively.So a new method named Progressive-Hint Prompting (PHP) was proposed. The PHP enables automatic multiple interactions between users and LLMs by using previously generated answers as hints to progressively guide toward the correct answers.



## How it Works?

A salient aspect of humans is that during reasoning we have the ability to not only think once, but also to double-check our answers after arriving at them. Therefore, we consider that this process can be simulated in the LLM by sequentially adopting previous answers.

This process can be summarized in the following three steps: (1) given a question, we ask the LLM for a basic answer; (2) combining the question and the answer, we re-interrogate the LLM and obtain subsequent answers; (3) we repeat the operations in (2) until the answers converge and do not change on the last two answers.

In addition, this procedure can be combined with existing methods (CoT, self-consistency, etc.) to further improve the performance of large models.

The main process is shown in the figure below:

![](picture\PHP.png)



## Prompt Example

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

## Datasets

### AddSub
The AddSub dataset is used to test arithmetic circuits and contains test cases for the addition and subtracter.

### MultiArith
MultiArith is a data set designed to test arithmetic reasoning, containing a variety of arithmetic operations such as addition, subtraction, multiplication, division, and so on.

### SingleEQ
SingleEQ is a dataset for testing single-equation algebraic problems, containing single-equation algebraic word problems that perform multiple mathematical operations on non-rational numbers and a variable.

### SVAMP
SVAMP is a data set for testing hardware validation that covers various hardware validation issues such as equivalence checking, model checking, and so on.

### GSM8K
GSM8K is a dataset used to test Chinese phoneme conversion, containing mapping relationships between various Chinese characters and pinyin.

### AQuA
AQuA is a dataset for testing assembly code understanding and contains mapping relationships between various assembly code and problems.







