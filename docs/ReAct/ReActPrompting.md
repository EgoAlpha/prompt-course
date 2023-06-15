# **ReAct Prompting**

## Introduction

[Yao et al., 2022(opens in a new tab)](https://arxiv.org/abs/2210.03629) introduced a framework named ReAct where LLMs are used to generate both **reasoning traces** and **task-specific actions** in an interleaved manner.

Generating reasoning traces allow the model to induce, track, and update action plans, and even handle exceptions. The action step allows to interface with and gather information from external sources such as knowledge bases or environments.

The ReAct framework can allow LLMs to interact with external tools to retrieve additional information that leads to more reliable and factual responses.

The React framework is a general paradigm that combines reasoning and action with language models to solve various language reasoning and decision tasks.

Results show that ReAct can outperform several state-of-the-art baselines on language and decision-making tasks. ReAct also leads to improved human interpretability and trustworthiness of LLMs. Overall, the authors found that best approach uses ReAct combined with chain-of-thought (CoT) that allows use of both internal knowledge and external information obtained during reasoning.

Experimental results demonstrate that the combination of COT and React is significantly superior to either approach alone.


## How it Works?

ReAct prompts LLMs to generate both verbal reasoning traces and actions pertaining to a task in an interleaved manner, which allows the model to perform dynamic reasoning to create, maintain, and adjust high-level plans for acting (reason to act), while also interact with the external environments (e.g. Wikipedia) to incorporate additional information into reasoning (act to reason).

 

Chain-of-thought (CoT) prompting has shown the capabilities of LLMs to carry out reasoning traces to generate answers to questions involving arithmetic and commonsense reasoning, among other tasks [(Wei et al., 2022)(opens in a new tab)](https://arxiv.org/abs/2201.11903). But it's lack of access to the external world or inability to update its knowledge can lead to issues like fact hallucination and error propagation.

![react.png](img/react.png)

ReAct is a general paradigm that combines reasoning and acting with LLMs. ReAct prompts LLMs to generate verbal reasoning traces and actions for a task. This allows the system to perform dynamic reasoning to create, maintain, and adjust plans for acting while also enabling interaction to external environments (e.g., Wikipedia) to incorporate additional information into the reasoning. The figure below shows an example of ReAct and the different steps involved to perform question answering.





## Prompt Example

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

## Datasets

### MultiWOZ 2.4
A large-scale task-oriented conversation dataset containing multiple domains and complex user objectives.

### Persona-Cha
A dataset used for small talk conversations, where each speaker is assigned a personal profile, including personal information such as interests and hobbies.




