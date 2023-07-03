# GPS Genetic Prompt Search for Efficient Few-shot Learning

## Introduction

[Hanwei Xu et al., 2022](https://arxiv.org/pdf/2210.17041.pdf) propose a novel Genetic Prompt Search (GPS) algorithm that gradually mutates the prompts with a generative model and selects candidates according to their performance on a small development set. This evolutionary procedure relies on a tiny set of labeled data, only used for validation but not training.

GPS does not require updating any parameter, but only searches for the optimal hard prompts for every downstream task. Similar to prompt tuning, GPS allows the pretrained model to serve a large number of applications simultaneously. Meanwhile, GPS is even easier to deploy than prompt tuning, because it does not need to store the tuned continuous soft prompts. Empirically, GPS achieves substantial improvement over the baseline of manual prompts, and it also outperforms other parameter-efficient few-shot tuning methods.

The author followed the T0 baseline and compared five methods using ten T0 testing tasks, namely Model Tuning, Prompt Tuning, Black Box Tuning, In Context Learning, and GRIPS. Result showed that GPS outperforms not only the manual prompt baseline, but also other parameterefficient few-shot learning methods. Extensive experiments verified the effectiveness of the proposed GPS.


## How it Works?

It is challenging to automatically find highperforming prompts for a new unseen task.Inspired by Genetic Algorithms (Mitchell, 1980), we propose Genetic Prompt Search (GPS) for this purpose.The specific algorithm is shown in the following figure:

![Algorithm](img/Algorithm.png)

In GPS, we will first sample a tiny number of data as a development set Ddev for each downstream task.Then, we will design two genetic functions, where fGPS is the metric function to decide which prompts will be reserved or eliminated at each iteration, and gGPS represents the genetic function to generate new prompts.According to the algorithm, GPS is firstly initialized with a set of handcrafted prompts, G0. And the key process of GPS is to reproduce the current generation of prompts and use re-scoring to select prompts iteratively. For each iteration, we calculate the scores of prompts in Gt using fGPS, and select the top-K prompts as Gt∗ . Then we generate Gt+1 using gGPS based on Gt∗ . After several steps of genetic search, we will collect all the top-K prompts in each generation, and rescore all these prompts to make the final decision on which prompts are optimal.There are three different ways to use gGPS.

**Back Translation**:Back Translation (BT), a common technique for data augmentation in NLP, is applied for prompt reproduction. Here we first translate the manual prompts from English to 11 other languages including Chinese, Japanese, Korean, French, Spanish, Italian, Russian, German, Arabic, Greek, Cantonese, and then translate them back to English.

**Cloze**:we use the large pretrained text-to-text transformer (T5) to generate templates. For each input example and its verbalizer, we compose the template with placeholders as prefix and suffix, and let T5 to fill in the placeholders. We apply beam search to generate multiple prompt candidates.However, this approach does not work well since our setting conducts no parameter update, which is different from the few-shot training setting in the original paper. Therefore, we instead use manual prompts as initial templates, replace some random tokens with placeholders, and then let T5 fill in the blanks to generate new prompts.

**Sentence Continuation**:Use the template "Write two sentences that mean the same thing. Sentence 1: Manual Prompt, Sentence 2:" to the pretrained model, and let it generate continuations as a new prompt. We conducted experiments with GPT2-XL (1.5B) and T5LM-XXL (11B) as our prompt generation models.

Finally, according to the different ways in which prompts are generated, our corresponding scoring standards are also different. The specific scoring rules are as follows:

**For Cloze**:we follow previous work to score the prompts with average logits on the validation set Ddev. 

**For Back Translation and Sentence Continuation**:since averaging logits is not applicable, we score each prompt using accuracy on Ddev.



## Prompt Example

There are currently no prompt examples available

## Datasets

### Natural language inference
ANLI R1,ANLI R2, ANLI R3, CB, RTE.

### Coreference resolution
WSC, Winogrande.

### Sentence completion
COPA, HellaSwag.

### Word sense disambiguation
WiC.