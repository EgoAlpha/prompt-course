# RePrompt: Automatic Prompt Editing to Refine AI-Generative Art Towards Precise Expressions

[Wang et al., 2023](https://doi.org/10.1145/3544548.3581402) developed RePrompt, by exploring the emotional
expression ability of AI-generated images, RePrompt Is an Automatic Prompt Engineering method based on XAI that can refine text prompts (prompt) and So as to optimize the image generation and realize the accurate emotional expression.

In RePrompt, we curated intuitive text features based on layperson editing (user-interpretable) strategies, designed image quality metrics ,trained machine learning models to predict image quality scores with the text features, then applied model explanations to the trained model to generate a rubric for automatically editing the text prompts

The simulation study and user study results suggest that RePrompt improves image generation with the AI model with respect to IEA, especially for negative emotions. The validators in our evaluation user study could not perceive differences in the expression of positive emotions across conditions, which might be due to a lower sensitivity to positive stimuli in humans and CLIP’s weaker ability in modeling positive emotions. The results were mixed for ITA.

## How it Works?

RePrompt process: 1) understanding what prompt features could lead to better output images by text-to-image generative models, and 2) automatically revising text prompts to achieve better output images.

**For 1) Step 1**, select features and design word-level features that are easy to understand and adjust;

**Step 2**, the image quality measurement criteria were set, using the CLIP scores of image-emotion alignment (IEA) and
image-text alignment (ITA) as the image quality measurement criteria.

**Step 3**, feature analysis, understand how the planning features affect the image generation;

    1) Use SHAP to calculate the importance of global eigenvalues, and select prominent features according to the importance of features and the ease of adjusting values;

    2) Identify the range of eigenvalues, and use PDP to analyze the influence of features on model output in the distribution of eigenvalues. By identifying the optimal range of eigenvalues, we finally plan the rule of eigenvalue adjustment.
**For 2) step 1**, given a text, first mark the part of speech (POS) of each word, and discard the words that are not
nouns, verbs or adjectives, then use the CLIP score of the word and attach the full text of the emotion label to calculate the significance of the word, according to the significance to determine the deletion and addition of the word;

**step 2** is to retrieve the related words of the first several significant words from the ConceptNet, and only
retain the adjectives in the text.The third step is to calculate the significance of the word and find the specificity of the word, and then keep some of the most significant words;The fourth step is to add emotional tags and finally determine the output of the RePrompt.

*Prompt:*

`Original input:`"My best friend will be going to school in another country for 4 years".Emotion:"Sad"

Then find the nouns, verbs and adjectives in the sentence, calculate their Saliency, rank them according to their Saliency, and you will get the table below.

Add or delete words in the table according to existing rules,in this example,the word "years" will be deleted,and We retrieve relevant words of the top-3 salient words (i.e., “friend”, “going”, and “school”) from ConceptNet(i.e.,"current","cold","advance").We kept only adjectives from the retrieved words according to the rubric.We appended the emotion label and finalized the output of the prompt revision.

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

`Final prompt:`"best,friend,going,school,country,current,cold,advance,sad"
