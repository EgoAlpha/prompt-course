# 💡 Prompt Techniques

---

#### Basic prompting

Prompt-based approaches offer a powerful and flexible tool for resolving a wide range of NLP tasks.  We can complete the task by expressing ourselves in natural language, so there's no need to adhere to a specific format.

**Sentiment Analysis**: LLMs can be trained to identify the sentiment expressed in a piece of text, such as positive, negative, or neutral. This can be useful for analyzing customer feedback, social media posts, and product reviews.

👁️ **[Prompt example]**:

```Analyze the sentiment of the following text:```

```Text:  'I absolutely loved the movie! The acting was fantastic and the plot kept me engaged throughout the entire film.'```

**Entity Recognition**: LLMs can identify entities in a text, such as people, places, organizations, and products. This can be used for named entity recognition in various domains, such as news articles or legal documents.

👁️ **[Prompt example]**:

```Analyze the following paragraph and identify all the people, places, and organizations mentioned in the text, and then output the results in json format. ```

```Text: 'The Apple event took place at the Steve Jobs Theater in Cupertino, California. Tim Cook, the CEO of Apple, introduced the new iPhone 13, which will be available for pre-order starting next week. The phone comes in several colors and features a faster processor and longer battery life.'```

**Relation Extraction**: LLMs can also extract relationships between entities in a piece of text, such as identifying that a person is the CEO of a company or that a product is made by a certain brand. This can be useful for tasks such as knowledge graph construction, where the relationships between entities are important for understanding the domain.

👁️ **[Prompt example]**:

```Analyze the following sentence and extract the relationship between two entities, and then output the results in JSON format.```

```Text: 'Elon Musk founded SpaceX in 2002 with the goal of reducing space transportation costs and enabling the colonization of Mars.'```

**Text Summarization**: LLMs can also be used for text summarization, where they can automatically generate a summary of a longer piece of text, such as an article or report. This can be useful for quickly understanding the key points of a document without having to read the entire thing.

👁️ **[Prompt example]**:

```Summarize the following paragraph:  {Paste your paragraph}```

**Text Classification**: LLMs can classify text into predefined categories, such as classifying news articles into different topics or categorizing customer inquiries into different types. This can be useful for tasks such as content moderation, where incoming text needs to be classified quickly and accurately.

👁️ **[Prompt example]**:

```Classify the following keyword list in groups based on their search intent, whether commercial, transactional or informational: {Paste your keywords}```

**Text Clustering**: LLMs can group similar texts together based on their content or features. This feature can be useful in tasks such as data mining, topic modeling, and customer segmentation.

👁️ **[Prompt example]**:

```Cluster the following set of news articles into distinct groups based on their content:```

```"Tesla's electric car sales continue to soar despite the pandemic"```

```"New study suggests coffee consumption may lower risk of heart disease"```

```"The latest iPhone model features a larger screen and improved camera"```

```"COVID-19 cases surge in India, overwhelming healthcare system" ```

```"Amazon announces plans to build a new fulfillment center in Texas"```

```"Scientists discover new species of bird in the Amazon rainforest"```

```"Global climate change conference to be held in Paris next month"```

```"Starbucks introduces plant-based milk options in all U.S. stores"```

**Machine Translation**: LLMs can be used for machine translation, where they can translate text from one language to another. This can be useful for businesses that operate globally, as well as for individuals who need to communicate with people who speak different languages.

👁️ **[Prompt example]**

```Translate the following paragraph into Chinese. {Paste your paragraph}```


**Question Answering**: LLMs can be used for question answering, where they can read a passage of text and answer questions about it. This can be useful for applications such as customer support, where customers can ask questions and receive quick, accurate answers.

👁️ **[Prompt example]**:

```The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometers. It is located in the Coral Sea, off the coast of Australia. The reef is home to a diverse range of marine life, including over 1,500 species of fish, 600 types of coral, and numerous other species. What is the Great Barrier Reef and where is it located?```
