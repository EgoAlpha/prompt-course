<!-- # Practical Chapter

## Contents

This chapter is divided into two major sections, namely, ChatGPT Usage Guide and LangChain for LLM usage, in which ChatGPT Usage Guide is introduced in the following sections: Help Us study, Assist Us Work, Enrich Our Experience and Convenient to Our Life. The content of ChatGPT prompt is provided in the above directions, so that people from all fields can directly use the written prompts to search for queries. At the same time, a template example of prompt writing is given, so you only need to imitate the relevant content writing method and nest it into your own query content to complete the corresponding specific tasks. For the part of using LangChain to operate large language models, through a quick introduction to the operation of the open source machine learning library LangChain, you can be familiar with and operate large models such as OpenAI in a short time, thus making it more convenient for many developers.

### ChatGPT Usage Guide

The outline of the content included in this section is as follows: You can click on the relevant content directly for easy navigation and reference.

- [Help us study](docs/chatgptprompt.md#help-us-study)
  - [Reading and writing](docs/chatgptprompt.md#reading-and-writing)
  - [Learning programming](docs/chatgptprompt.md#learning-programming)
- [Assist in our work](docs/chatgptprompt.md#assist-in-our-work)
  - [Competition analysis](docs/chatgptprompt.md#competition-analysis)
  - [Customer Service](docs/chatgptprompt.md#customer-service)
  - [Aid in software development](docs/chatgptprompt.md#aid-in-software-development)
  - [Aid in making videos](docs/chatgptprompt.md#aid-in-making-videos)
  - [Start-up](docs/chatgptprompt.md#Start-up)
  - [Educational work](docs/chatgptprompt.md#educational-work)
- [Enrich our experience](docs/chatgptprompt.md#enrich-our-experience)
  - [Debate Competition Simulation](docs/chatgptprompt.md#debate-competition-simulation)
  - [Mock interview](docs/chatgptprompt.md#mock-interview)
  - [Speech Design](docs/chatgptprompt.md#speech-design)
- [Convenient to our lives](docs/chatgptprompt.md#convenient-to-our-lives)
  - [Sports and fitness](docs/chatgptprompt.md#sports-and-fitness)
  - [Music and Art](docs/chatgptprompt.md#music-and-art)
  - [Travel Guide](docs/chatgptprompt.md#travel-guide)
  - [Learning cooking](docs/chatgptprompt.md#learning-cooking)

### LangChain for LLM Usage

We give here a tutorial on how to use concrete code to manipulate the Big Model, LangChain is a Big Model upper toolchain, an application development framework based on LLMs, to build applications using LLMs through composability. The focus is on "composability". LangChain can be used for chatbots, generative question and answer (GQA), text extraction, etc.
The goals of LangChain are to

- Allow big language models to process data from different sources
- Allow large language models to interact with the environment in which they are placed

LangChain library contains six main parts.

- [**Models**](docs/langchainguide/guide.md#models): Provides large models encapsulated based on OpenAI API, including common OpenAI large models, and also supports custom large model encapsulation.
- [**Prompt**](docs/langchainguide/guide.md#prompt): Support for fast implementation of custom Prompt projects and interfacing with LLMs.
- [**Index**](docs/langchainguide/guide.md#index): accept user query, index the most relevant content to return.
- [**Memory**](docs/langchainguide/guide.md#memory): standard interface, stores state between chains/calls.
- [**Chains**](docs/langchainguide/guide.md#chains): a set of calls (LLMs or other, e.g. network, OS), Chains provides a standard interface and settings to combine these calls. The larger model executes a logical chain of sequential execution for a series of tasks.
- [**Agents**](docs/langchainguide/guide.md#agents): Agents, a very important part, about what actions to do to LLMs and how to do them. Usually the capabilities in Utils and the various logic chains in Chains are encapsulated as Tools for Agents to call intelligently.
- [**Coding Examples**](docs/langchainguide/guide.md#coding-examples): Combined with the above code examples, three classic cases are given, namely Document Query, Auto-Agent and Auto-GPT.

Now, let's start the journey of super learners! -->
