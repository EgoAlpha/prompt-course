# 🎭 Practical Chapter

## Contents

This chapter is divided into two major sections, namely, ChatGPT Usage Guide and LangChain for LLM usage, in which ChatGPT Usage Guide is introduced in the following sections: Help Us study, Assist Us Work, Enrich Our Experience and Convenient to Our Life. The content of ChatGPT prompt is provided in the above directions, so that people from all fields can directly use the written prompts to search for queries. At the same time, a template example of prompt writing is given, so you only need to imitate the relevant content writing method and nest it into your own query content to complete the corresponding specific tasks. For the part of using LangChain to operate large language models, through a quick introduction to the operation of the open source machine learning library LangChain, you can be familiar with and operate large models such as OpenAI in a short time, thus making it more convenient for many developers.

### ChatGPT Usage Guide

The outline of the content included in this section is as follows: You can click on the relevant content directly for easy navigation and reference.

- [Help us study](chatgptprompt#help-us-study)
  - [Reading and writing](chatgptprompt.md#reading-and-writing)
  - [Learning programming](chatgptprompt.md#learning-programming)
- [Assist in our work](chatgptprompt.md#assist-in-our-work)
  - [Competition analysis](chatgptprompt.md#competition-analysis)
  - [Customer Service](chatgptprompt.md#customer-service)
  - [Aid in software development](chatgptprompt.md#aid-in-software-development)
  - [Aid in making videos](chatgptprompt.md#aid-in-making-videos)
  - [Start-up](chatgptprompt.md#Start-up)
  - [Educational work](chatgptprompt.md#educational-work)
- [Enrich our experience](chatgptprompt.md#enrich-our-experience)
  - [Debate Competition Simulation](chatgptprompt.md#debate-competition-simulation)
  - [Mock interview](chatgptprompt.md#mock-interview)
  - [Speech Design](chatgptprompt.md#speech-design)
- [Convenient to our lives](chatgptprompt.md#convenient-to-our-lives)
  - [Sports and fitness](chatgptprompt.md#sports-and-fitness)
  - [Music and Art](chatgptprompt.md#music-and-art)
  - [Travel Guide](chatgptprompt.md#travel-guide)
  - [Learning cooking](chatgptprompt.md#learning-cooking)

### LangChain for LLM Usage

We give here a tutorial on how to use concrete code to manipulate the Big Model, LangChain is a Big Model upper toolchain, an application development framework based on LLMs, to build applications using LLMs through composability. The focus is on "composability". LangChain can be used for chatbots, generative question and answer (GQA), text extraction, etc.
The goals of LangChain are to

- Allow big language models to process data from different sources
- Allow large language models to interact with the environment in which they are placed

LangChain library contains six main parts.

- [**Models**](langchainguide/guide.md#models): Provides large models encapsulated based on OpenAI API, including common OpenAI large models, and also supports custom large model encapsulation.
- [**Prompt**](langchainguide/guide.md#prompt): Support for fast implementation of custom Prompt projects and interfacing with LLMs.
- [**Index**](langchainguide/guide.md#index): accept user query, index the most relevant content to return.
- [**Memory**](langchainguide/guide.md#memory): standard interface, stores state between chains/calls.
- [**Chains**](langchainguide/guide.md#chains): a set of calls (LLMs or other, e.g. network, OS), Chains provides a standard interface and settings to combine these calls. The larger model executes a logical chain of sequential execution for a series of tasks.
- [**Agents**](langchainguide/guide.md#agents): Agents, a very important part, about what actions to do to LLMs and how to do them. Usually the capabilities in Utils and the various logic chains in Chains are encapsulated as Tools for Agents to call intelligently.
- [**Coding Examples**](langchainguide/guide.md#coding-examples): Combined with the above code examples, three classic cases are given, namely Document Query, Auto-Agent and Auto-GPT.

>Now, let's start the journey of super learners!
