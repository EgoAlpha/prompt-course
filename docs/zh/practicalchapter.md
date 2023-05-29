# 实践篇章

## 内容概览

本章节分为两大板块，分别是ChatGPT的使用指南和使用LangChain操作LLM。其中，ChatGPT使用指南会从以下几个部分进行介绍，分别是**帮助我们学习**，**协助我们工作**，**丰富我们的经验**和**方便我们的生活**四个部分，从上述几个方向提供了基于ChatGPT的提示内容，方便各行各业的人直接使用写好的提示内容进行搜索查询。同时给出了提示撰写的模版示例，您只需要效仿相关内容的写法，嵌套入自己的查询内容中即可完成对应的具体任务；使用LangChain操作大模型的部分，通过快速入门开源机器学习库LangChain的操作方法，可以短时间内熟悉和操作OpenAI等大模型，从而更加方便于诸多的开发者。

### ChatGPT 使用指南

该板块包含的内容提纲如下所示，您可以直接点击相关内容进行跳转，便于浏览和查阅。

- [帮助我们的学习](chatgptprompt#帮助我们学习)
  - [阅读与写作](chatgptprompt#阅读与写作)
  - [学习编程](chatgptprompt#学习编程)
- [协助我们的工作](chatgptprompt#协助我们的工作)
  - [竞争分析](chatgptprompt#竞争分析)
  - [客户服务](chatgptprompt#客户服务)
  - [协助软件开发](chatgptprompt#协助软件开发)
  - [视频编辑](chatgptprompt#视频编辑)
  - [初创企业](chatgptprompt#初创企业)
  - [教育工作](chatgptprompt#教育工作)
- [丰富我们的经验](chatgptprompt#丰富我们的经验)
  - [辩论比赛模拟](chatgptprompt#辩论比赛模拟)
  - [模拟面试](chatgptprompt#模拟面试)
  - [演讲稿设计](chatgptprompt#演讲稿设计)
- [方便我们的生活](chatgptprompt#方便我们的生活)
  - [运动健身](chatgptprompt#运动健身)
  - [音乐与艺术](chatgptprompt#音乐与艺术)
  - [旅游指南](chatgptprompt#旅游指南)
  - [学习厨艺](chatgptprompt#学习厨艺)

### 使用LangChain操作大模型

如何使用具体代码操作大模型，我们这里给出LangChain的教程说明，LangChain是一个大模型上层工具链，一个基于LLMs的应用程序开发框架, 通过可组合性来使用LLM构建应用程序. 其重点在于"可组合性"。设计一系列便于集成到实际应用中的接口，降低了在实际场景中部署大语言模型的难度。LangChain可用于聊天机器人、生成式问答(GQA)、文本摘要提取等。
LangChain的目标在于：

- 允许大语言模型处理不同来源的数据
- 让大语言模型能和布置它的环境之间进行交互

LangChain库主要包含六个部分:

- [**Models**](langchainguide/guide.md#models): 提供基于OpenAI API封装好的大模型，包含常见的OpenAI大模型，也支持自定义大模型的封装。
- [**Prompt**](langchainguide/guide.md#prompt): 支持自定义Prompt工程的快速实现以及和LLMs的对接。
- [**Index**](langchainguide/guide.md#index): 接受用户查询，索引最相关内容返回。
- [**Memory**](langchainguide/guide.md#memory): 标准的接口, 在chains/call之间保存状态。
- [**Chains**](langchainguide/guide.md#chains): 一系列的调用(LLMs或者其他, 如网络, 操作系统), Chains提供了标准的接口和设置来组合这些调用。 先从外部的源获取信息, 然后喂给LLMs。大模型针对一系列任务的顺序执行逻辑链。
- [**Agents**](langchainguide/guide.md#agents): 代理, 非常重要的一环, 关于对LLMs做何种action, 如何做。通常Utils中的能力、Chains中的各种逻辑链都会封装成一个个工具（Tools）供Agents进行智能化调用。
- [**Coding Examples**](langchainguide/guide.md#coding-examples): 结合上述内容的代码示例，给出三个很经典的案例，分别是文档查询，自动代理和Auto-GPT。

接下来，让我们开启超级学习者的旅程吧！
