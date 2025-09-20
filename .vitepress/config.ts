import { defineConfig } from 'vitepress'
// import path from 'node:path'
// import { defineConfig } from 'vite'
// import Vue from '@vitejs/plugin-vue'
import Components from 'unplugin-vue-components/vite'
import AutoImport from 'unplugin-auto-import/vite'
import Unocss from 'unocss/vite'
// import VueMacros from 'unplugin-vue-macros/vite'
import { presetAttributify, presetIcons, presetUno, presetWebFonts } from 'unocss'
import extractorPug from '@unocss/extractor-pug'
import mathjax3 from 'markdown-it-mathjax3'

const customElements = [
  'math',
  'maction',
  'maligngroup',
  'malignmark',
  'menclose',
  'merror',
  'mfenced',
  'mfrac',
  'mi',
  'mlongdiv',
  'mmultiscripts',
  'mn',
  'mo',
  'mover',
  'mpadded',
  'mphantom',
  'mroot',
  'mrow',
  'ms',
  'mscarries',
  'mscarry',
  'mscarries',
  'msgroup',
  'mstack',
  'mlongdiv',
  'msline',
  'mstack',
  'mspace',
  'msqrt',
  'msrow',
  'mstack',
  'mstack',
  'mstyle',
  'msub',
  'msup',
  'msubsup',
  'mtable',
  'mtd',
  'mtext',
  'mtr',
  'munder',
  'munderover',
  'semantics',
  'math',
  'mi',
  'mn',
  'mo',
  'ms',
  'mspace',
  'mtext',
  'menclose',
  'merror',
  'mfenced',
  'mfrac',
  'mpadded',
  'mphantom',
  'mroot',
  'mrow',
  'msqrt',
  'mstyle',
  'mmultiscripts',
  'mover',
  'mprescripts',
  'msub',
  'msubsup',
  'msup',
  'munder',
  'munderover',
  'none',
  'maligngroup',
  'malignmark',
  'mtable',
  'mtd',
  'mtr',
  'mlongdiv',
  'mscarries',
  'mscarry',
  'msgroup',
  'msline',
  'msrow',
  'mstack',
  'maction',
  'semantics',
  'annotation',
  'annotation-xml',
  'mjx-container',
  'mjx-assistive-mml',
]
// https://vitepress.dev/reference/site-config
export default defineConfig({
  titleTemplate: 'Prompt Engineering Course, Resource and News',
  head: [
    ['meta', { name: 'keywords', content: 'Prompt Engineering, in-context learning, Prompt learning, chain-of-thought, LLMs, Artificial General Intelligence, AGI' }],
    ['meta', { name: 'description', content: 'The mecca for studying Prompt engineering and In-Context Learning.' }],
    ['meta', { name: 'subject', content: 'Prompt Engineering, In-context Learning,prompt-based-learning,Large Language Model, ChatGPT' }],
    ['meta', { name: 'robots', content: 'index, follow' }],
  ],
  srcDir: './docs',
  locales: {
    zh: {
      label: '中文',
      lang: 'zh',
      title: 'EgoAlpha',
      // description: 'A Prompt 教程',
      themeConfig: {
        // https://vitepress.dev/reference/default-theme-config
        nav: [
          { text: '首页', link: '/zh/' },
          { text: '课程', link: '/zh/intro' },
          { text: 'Agents', link: '/zh/agentsintro' },
          { text: 'Symposium', link: '/symposium' },
          { text: 'Hackthon', link: '/hackthon' },
        ],

        sidebar: [
          {
            text: '课程',
            items: [
              {
                text: '介绍', link: '/zh/intro',
              },
              {
                text: '实践篇章',
                link: 'zh/practicalchapter',
                items: [
                  {
                    text: 'ChatGPT 使用指南',
                    link: 'zh/chatgptprompt#🌋-chatgpt-提示使用指南',
                    items: [
                      {
                        text: '帮助我们学习',
                        link: 'zh/chatgptprompt#帮助我们学习',
                        items: [
                          {
                            text: '阅读和写作', link: 'zh/chatgptprompt#阅读与写作',
                          },
                          {
                            text: '学习与编程', link: 'zh/chatgptprompt#学习编程',
                          },
                        ],
                      },
                      {
                        text: '协助我们工作',
                        link: 'zh/chatgptprompt#协助我们的工作',
                        items: [
                          {
                            text: '竞争分析', link: 'zh/chatgptprompt#竞争分析',
                          },
                          {
                            text: '客户服务', link: 'zh/chatgptprompt#客户服务',
                          },
                          {
                            text: '协助软件开发', link: 'zh/chatgptprompt#协助软件开发',
                          },
                          {
                            text: '视频编辑', link: 'zh/chatgptprompt#视频编辑',
                          },
                          {
                            text: '初创企业', link: 'zh/chatgptprompt#初创企业',
                          },
                          {
                            text: '教育工作', link: 'zh/chatgptprompt#教育工作',
                          },
                        ],
                      },
                      {
                        text: '丰富我们的经验',
                        link: 'zh/chatgptprompt#丰富我们的经验',
                        items: [
                          {
                            text: '辩论比赛模拟 ', link: 'zh/chatgptprompt#辩论比赛模拟',
                          },
                          {
                            text: '模拟面试', link: 'zh/chatgptprompt#模拟面试',
                          },
                          {
                            text: '演讲稿设计', link: 'zh/chatgptprompt#演讲稿设计',
                          },
                        ],
                      },
                      {
                        text: '方便我们的生活',
                        link: 'zh/chatgptprompt#方便我们的生活',
                        items: [
                          {
                            text: '运动与健身', link: 'zh/chatgptprompt#运动与健身',
                          },
                          {
                            text: '音乐与艺术', link: 'zh/chatgptprompt#音乐与艺术',
                          },
                          {
                            text: '旅游指南', link: 'zh/chatgptprompt#旅游指南',
                          },
                          {
                            text: '学习厨艺', link: 'zh/chatgptprompt#学习厨艺',
                          },
                        ],
                      },
                    ],
                  },
                  {
                    text: '使用LangChain操作大模型',
                    link: 'zh/langchainguide/guide#🎇-langchain',
                    items: [
                      {
                        text: '开始之前', link: 'zh/langchainguide/guide#before-start',
                      },
                      {
                        text: '模型', link: 'zh/langchainguide/guide#models',
                      },
                      {
                        text: '提示', link: 'zh/langchainguide/guide#prompt',
                      },
                      {
                        text: '索引', link: 'zh/langchainguide/guide#index',
                      },
                      {
                        text: '存储', link: 'zh/langchainguide/guide#memory',
                      },
                      {
                        text: '链', link: 'zh/langchainguide/guide#chains',
                      },
                      {
                        text: '代理', link: 'zh/langchainguide/guide#agents',
                      },
                      {
                        text: '代码样例', link: 'zh/langchainguide/guide#coding-examples',
                      },
                    ],
                  },
                ],
              },
              {
                text: '方法篇章',
                link: 'zh/methodchapter',
                items: [
                  {
                    text: '设计原则', link: 'zh/principle#设计原则',
                  },
                  {
                    text: '框架', link: 'zh/principle#框架',
                  },
                  {
                    text: '基本提示设计', link: 'zh/basicprompting_zh',
                  },
                  {
                    text: '高级提示设计',
                    link: 'zh/advanced',
                    items: [
                      {
                        text: '批量提示设计', link: 'zh/Batch_Prompting/BatchPrompting',
                      },
                      {
                        text: '连续提示设计', link: 'zh/SuccessivePrompt/Suc_Prompting_Dec_Com_Que',
                      },
                      {
                        text: 'PAL', link: 'zh/PAL/PALPrompting',
                      },
                      {
                        text: 'ReAct', link: 'zh/ReAct/ReActPrompting',
                      },
                      {
                        text: 'Self-Ask', link: 'zh/Self_Ask/MEA_NARROWING',
                      },
                      {
                        text: '忠实于上下文的提示', link: 'zh/Context_faithful_Prompting/Context_faithful',
                      },
                      {
                        text: 'REFINER', link: 'zh/REFINER/REFINER',
                      },
                      {
                        text: '反思', link: 'zh/Reflexion/Reflexion',
                      },
                      {
                        text: 'Progressive-Hint提示设计', link: 'zh/Progressive_Hint_Prompting/Progressive_Hint_Prompting',
                      },
                      {
                        text: '自我提问', link: 'zh/Self_Refine/Self_Refine',
                      },
                      {
                        text: '循环GPT', link: 'zh/RecurrentGPT/RecurrentGPT',
                      },
                    ],
                  },
                  {
                    text: '自动化提示设计',
                    link: 'zh/AutomaticPrompt/intro_automaticprompt',
                    items: [
                      {
                        text: '自动提示优化使用梯度下降和束搜索', link: 'zh/AutomaticPrompt/optim/autooptim',
                      },
                      {
                        text: '基于遗传提示搜索用于高效的少样本学习', link: 'zh/AutomaticPrompt/GPSPrompt/GPSPrompt',
                      },
                      {
                        text: 'iPrompt 用自然语言解释数据模式', link: 'zh/AutomaticPrompt/IPrompt/AutoiPrompt',
                      },
                      {
                        text: 'PromptGen 使用生成模型自动生成提示', link: 'zh/AutomaticPrompt/PromptGen/PromptGen',
                      },
                      {
                        text: 'RePrompt 自动提示编辑以改进AI生成', link: 'zh/AutomaticPrompt/RePrompt/Reprompt',
                      },
                    ],
                  },
                  {
                    text: '思维链',
                    link: 'zh/METHOD/cotintro',
                    items: [
                      {
                        text: '自动思维链提示', link: 'zh/paper/COT/key works/Auto_CoT_Prompting',
                      },
                      {
                        text: '少样本思维链提示', link: 'zh/paper/COT/key works/One_Few_Shot_CoT_Prompting',
                      },
                      {
                        text: '自一致性', link: 'zh/paper/COT/key works/Self-consistency',
                      },
                      {
                        text: '零样本思维链提示', link: 'zh/paper/COT/key works/Zero_shot_CoT_Prompting',
                      },
                    ],
                  },
                  {
                    text: '上下文学习',
                    link: 'zh/paper/ICL/introduction',
                    items: [
                      {
                        text: 'Transformers learn in-context by gradient descent', link: 'zh/paper/ICL/with/theoryanalysis/gradient_descent/gradient_descent',
                      },
                      {
                        text: 'Language Models Secretly Perform Gradient Descent as Meta-Optimizers', link: 'zh/paper/ICL/with/theoryanalysis/MetaOptimizers/MetaOptimizers',
                      },
                      {
                        text: 'Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale', link: 'zh/paper/ICL/with/theoryanalysis/Rethinking/Rethinking',
                      },
                      {
                        text: 'Complementary Explanations for Effective In-Context Learning', link: 'zh/paper/ICL/with/design/ComExp/ComExp',
                      },
                      {
                        text: 'PLACES: Prompting Language Models for Social Conversation Synthesis', link: 'zh/paper/ICL/with/design/ConversationSynthesis/ConversationSynthesis',
                      },
                      {
                        text: 'In-context Learning Distillation: Transferring Few-shot Learning Ability of Pre-trained Language Models', link: 'zh/paper/ICL/with/design/Distillation/Distillation',
                      },
                      {
                        text: 'Diverse Demonstrations Improve In-context Compositional Generalization', link: 'zh/paper/ICL/with/design/DiverseDemo/DiverseDemo',
                      },
                    ],
                  },
                  {
                    text: '知识增强提示设计', link: 'zh/KAP/kap',
                  },
                  {
                    text: '评估和可靠性', link: 'zh/EvaRelia/evalua',items:[
                      {
                        text: '用于评估大模型的以人为本的基准', link: 'zh/EvaRelia/evalua#以人为中心基准的评估模型',
                      },
                      {
                        text: '大语言模型是否已经足够先进？', link: 'zh/EvaRelia/evalua#大语言模型已经足够先进了吗',
                      },
                      {
                        text: '大型语言模型的事件语义的综合评价', link: 'zh/EvaRelia/evalua#大型语言模型事件语义的综合评价',
                      },
                      {
                        text: 'GPT-4是否是一位很好的数据分析师？', link: 'zh/EvaRelia/evalua#gpt-4是一个好的数据分析师吗',
                      },
                    ]
                  },
                ],
              },
              {
                text: '理论篇',
                link: 'zh/theorychapter',
                items: [
                  {
                    text: '大语言模型概览', link: 'zh/LLMoverview/LLM',
                  },
                  {
                    text: 'Transformer', link: 'zh/Transformer_md/Transformer',
                  },
                  {
                    text: 'Tokenizer', link: 'zh/token',
                  },
                  {
                    text: 'BERT', link: 'zh/BERT/BERT',
                  },
                  {
                    text: 'GPT系列', link: 'zh/gpt2_finetuning',
                  },
                  {
                    text: 'T5', link: 'zh/T5/T5contents',
                  },
                ],
              },
            ],
          },
        ],

        socialLinks: [
          { icon: 'github', link: 'https://github.com/EgoAlpha/prompt-in-context-learning' },
        ],
      },
    },
    root: {
      label: 'English',
      lang: 'en',
      title: 'EgoAlpha',
      description: 'A Prompt Course',
      themeConfig: {
        // https://vitepress.dev/reference/default-theme-config
        nav: [
          { text: 'Home', link: '/' },
          { text: 'Symposium', link: '/symposium' },
          { text: 'Hackthon', link: '/hackthon' },
          { text: 'Course', link: '/intro' },
          { text: 'Agents', link: '/agentsintro' },
        ],

        sidebar: [
          {
            text: 'Course',
            items: [
              {
                text: 'Introduction', link: '/intro',
              },
              {
                text: 'Practical Chapter',
                link: '/practicalchapter',
                items: [
                  {
                    text: 'ChatGPT Usage Guide',
                    link: '/chatgptprompt#🌋-chatgpt-usage-guide',
                    items: [
                      {
                        text: 'Help us study',
                        link: '/chatgptprompt#help-us-study',
                        items: [
                          {
                            text: 'Reading and Writing', link: '/chatgptprompt#reading-and-writing',
                          },
                          {
                            text: 'Learning and Programming', link: '/chatgptprompt#learning-programming',
                          },
                        ],
                      },
                      {
                        text: 'Assist in our work',
                        link: '/chatgptprompt#assist-in-our-work',
                        items: [
                          {
                            text: 'Competition and Analysis', link: '/chatgptprompt#competition-analysis',
                          },
                          {
                            text: 'Customer and Service', link: '/chatgptprompt#customer-service',
                          },
                          {
                            text: 'Aid in Software Development', link: '/chatgptprompt#aid-in-software-development',
                          },
                          {
                            text: 'Aid in Making Videos', link: '/chatgptprompt#aid-in-making-videos',
                          },
                          {
                            text: 'Start-up', link: '/chatgptprompt#start-up',
                          },
                          {
                            text: 'Educational Work', link: '/chatgptprompt#educational-work',
                          },
                        ],
                      },
                      {
                        text: 'Enrich our experience',
                        link: '/chatgptprompt#enrich-our-experience',
                        items: [
                          {
                            text: 'Debate Competition Simulation ', link: '/chatgptprompt#debate-competition-simulation',
                          },
                          {
                            text: 'Mock Interview', link: '/chatgptprompt#mock-interview',
                          },
                          {
                            text: 'Speech Design', link: '/chatgptprompt#speech-design',
                          },
                        ],
                      },
                      {
                        text: 'Convenient to our lives',
                        link: '/chatgptprompt#convenient-to-our-lives',
                        items: [
                          {
                            text: 'Sports and Fitness', link: '/chatgptprompt#sports-and-fitness',
                          },
                          {
                            text: 'Music and Art', link: '/chatgptprompt#music-and-art',
                          },
                          {
                            text: 'Travel Guide', link: '/chatgptprompt#travel-guide',
                          },
                          {
                            text: 'Learning Cooking', link: '/chatgptprompt#learning-cooking',
                          },
                        ],
                      },
                    ],
                  },
                  {
                    text: 'LangChain for LLMs Usage',
                    link: '/langchainguide/guide#🎇-langchain',
                    items: [
                      {
                        text: 'Before Start', link: '/langchainguide/guide#before-start',
                      },
                      {
                        text: 'Models', link: '/langchainguide/guide#models',
                      },
                      {
                        text: 'Prompt', link: '/langchainguide/guide#prompt',
                      },
                      {
                        text: 'Index', link: '/langchainguide/guide#index',
                      },
                      {
                        text: 'Memory', link: '/langchainguide/guide#memory',
                      },
                      {
                        text: 'Chains', link: '/langchainguide/guide#chains',
                      },
                      {
                        text: 'Agents', link: '/langchainguide/guide#agents',
                      },
                      {
                        text: 'Coding Examples', link: '/langchainguide/guide#coding-examples',
                      },
                    ],
                  },
                ],
              },
              {
                text: 'Methodology Chapter',
                link: '/methodchapter',
                items: [
                  {
                    text: 'Design Principle', link: '/principle#design-principle',
                  },
                  {
                    text: 'Framework', link: '/principle#framework',
                  },
                  {
                    text: 'Basic Prompt', link: '/basicprompting',
                  },
                  {
                    text: 'Advanced Prompt',
                    link: '/advanced',
                    items: [
                      {
                        text: 'Batch Prompting', link: '/Batch_Prompting/BatchPrompting',
                      },
                      {
                        text: 'Successive Prompting', link: '/SuccessivePrompt/Suc_Prompting_Dec_Com_Que',
                      },
                      {
                        text: 'PAL', link: '/PAL/PALPrompting',
                      },
                      {
                        text: 'ReAct', link: '/ReAct/ReActPrompting',
                      },
                      {
                        text: 'Self-Ask', link: '/Self_Ask/MEA_NARROWING',
                      },
                      {
                        text: 'Context-faithful Prompting', link: '/Context_faithful_Prompting/Context_faithful',
                      },
                      {
                        text: 'REFINER', link: 'REFINER/REFINER',
                      },
                      {
                        text: 'Reflections', link: '/Reflexion/Reflexion',
                      },
                      {
                        text: 'Progressive-Hint Prompt', link: 'Progressive_Hint_Prompting/Progressive_Hint_Prompting',
                      },
                      {
                        text: 'Self-Refine', link: 'Self_Refine/Self_Refine',
                      },
                      {
                        text: 'RecurrentGPT', link: 'RecurrentGPT/RecurrentGPT',
                      },
                    ],
                  },
                  {
                    text: 'Automatic Prompt',
                    link: 'AutomaticPrompt/intro_automaticprompt',
                    items: [
                      {
                        text: 'Optimization with Gradient Descent and Beam Search', link: 'AutomaticPrompt/optim/autooptim',
                      },
                      {
                        text: 'Genetic Prompt Search for Efficient Few-shot Learning', link: 'AutomaticPrompt/GPSPrompt/GPSPrompt',
                      },
                      {
                        text: 'Explaining Data Patterns in Natural Language', link: 'AutomaticPrompt/IPrompt/AutoiPrompt',
                      },
                      {
                        text: 'Automatically Generate Prompts using Generative Models', link: 'AutomaticPrompt/PromptGen/PromptGen',
                      },
                      {
                        text: 'Automatic Prompt Editing to Refine AI-Generative Art', link: 'AutomaticPrompt/RePrompt/Reprompt',
                      },
                    ],
                  },
                  {
                    text: 'CoT',
                    link: '/METHOD/cotintro',
                    items: [
                      {
                        text: 'Auto-COT Prompting', link: 'paper/COT/key works/Auto_CoT_Prompting',
                      },
                      {
                        text: 'One-Few Shot CoT Prompting', link: 'paper/COT/key works/One_Few_Shot_CoT_Prompting',
                      },
                      {
                        text: 'Self-consistency', link: 'paper/COT/key works/Self-consistency',
                      },
                      {
                        text: 'Zero-shot CoT Prompting', link: 'paper/COT/key works/Zero_shot_CoT_Prompting',
                      },
                    ],
                  },
                  {
                    text: 'In-Context Learning',
                    link: 'paper/ICL/introduction',
                    items: [
                      {
                        text: 'Transformers learn in-context by gradient descent', link: 'paper/ICL/with/theoryanalysis/gradient_descent/gradient_descent',
                      },
                      {
                        text: 'Language Models Secretly Perform Gradient Descent as Meta-Optimizers', link: 'paper/ICL/with/theoryanalysis/MetaOptimizers/MetaOptimizers',
                      },
                      {
                        text: 'Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale', link: 'paper/ICL/with/theoryanalysis/Rethinking/Rethinking',
                      },
                      {
                        text: 'Complementary Explanations for Effective In-Context Learning', link: 'paper/ICL/with/design/ComExp/ComExp',
                      },
                      {
                        text: 'PLACES: Prompting Language Models for Social Conversation Synthesis', link: 'paper/ICL/with/design/ConversationSynthesis/ConversationSynthesis',
                      },
                      {
                        text: 'In-context Learning Distillation: Transferring Few-shot Learning Ability of Pre-trained Language Models', link: 'paper/ICL/with/design/Distillation/Distillation',
                      },
                      {
                        text: 'Diverse Demonstrations Improve In-context Compositional Generalization', link: 'paper/ICL/with/design/DiverseDemo/DiverseDemo',
                      },
                    ],
                  },
                  {
                    text: 'Knowledge Augumented Prompt', link: '/KAP/kap',
                  },
                  {
                    text: 'Evaluation and Reliability',
                    link: 'EvaRelia/evalua',
                    items: [
                      {
                        text: 'AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models', link: 'EvaRelia/evalua#agieval-a-human-centric-benchmark-for-evaluating-foundation-models',
                      },
                      {
                        text: 'Have LLMs Advanced Enough?A Challenging Problem Solving Benchmark For Large Language Models', link: 'EvaRelia/evalua#have-llms-advanced-enough-a-challenging-problem-solving-benchmark-for-large-language-models',
                      },
                      {
                        text: 'EVEVAL:A Comprehensive Evaluation of Event Semantics for Large Language Model', link: 'EvaRelia/evalua#eveval-a-comprehensive-evaluation-of-event-semantics-for-large-language-model',
                      },
                      {
                        text: 'Is GPT-4 a Good Data Analyst?', link: 'EvaRelia/evalua#is-gpt-4-a-good-data-analyst',
                      },
                    ],
                  },
                ],
              },
              {
                text: 'Theory Chapter',
                link: '/theorychapter',
                items: [
                  {
                    text: 'The Overview of LLM', link: 'LLMoverview/LLM',
                  },
                  {
                    text: 'Transformer', link: '/Transformer_md/Transformer',
                  },
                  {
                    text: 'Tokenizer', link: '/token',
                  },
                  {
                    text: 'BERT', link: '/BERT/BERT',
                  },
                  {
                    text: 'GPT Series', link: '/gpt2_finetuning',
                  },
                  {
                    text: 'T5', link: '/T5/T5contents',
                  },
                ],
              },
            ],
          },
        ],

        socialLinks: [
          { icon: 'github', link: 'https://github.com/EgoAlpha/prompt-in-context-learning' },
        ],
      },
    },
  },
  themeConfig: {
    logo: '/EgoAlpha.svg',
    pinia: {
    },
  },
  vite: {
    server: {
      host: '0.0.0.0',
      hmr: false,
      proxy: {
        '/api': {
          target: 'https://www.motionvision.cn',
          changeOrigin: true,
        },
        // '/amis': {
        //   target: 'https://aisuda.bce.baidu.com',
        //   changeOrigin: true,
        // },
      },
    },
    plugins: [
      // VueMacros({
      //   plugins: {
      //     // vue: Vue({
      //     //   // reactivityTransform: true,
      //     // }),
      //   },
      // }),

      // https://github.com/antfu/unplugin-auto-import
      AutoImport({
        imports: [
          'vue',
          'vue/macros',
          'vue-router',
          '@vueuse/core',
        ],
        dts: '../.vitepress/theme/auto-imports.d.ts',
        dirs: [
          '../.vitepress/theme/composables',
        ],
        vueTemplate: true,
      }),

      // https://github.com/antfu/vite-plugin-components
      Components({
        dts: '../.vitepress/theme/components.d.ts',
        dirs: [
          '../.vitepress/theme/components',
        ],
      }),

      // https://github.com/antfu/unocss
      // see unocss.config.ts for config
      Unocss({
        shortcuts: [
          // ['btn', 'px-4 py-1 rounded inline-block bg-teal-600 text-white cursor-pointer hover:bg-teal-700 disabled:cursor-default disabled:bg-gray-600 disabled:opacity-50'],
          // ['icon-btn', 'text-[0.9em] inline-block cursor-pointer select-none opacity-75 transition duration-200 ease-in-out hover:opacity-100 hover:text-teal-600 !outline-none'],
        ],
        presets: [
          presetUno(),
          presetAttributify(),
          presetIcons({
            scale: 1.2,
            warn: true,
            collections: {
              ri: () => import('@iconify-json/ri/icons.json').then(i => i.default),
            },
          }),
          presetWebFonts({
            fonts: {
              sans: 'DM Sans',
              serif: 'DM Serif Display',
              mono: 'DM Mono',
            },
          }),
        ],
        extractors: [
          extractorPug(),
        ],
        // transformers: [
        //   transformerDirectives(),
        //   transformerVariantGroup(),
        // ],
      }),
    ],
  },
  markdown: {
    config: (md) => {
      md.use(mathjax3)
    },
  },
  vue: {
    template: {
      compilerOptions: {
        isCustomElement: tag => customElements.includes(tag),
      },
    },
  },
})
