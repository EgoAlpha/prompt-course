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

// https://vitepress.dev/reference/site-config
export default defineConfig({
  titleTemplate: 'Prompt Engineering Course, Resource and News',
  head: [
    ['meta', { name:"keywords",content:"Prompt Engineering, in-context learning, Prompt learning, chain-of-thought, LLMs, Artificial General Intelligence, AGI" }],
    ['meta', { name:"description",content:"The mecca for studying Prompt engineering and In-Context Learning." }],
    ['meta', { name:"subject",content:"Prompt Engineering, In-context Learning,prompt-based-learning,Large Language Model, ChatGPT" }],
    ['meta', { name:"robots",content:"index, follow" }],
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
          { text: 'TrustGPT', link: 'https://trustgpt.co' },
        ],

        sidebar: [
          {
            text: '课程',
            items: [
              {
                text: '介绍', link: 'zh/intro',
              },
              {
                text: '实践篇章',
                link: 'zh/practicalchapter',
                items: [
                {
                  text: 'ChatGPT 使用指南', link: 'zh/chatgptprompt#🌋-chatgpt-提示使用指南',items:[
                    {text: '帮助我们学习', link: 'zh/chatgptprompt#帮助我们学习',items:[
                      {text: '阅读和写作', link: 'zh/chatgptprompt#阅读与写作'},
                      {text: '学习与编程', link: 'zh/chatgptprompt#学习编程'}
                    ]},
                    {text: '协助我们工作', link: 'zh/chatgptprompt#协助我们的工作',items:[
                      {text: '竞争分析', link: 'zh/chatgptprompt#竞争分析'},
                      {text: '客户服务', link: 'zh/chatgptprompt#客户服务'},
                      {text: '协助软件开发', link: 'zh/chatgptprompt#协助软件开发'},
                      {text: '视频编辑', link: 'zh/chatgptprompt#视频编辑'},
                      {text: '初创企业', link: 'zh/chatgptprompt#初创企业'},
                      {text: '教育工作', link: 'zh/chatgptprompt#教育工作'}
                    ]},
                    {text: '丰富我们的经验', link: 'zh/chatgptprompt#丰富我们的经验',items:[
                      {text: '辩论比赛模拟 ', link: 'zh/chatgptprompt#辩论比赛模拟'},
                      {text: '模拟面试', link: 'zh/chatgptprompt#模拟面试'},
                      {text: '演讲稿设计', link: 'zh/chatgptprompt#演讲稿设计'},
                    ]},
                    {text: '方便我们的生活', link: 'zh/chatgptprompt#方便我们的生活',items:[
                      {text: '运动与健身', link: 'zh/chatgptprompt#运动与健身'},
                      {text: '音乐与艺术', link: 'zh/chatgptprompt#音乐与艺术'},
                      {text: '旅游指南', link: 'zh/chatgptprompt#旅游指南'},
                      {text: '学习厨艺', link: 'zh/chatgptprompt#学习厨艺'},
                    ]}
                  ] 
                },
                {
                  text: '使用LangChain操作大模型', link: 'zh/langchainguide/guide#🎇-langchain',items:[
                    {text: '开始之前', link: 'zh/langchainguide/guide#before-start'},
                    {text: '模型', link: 'zh/langchainguide/guide#models'},
                    {text: '提示', link: 'zh/langchainguide/guide#prompt'},
                    {text: '索引', link: 'zh/langchainguide/guide#index'},
                    {text: '存储', link: 'zh/langchainguide/guide#memory'},
                    {text: '链', link: 'zh/langchainguide/guide#chains'},
                    {text: '代理', link: 'zh/langchainguide/guide#agents'},
                    {text: '代码样例', link: 'zh/langchainguide/guide#coding-examples'},
                  ]
                },],
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
                  text: '基本Prompt', link: 'zh/principle#design-principle',
                }, 
                {
                  text: '高级Prompt', link: 'zh/principle#框架',items:[
                    {
                      text: '批量prompt', link: '/Batch_Prompting/BatchPrompting_zh',
                    },
                    {
                      text: '连续prompt', link: '/SuccessivePrompt/Suc_Prompting_Dec_Com_Que_zh',
                    },
                    {
                      text: 'PAL', link: '/PAL/PALPrompting_zh',
                    },
                    {
                      text: 'ReAct', link: '/ReAct/ReActPrompting_zh',
                    },
                    {
                      text: 'Self-Ask', link: '/Self_Ask/MEA_NARROWING_zh',
                    },
                    {
                      text: 'Context-faithful Prompting', link: '/Context_faithful_Prompting/Context_faithful_zh',
                    },
                    {
                      text: 'REFINER', link: 'REFINER/REFINER_zh',
                    },
                    {
                      text: 'Reflections', link: '/Reflexion/Reflexion_zh',
                    },
                    {
                      text: 'Progressive-Hint Prompt', link: 'Progressive_Hint_Prompting/Progressive_Hint_Prompting_zh',
                    },
                    {
                      text: 'Self-Refine', link: 'Self_Refine/Self_Refine_zh',
                    },
                  ]
                },
                {
                  text: '自动化Prompt', link: 'zh/principle#框架',
                },
                {
                  text: '思维链', link: 'zh/METHOD/cotintro',items:[
                    {text: 'Auto-COT Prompting', link: 'zh/paper/COT/key works/Auto-CoT_Prompting'},
                    {text: 'One-Few Shot CoT Prompting', link: 'zh/paper/COT/key works/One_Few_Shot_CoT_Prompting'},
                    {text: 'Self-consistency', link: 'zh/paper/COT/key works/Self-consistency'},
                    {text: 'Zero-shot CoT Prompting', link: 'zh/paper/COT/key works/Zero_shot_CoT_Prompting'},
                  ]
                },
                {
                  text: '上下文学习', link: 'zh/principle#框架',
                },
                {
                  text: '知识增强Prompt', link: 'zh/principle#框架',
                },
                {
                  text: '评估和可靠性', link: 'zh/principle#框架',
                }],
              },
              {
                text: '理论篇',
                link: 'zh/theorychapter',
                items: [
                {
                  text: '大语言模型概览', link: 'zh/principle#框架',
                }, 
                {
                  text: 'Transformer', link: 'zh/Transformer_md/Transformer',
                },
                {
                  text: 'Tokenizer', link: 'zh/token',
                },
                {
                  text: 'BERT', link: 'zh/principle#框架',
                },
                {
                  text: 'GPT系列', link: 'zh/gpt2_fintuning',items:[
                    {text: 'GPT-2', link: '/gpt2_fintuning',},
                    {text: 'RecurrentGPT', link: '/RecurrentGPT/RecurrentGPT',}
                  ]
                },
                {
                  text: 'T5', link: 'zh/principle#框架',
                },],
              },
              // { text: 'Prompt Techniques', link: '/technique' },
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
          { text: 'Course', link: '/intro' },
          { text: 'TrustGPT', link: 'https://trustgpt.co', target: '_blank' },
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
                  {text: 'ChatGPT Usage Guide', link: '/chatgptprompt#🌋-chatgpt-usage-guide',items:[
                    {text: 'Help us study', link: '/chatgptprompt#help-us-study',items:[
                      {text: 'Reading and Writing', link: '/chatgptprompt#reading-and-writing'},
                      {text: 'Learning and Programming', link: '/chatgptprompt#learning-programming'}
                    ]},
                    {text: 'Assist in our work', link: '/chatgptprompt#assist-in-our-work',items:[
                      {text: 'Competition and Analysis', link: '/chatgptprompt#competition-analysis'},
                      {text: 'Customer and Service', link: '/chatgptprompt#customer-service'},
                      {text: 'Aid in Software Development', link: '/chatgptprompt#aid-in-software-development'},
                      {text: 'Aid in Making Videos', link: '/chatgptprompt#aid-in-making-videos'},
                      {text: 'Start-up', link: '/chatgptprompt#start-up'},
                      {text: 'Educational Work', link: '/chatgptprompt#educational-work'}
                    ]},
                    {text: 'Enrich our experience', link: '/chatgptprompt#enrich-our-experience',items:[
                      {text: 'Debate Competition Simulation ', link: '/chatgptprompt#debate-competition-simulation'},
                      {text: 'Mock Interview', link: '/chatgptprompt#mock-interview'},
                      {text: 'Speech Design', link: '/chatgptprompt#speech-design'}
                    ]},
                    {text: 'Convenient to our lives', link: '/chatgptprompt#convenient-to-our-lives',items:[
                      {text: 'Sports and Fitness', link: '/chatgptprompt#sports-and-fitness'},
                      {text: 'Music and Art', link: '/chatgptprompt#music-and-art'},
                      {text: 'Travel Guide', link: '/chatgptprompt#travel-guide'},
                      {text: 'Learning Cooking', link: '/chatgptprompt#learning-cooking'},
                    ]}
                  ] 
                },
                {
                  text: 'LangChain for LLMs Usage', link: '/langchainguide/guide#🎇-langchain',items:[{text: 'Before Start', link: '/langchainguide/guide#before-start'},
                  {text: 'Models', link: '/langchainguide/guide#models'},
                  {text: 'Prompt', link: '/langchainguide/guide#prompt'},
                  {text: 'Index', link: '/langchainguide/guide#index'},
                  {text: 'Memory', link: '/langchainguide/guide#memory'},
                  {text: 'Chains', link: '/langchainguide/guide#chains'},
                  {text: 'Agents', link: '/langchainguide/guide#agents'},
                  {text: 'Coding Examples', link: '/langchainguide/guide#coding-examples'}]
                }],
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
                  text: 'Basic Prompt', link: '/principle#design-principle',
                }, 
                {
                  text: 'Advanced Prompt', link: '/principle#framework',items:[
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
                  ]
                },
                {
                  text: 'Automatic Prompt', link: '/principle#framework',
                },
                {
                  text: 'CoT', link: '/METHOD/cotintro',items:[
                    {text: 'Auto-COT Prompting', link: 'paper/COT/key works/Auto_CoT_Prompting'},
                    {text: 'One-Few Shot CoT Prompting', link: 'paper/COT/key works/One_Few_Shot_CoT_Prompting'},
                    {text: 'Self-consistency', link: 'paper/COT/key works/Self-consistency'},
                    {text: 'Zero-shot CoT Prompting', link: 'paper/COT/key works/Zero_shot_CoT_Prompting'},
                  ]
                },
                {
                  text: 'In-Context Learning', link: '/principle#framework',
                },
                {
                  text: 'Knowledge Augumented Prompt', link: '/principle#framework',
                },
                {
                  text: 'Evaluation and Reliability', link: '/principle#framework',
                }],
              },
              {
                text: 'Theory Chapter',
                link: '/theorychapter',
                items: [
                {
                  text: 'The Overview of LLM', link: '/principle#design-principle',
                }, 
                {
                  text: 'Transformer', link: '/Transformer_md/Transformer',
                },
                {
                  text: 'Tokenizer', link: '/token',
                },
                {
                  text: 'BERT', link: '/principle#framework',
                },
                {
                  text: 'GPT Series', link: '/gpt2_fintuning',items:[
                    {text: 'GPT-2', link: '/gpt2_fintuning',},
                    {text: 'RecurrentGPT', link: '/RecurrentGPT/RecurrentGPT',}
                  ]
                },
                {
                  text: 'T5', link: '/principle#framework',
                },],
              },
              // { text: 'Prompt Techniques', link: '/technique' },
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
})
