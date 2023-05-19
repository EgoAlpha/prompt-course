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
                link: 'zh/principle',
                items: [
                  {
                  text: '设计原则', link: 'zh/principle#design-principle',
                }, 
                {
                  text: '框架', link: 'zh/principle#framework',
                },
                {
                  text: 'ChatGPT 使用指南', link: 'zh/principle#framework',items:[
                    {text: '帮助我们学习', link: 'zh/principle#framework',items:[
                      {text: '阅读和写作', link: 'zh/principle#framework'},
                      {text: '学习与编程', link: 'zh/principle#framework'}
                    ]},
                    {text: '协助我们工作', link: 'zh/principle#framework',items:[
                      {text: '竞争分析', link: 'zh/principle#framework'},
                      {text: '客户服务', link: 'zh/principle#framework'},
                      {text: '协助软件开发', link: 'zh/principle#framework'},
                      {text: '视频编辑', link: 'zh/principle#framework'},
                      {text: '初创企业', link: 'zh/principle#framework'},
                      {text: '教育工作', link: 'zh/principle#framework'}
                    ]},
                    {text: '丰富我们的经验', link: 'zh/principle#framework',items:[
                      {text: '辩论比赛模拟 ', link: 'zh/principle#framework'},
                      {text: '模拟面试', link: 'zh/principle#framework'},
                      {text: '演讲稿设计', link: 'zh/principle#framework'},
                    ]},
                    {text: '方便我们的生活', link: 'zh/principle#framework',items:[
                      {text: '运动与健身', link: 'zh/principle#framework'},
                      {text: '音乐与艺术', link: 'zh/principle#framework'},
                      {text: '旅游指南', link: 'zh/principle#framework'},
                      {text: '学习厨艺', link: 'zh/principle#framework'},
                    ]}
                  ] 
                },
                {
                  text: '使用LangChain操作大模型', link: 'zh/principle#framework',items:[
                    {text: '开始之前', link: 'zh/principle#framework'},
                    {text: '模型', link: 'zh/principle#framework'},
                    {text: '提示', link: 'zh/principle#framework'},
                    {text: '索引', link: 'zh/principle#framework'},
                    {text: '存储', link: 'zh/principle#framework'},
                    {text: '链', link: 'zh/principle#framework'},
                    {text: '代理', link: 'zh/principle#framework'},
                    {text: '代码样例', link: 'zh/principle#framework'},
                  ]
                },],
              },
              {
                text: '方法篇',
                link: 'zh/principle',
                items: [{
                  text: '基本Prompt', link: '/principle#design-principle',
                }, 
                {
                  text: '高级Prompt', link: '/principle#framework',items:[
                    {
                      text: '批量prompt', link: '/principle#framework',
                    },
                    {
                      text: '连续prompt', link: '/principle#framework',
                    },
                    {
                      text: 'PAL', link: 'zh/principle#framework',
                    },
                    {
                      text: 'ReAct', link: 'zh/principle#framework',
                    },
                    {
                      text: 'Self-Ask', link: 'zh/principle#framework',
                    },
                    {
                      text: 'Context-faithful Prompting', link: 'zh/principle#framework',
                    },
                    {
                      text: 'REFINER', link: 'zh/principle#framework',
                    },
                    {
                      text: 'Reflections', link: 'zh/principle#framework',
                    },
                    {
                      text: 'Progressive-Hint Prompt', link: 'zh/principle#framework',
                    },
                  ]
                },
                {
                  text: '自动化Prompt', link: 'zh/principle#framework',
                },
                {
                  text: '思维链', link: 'zh/principle#framework',
                },
                {
                  text: '上下文学习', link: 'zh/principle#framework',
                },
                {
                  text: '知识增强Prompt', link: 'zh/principle#framework',
                },
                {
                  text: '评估和可靠性', link: 'zh/principle#framework',
                }],
              },
              {
                text: '理论篇',
                link: 'zh/principle',
                items: [
                {
                  text: '大语言模型概览', link: 'zh/principle#design-principle',
                }, 
                {
                  text: 'Transformer', link: 'zh/principle#framework',
                },
                {
                  text: 'Tokenizer', link: 'zh/principle#framework',
                },
                {
                  text: 'BERT', link: 'zh/principle#framework',
                },
                {
                  text: 'GPT系列', link: 'zh/principle#framework',
                },
                {
                  text: 'T5', link: 'zh/principle#framework',
                },],
              },
              
              // {
              //   text: 'Design Principle and Framework',
              //   link: '/principle',
              //   items: [{
              //     text: 'Design Principle', link: '/principle#design-principle',
              //   }, {
              //     text: 'Framework', link: '/principle#framework',
              //   }],
              // },
              { text: 'Prompt Techniques', link: '/technique' },
            ],
          },
        ],
        //     items: [
        //       {
        //         text: '介绍', link: '/zh/intro',
        //       },
        //       {
        //         text: '设计原则和框架',
        //         link: '/zh/principle',
        //         items: [{
        //           text: '设计原则', link: '/zh/principle#设计原则',
        //         }, {
        //           text: '框架', link: '/zh/principle#框架',
        //         }],
        //       },

        //       { text: '提示技巧', link: '/zh/technique' },
        //     ],
        //   },
        // ],

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
                {
                  text: 'ChatGPT Usage Guide', link: '/chatgptprompt',items:[
                    {text: 'Help us study', link: '/chatgptprompt#help-us-study',items:[
                      {text: 'Reading and Writing', link: '/chatgptprompt#reading-and-writing'},
                      {text: 'Learning and Programming', link: '/chatgptprompt#learning-programming'}
                    ]},
                    {text: 'Assist in our work', link: '/chatgptprompt/assist-in-our-work',items:[
                      {text: 'Competition and Analysis', link: '/chatgptprompt#competition-and-analysis'},
                      {text: 'Customer and Service', link: '/chatgptprompt#customer-and-service'},
                      {text: 'Aid in Software Development', link: '/chatgptprompt#aid-in-software-development'},
                      {text: 'Aid in Making Videos', link: '/chatgptprompt#aid-in-making-videos'},
                      {text: 'Start-up', link: '/chatgptprompt#start-up'},
                      {text: 'Educational Work', link: '/chatgptprompt#educational-work'}
                    ]},
                    {text: 'Enrich our experience', link: '/chatgptprompt#enrich-our-experience',items:[
                      {text: 'Debate Competition Simulation ', link: '/chatgptprompt#debate-competition-simulation'},
                      {text: 'Mock Interview', link: '/chatgptprompt#mock-interview'},
                      {text: 'Speech Design', link: '/chatgptprompt#speech-design'},
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
                  text: 'LangChain for LLMs Usage', link: '/principle#framework',items:[
                    {text: 'Introduction', link: '/principle#framework'},
                    {text: 'Before Start', link: '/principle#framework'},
                    {text: 'Models', link: '/principle#framework'},
                    {text: 'Prompt', link: '/principle#framework'},
                    {text: 'Index', link: '/principle#framework'},
                    {text: 'Memory', link: '/principle#framework'},
                    {text: 'Chains', link: '/principle#framework'},
                    {text: 'Agents', link: '/principle#framework'},
                    {text: 'Coding Examples', link: '/principle#framework'},
                  ]
                },],
              },
              {
                text: 'Methodology Chapter',
                link: '/principle',
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
                      text: 'Batch Prompting', link: '/principle#framework',
                    },
                    {
                      text: 'Successive Prompting', link: '/principle#framework',
                    },
                    {
                      text: 'PAL', link: '/principle#framework',
                    },
                    {
                      text: 'ReAct', link: '/principle#framework',
                    },
                    {
                      text: 'Self-Ask', link: '/principle#framework',
                    },
                    {
                      text: 'Context-faithful Prompting', link: '/principle#framework',
                    },
                    {
                      text: 'REFINER', link: '/principle#framework',
                    },
                    {
                      text: 'Reflections', link: '/principle#framework',
                    },
                    {
                      text: 'Progressive-Hint Prompt', link: '/principle#framework',
                    },
                  ]
                },
                {
                  text: 'Automatic Prompt', link: '/principle#framework',
                },
                {
                  text: 'CoT', link: '/principle#framework',
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
                link: '/principle',
                items: [
                {
                  text: 'The Overview of LLM', link: '/principle#design-principle',
                }, 
                {
                  text: 'Transformer', link: '/principle#framework',
                },
                {
                  text: 'Tokenizer', link: '/principle#framework',
                },
                {
                  text: 'BERT', link: '/principle#framework',
                },
                {
                  text: 'GPT Series', link: '/principle#framework',
                },
                {
                  text: 'T5', link: '/principle#framework',
                },],
              },
              
              // {
              //   text: 'Design Principle and Framework',
              //   link: '/principle',
              //   items: [{
              //     text: 'Design Principle', link: '/principle#design-principle',
              //   }, {
              //     text: 'Framework', link: '/principle#framework',
              //   }],
              // },
              { text: 'Prompt Techniques', link: '/technique' },
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
      // user:1,
    },
  },
  vite: {
    server: {
      host: '0.0.0.0',
      hmr: true,
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
