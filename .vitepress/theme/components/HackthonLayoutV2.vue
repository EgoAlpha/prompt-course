<template>
  <div class="min-h-screen bg-white">
    <!-- Hero Section -->
    <section class="bg-white py-10">
      <div class="max-w-7xl mx-auto px-6 lg:px-8">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <!-- Left Column: Text Content -->
          <div class="text-left">
            <div class="flex items-center mb-8">
              <img src="/EgoAlpha.svg" alt="EgoAlpha Logo" class="h-9 w-auto mr-3">
              <span class="text-2xl font-semibold text-gray-800">EgoAlpha</span>
            </div>
            <h1 class="text-4xl md:text-5xl font-bold dark:text-gray-900 text-gray-900 mb-3 leading-tight">
              AI Agent & LangGraph Hackathon
            </h1>
            <p class="text-xl text-gray-600 mb-12">
              Hack the Future, One Agent at a Time.
            </p>
            <div class="flex space-x-12">
              <div>
                <p class="text-4xl font-bold text-blue-600">
                  150+
                </p>
                <p class="text-sm text-gray-500 tracking-wider mt-1">
                  HACKATHON PARTICIPANTS
                </p>
              </div>
              <div>
                <p class="text-4xl font-bold text-blue-600">
                  15
                </p>
                <p class="text-sm text-gray-500 tracking-wider mt-1">
                  SOLUTIONS BUILT
                </p>
              </div>
            </div>
          </div>
          <!-- Right Column: Image -->
          <div class="flex justify-center lg:justify-end">
            <img src="/hackthon-hero.png" alt="AI Agent & LangGraph Hackathon" class="rounded-lg shadow-2xl w-full max-w-lg h-auto object-cover">
          </div>
        </div>
      </div>
    </section>

    <!-- Navigation -->
    <nav class="sticky top-0 bg-white/95 backdrop-blur-sm border-b border-gray-200 z-40">
      <div class="max-w-4xl mx-auto px-6 lg:px-8">
        <div class="flex items-center justify-between h-16">
          <div class="flex items-center space-x-8">
            <a href="/" class="text-gray-600 hover:dark:text-gray-900 text-gray-900 transition-colors cursor-pointer">← Back to Home</a>
          </div>
          <div class="flex items-center space-x-6">
            <a href="#title" class="text-gray-600 hover:dark:text-gray-900 text-gray-900 transition-colors cursor-pointer">Project</a>
            <a href="#background" class="text-gray-600 hover:dark:text-gray-900 text-gray-900 transition-colors cursor-pointer">Background</a>
            <a href="#overview" class="text-gray-600 hover:dark:text-gray-900 text-gray-900 transition-colors cursor-pointer">Overview</a>
            <a href="#technical" class="text-gray-600 hover:dark:text-gray-900 text-gray-900 transition-colors cursor-pointer">Technical</a>
            <a href="#application" class="text-gray-600 hover:dark:text-gray-900 text-gray-900 transition-colors cursor-pointer">Application</a>
          </div>
        </div>
      </div>
    </nav>

    <!-- Winner Introduction Section -->
    <section class="bg-gray-50 py-16">
      <div class="max-w-4xl mx-auto px-6 lg:px-8 text-center">
        <h2 class="text-3xl font-bold dark:text-gray-900 text-gray-900 mb-4">
          🏆 Announcing the Winning Project
        </h2>
        <p class="text-lg text-gray-600 leading-relaxed max-w-3xl mx-auto">
          After an intense competition featuring numerous innovative solutions, we are thrilled to present the winning project. The following sections provide a deep dive into the project's background, its groundbreaking technical architecture, and its significant application value.
        </p>
      </div>
    </section>

    <!-- Main Content -->
    <div class="max-w-4xl mx-auto px-6 lg:px-8 py-16">
      <!-- Project Title Section -->
      <section id="title" class="mb-16 text-center">
        <h1 class="text-4xl lg:text-4xl font-bold dark:text-gray-900 text-gray-900 mb-8 leading-tight">
          Enterprise Market Value Forecasting System Based on Augmented Reflexion-Agent
        </h1>
        <div class="flex justify-center mb-8">
          <img src="/hackthon-flow.png" alt="System Architecture Flow" class="max-w-full h-auto rounded-lg shadow-lg">
        </div>
      </section>

      <!-- Background Section -->
      <section id="background" class="mb-16">
        <h2 class="text-3xl font-bold dark:text-gray-900 text-gray-900 mb-6">
          Background
        </h2>
        <div class="prose prose-lg max-w-none">
          <p class="text-gray-700 leading-relaxed mb-6">
            With the rapid growth of the global economy and the increasing complexity of financial markets, enterprise market value forecasting has become crucial for capital allocation, risk management, and strategic decision-making.
          </p>
          <p class="text-gray-700 leading-relaxed mb-4">
            However, traditional forecasting approaches often:
          </p>
          <ul class="list-disc pl-6 mb-6 space-y-2">
            <li class="text-gray-700">
              Over-rely on historical financial data and statistical models;
            </li>
            <li class="text-gray-700">
              Neglect unstructured factors such as macroeconomic changes, supply chain dynamics, policy shifts, and news sentiment;
            </li>
            <li class="text-gray-700">
              Lack dynamic adaptability, leading to limited accuracy and unstable performance.
            </li>
          </ul>
          <div class="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-r-lg">
            <p class="text-blue-900 font-medium">
              Thus, there is an urgent need for a new forecasting paradigm that can integrate heterogeneous data sources while providing self-reflection and adaptive optimization capabilities.
            </p>
          </div>
        </div>
      </section>

      <!-- Project Overview Section -->
      <section id="overview" class="mb-16">
        <h2 class="text-3xl font-bold dark:text-gray-900 text-gray-900 mb-6">
          Project Overview
        </h2>
        <div class="prose prose-lg max-w-none">
          <p class="text-gray-700 leading-relaxed mb-6">
            We propose an Enterprise Market Value Forecasting System based on an Augmented Reflexion-Agent and a Retrieval-Augmented Generation (RAG) framework.
          </p>
          <p class="text-gray-700 leading-relaxed mb-4">
            The system integrates:
          </p>
          <ul class="list-disc pl-6 mb-6 space-y-2">
            <li class="text-gray-700">
              <strong>Structured data:</strong> financial statements, market trading data, supply chain knowledge graphs;
            </li>
            <li class="text-gray-700">
              <strong>Unstructured data:</strong> news sentiment, financial reports (text), and macroeconomic analysis.
            </li>
          </ul>
          <p class="text-gray-700 leading-relaxed mb-6">
            By leveraging multimodal data fusion, the system captures key drivers of market value fluctuations, enabling forecasts that are more comprehensive, adaptive, and accurate.
          </p>
          <p class="text-gray-700 leading-relaxed mb-4">
            At the core lies the Reflexion mechanism, which allows the agent to:
          </p>
          <ul class="list-disc pl-6 mb-6 space-y-2">
            <li class="text-gray-700">
              Perform self-feedback and strategy adjustment during continuous forecasting;
            </li>
            <li class="text-gray-700">
              Evaluate historical predictions and refine reasoning strategies;
            </li>
            <li class="text-gray-700">
              Establish a closed-loop optimization process, enhancing long-term forecasting stability.
            </li>
          </ul>
        </div>
      </section>

      <!-- Technical Highlights Section -->
      <section id="technical" class="mb-16">
        <h2 class="text-3xl font-bold dark:text-gray-900 text-gray-900 mb-6">
          Technical Highlights
        </h2>
        <div class="prose prose-lg max-w-none">
          <div class="mb-8">
            <h3 class="text-xl font-semibold dark:text-gray-900 text-gray-900 mb-4">
              1. Automated Data Extraction & Knowledge Construction
            </h3>
            <ul class="list-disc pl-6 space-y-2">
              <li class="text-gray-700">
                Table parsing and structured data extraction;
              </li>
              <li class="text-gray-700">
                Chunking + multimodal vectorization to build high-quality knowledge bases.
              </li>
            </ul>
          </div>

          <div class="mb-8">
            <h3 class="text-xl font-semibold dark:text-gray-900 text-gray-900 mb-4">
              2. Enhanced RAG (Retrieval-Augmented Generation)
            </h3>
            <ul class="list-disc pl-6 space-y-2">
              <li class="text-gray-700">
                Hybrid retrieval combining sparse and dense vectors;
              </li>
              <li class="text-gray-700">
                Semantic re-ranking to ensure contextual relevance;
              </li>
              <li class="text-gray-700">
                Effectively mitigates "hallucination" in generative models.
              </li>
            </ul>
          </div>

          <div class="mb-8">
            <h3 class="text-xl font-semibold dark:text-gray-900 text-gray-900 mb-4">
              3. Reflexion-Agent
            </h3>
            <ul class="list-disc pl-6 space-y-2">
              <li class="text-gray-700">
                Periodic self-evaluation mechanism;
              </li>
              <li class="text-gray-700">
                Continuous optimization of forecasting strategies;
              </li>
              <li class="text-gray-700">
                Builds a self-evolving predictive loop.
              </li>
            </ul>
          </div>
        </div>
      </section>

      <!-- Application Value Section -->
      <section id="application" class="mb-16">
        <h2 class="text-3xl font-bold dark:text-gray-900 text-gray-900 mb-6">
          Application Value
        </h2>
        <div class="prose prose-lg max-w-none">
          <ol class="list-decimal pl-6 space-y-3">
            <li class="text-gray-700">
              <strong>Investment Institutions:</strong> Optimize asset allocation and improve returns.
            </li>
            <li class="text-gray-700">
              <strong>Enterprise Management:</strong> Support strategic planning and risk control.
            </li>
            <li class="text-gray-700">
              <strong>Government & Regulators:</strong> Assist in macroeconomic regulation and policy-making.
            </li>
          </ol>
        </div>
      </section>
    </div>

    <!-- Footer -->
    <!-- <footer class="bg-gray-900 text-white py-16 px-6 lg:px-8">
      <div class="max-w-4xl mx-auto text-center">
        <h3 class="text-2xl font-bold mb-4">EgoAlpha Prompt Course</h3>
        <p class="text-gray-400 mb-6 leading-relaxed">
          Exploring the frontiers of artificial intelligence through innovative prompt engineering and practical applications in AI agent development.
        </p>
        <div class="flex justify-center space-x-4">
          <a href="#" class="w-10 h-10 bg-gray-800 rounded-full flex items-center justify-center hover:bg-gray-700 transition-colors cursor-pointer">
            <i class="i-ri-github-fill text-blue-400"></i>
          </a>
          <a href="#" class="w-10 h-10 bg-gray-800 rounded-full flex items-center justify-center hover:bg-gray-700 transition-colors cursor-pointer">
            <i class="i-ri-book-line text-blue-400"></i>
          </a>
        </div>
      </div>
    </footer> -->
  </div>
</template>

<style scoped>
.prose {
  /* color: inherit; */
}
.prose h1, .prose h2, .prose h3, .prose h4 {
  /* color: inherit; */
}
.prose strong {
  /* color: inherit; */
  font-weight: 600;
}
img{
    width: unset;
    padding-left: unset;
}
</style>
