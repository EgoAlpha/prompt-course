<template>
  <div class="min-h-screen bg-white">
    <!-- Hero Section -->
    <section class="relative min-h-screen bg-cover bg-center bg-no-repeat flex items-center" 
             style="background-image: linear-gradient(rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.6)), url('/hackthon-hero.png');">
      <div class="w-full px-6 lg:px-8">
        <div class="max-w-4xl">
          <div class="text-white space-y-8">
            <div class="space-y-4">
              <div class="flex items-center space-x-4 mb-6">
                <div class="bg-yellow-500 text-black px-4 py-2 rounded-full font-bold text-lg">🏆 Winner</div>
              </div>
              <h1 class="text-5xl lg:text-6xl font-bold leading-tight">
                AI Agent & LangGraph
                <span class="block text-blue-400">Hackathon</span>
              </h1>
              <h2 class="text-2xl lg:text-3xl text-purple-400 font-semibold">
                Enterprise Market Value Forecasting System Based on Augmented Reflexion-Agent
              </h2>
            </div>
            <div class="flex flex-col sm:flex-row gap-4">
              <button class="bg-yellow-500 hover:bg-yellow-600 text-black px-8 py-4 rounded-lg text-lg font-semibold transition-colors cursor-pointer whitespace-nowrap">
                View Demo
              </button>
              <button class="border-2 border-white text-white hover:bg-white hover:text-gray-900 px-8 py-4 rounded-lg text-lg font-semibold transition-colors cursor-pointer whitespace-nowrap">
                Technical Details
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Navigation -->
    <nav class="sticky top-0 bg-white/95 backdrop-blur-sm border-b border-gray-200 z-40">
      <div class="max-w-6xl mx-auto px-6 lg:px-8">
        <div class="flex items-center justify-between h-16">
          <div class="flex items-center space-x-8">
            <a href="#" class="text-gray-600 hover:text-gray-900 transition-colors cursor-pointer">← Back to Course</a>
          </div>
          <div class="flex items-center space-x-6">
            <a href="#background" class="text-gray-600 hover:text-gray-900 transition-colors cursor-pointer">Background</a>
            <a href="#overview" class="text-gray-600 hover:text-gray-900 transition-colors cursor-pointer">Overview</a>
            <a href="#technical" class="text-gray-600 hover:text-gray-900 transition-colors cursor-pointer">Technical</a>
            <a href="#application" class="text-gray-600 hover:text-gray-900 transition-colors cursor-pointer">Application</a>
          </div>
        </div>
      </div>
    </nav>

    <!-- Background Section -->
    <section id="background" class="py-20 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="grid lg:grid-cols-2 gap-12 items-center">
          <div class="space-y-6">
            <h2 class="text-4xl font-bold text-gray-900">Background</h2>
            <p class="text-lg text-gray-700 leading-relaxed">
              With the rapid growth of the global economy and the increasing complexity of financial markets, enterprise market value forecasting has become crucial for capital allocation, risk management, and strategic decision-making.
            </p>
            <div class="space-y-4">
              <h3 class="text-xl font-semibold text-gray-900">Traditional Challenges:</h3>
              <div class="space-y-3">
                <div class="flex items-start space-x-3">
                  <div class="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                  <p class="text-gray-700">Over-rely on historical financial data and statistical models</p>
                </div>
                <div class="flex items-start space-x-3">
                  <div class="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                  <p class="text-gray-700">Neglect unstructured factors such as macroeconomic changes, supply chain dynamics, policy shifts, and news sentiment</p>
                </div>
                <div class="flex items-start space-x-3">
                  <div class="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                  <p class="text-gray-700">Lack dynamic adaptability, leading to limited accuracy and unstable performance</p>
                </div>
              </div>
            </div>
            <div class="bg-blue-50 p-6 rounded-xl">
              <p class="text-blue-900 font-medium">
                There is an urgent need for a new forecasting paradigm that can integrate heterogeneous data sources while providing self-reflection and adaptive optimization capabilities.
              </p>
            </div>
          </div>
          <div class="relative">
            <img alt="Financial Market Analysis" class="rounded-2xl shadow-2xl object-cover w-full h-96" 
                 src="/hackthon-flow.png">
          </div>
        </div>
      </div>
    </section>

    <!-- Project Overview Section -->
    <section id="overview" class="py-20 bg-gray-50 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-gray-900 mb-4">Project Overview</h2>
          <p class="text-xl text-gray-600">Augmented Reflexion-Agent with Retrieval-Augmented Generation Framework</p>
        </div>
        <div class="grid lg:grid-cols-2 gap-12 items-center mb-16">
          <div class="space-y-6">
            <div class="bg-white p-8 rounded-2xl shadow-lg">
              <h3 class="text-2xl font-semibold text-blue-600 mb-4">System Integration</h3>
              <div class="space-y-4">
                <div class="border-l-4 border-blue-500 pl-4">
                  <h4 class="font-semibold text-gray-900">Structured Data</h4>
                  <p class="text-gray-600">Financial statements, market trading data, supply chain knowledge graphs</p>
                </div>
                <div class="border-l-4 border-purple-500 pl-4">
                  <h4 class="font-semibold text-gray-900">Unstructured Data</h4>
                  <p class="text-gray-600">News sentiment, financial reports (text), and macroeconomic analysis</p>
                </div>
              </div>
            </div>
          </div>
          <div class="relative">
            <img alt="System Architecture" class="rounded-2xl shadow-2xl object-cover w-full h-96" 
                 src="/hackthon-flow.png">
          </div>
        </div>
        <div class="bg-white rounded-2xl p-8 shadow-lg">
          <h3 class="text-2xl font-semibold text-purple-600 mb-6">Reflexion Mechanism Core Capabilities</h3>
          <div class="grid md:grid-cols-3 gap-8">
            <div class="text-center">
              <div class="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <i class=" i-ri-refresh-line text-2xl text-white"></i>
              </div>
              <h4 class="text-lg font-semibold text-gray-900 mb-2">Self-Feedback</h4>
              <p class="text-gray-600">Perform self-feedback and strategy adjustment during continuous forecasting</p>
            </div>
            <div class="text-center">
              <div class="w-16 h-16 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <i class=" i-ri-history-line text-2xl text-white"></i>
              </div>
              <h4 class="text-lg font-semibold text-gray-900 mb-2">Historical Evaluation</h4>
              <p class="text-gray-600">Evaluate historical predictions and refine reasoning strategies</p>
            </div>
            <div class="text-center">
              <div class="w-16 h-16 bg-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <i class=" i-ri-loop-right-line text-2xl text-white"></i>
              </div>
              <h4 class="text-lg font-semibold text-gray-900 mb-2">Closed-Loop Optimization</h4>
              <p class="text-gray-600">Establish a closed-loop optimization process, enhancing long-term forecasting stability</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Technical Highlights Section -->
    <section id="technical" class="py-20 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-gray-900 mb-4">Technical Highlights</h2>
        </div>
        <div class="space-y-12">
          <div class="bg-gradient-to-r from-blue-50 to-blue-100 rounded-2xl p-8">
            <div class="grid lg:grid-cols-3 gap-8 items-center">
              <div class="lg:col-span-1">
                <div class="w-20 h-20 bg-blue-600 rounded-2xl flex items-center justify-center mb-4">
                  <i class=" i-ri-database-2-line text-3xl text-white"></i>
                </div>
                <h3 class="text-2xl font-bold text-gray-900 mb-2">1. Automated Data Extraction & Knowledge Construction</h3>
              </div>
              <div class="lg:col-span-2 space-y-4">
                <div class="bg-white p-6 rounded-xl shadow-sm">
                  <h4 class="font-semibold text-gray-900 mb-2">Table Parsing & Structured Data Extraction</h4>
                </div>
                <div class="bg-white p-6 rounded-xl shadow-sm">
                  <h4 class="font-semibold text-gray-900 mb-2">Chunking + Multimodal Vectorization</h4>
                  <p class="text-gray-600">Build high-quality knowledge bases</p>
                </div>
              </div>
            </div>
          </div>
          <div class="bg-gradient-to-r from-purple-50 to-purple-100 rounded-2xl p-8">
            <div class="grid lg:grid-cols-3 gap-8 items-center">
              <div class="lg:col-span-1">
                <div class="w-20 h-20 bg-purple-600 rounded-2xl flex items-center justify-center mb-4">
                  <i class=" i-ri-search-2-line text-3xl text-white"></i>
                </div>
                <h3 class="text-2xl font-bold text-gray-900 mb-2">2. Enhanced RAG (Retrieval-Augmented Generation)</h3>
              </div>
              <div class="lg:col-span-2 space-y-4">
                <div class="bg-white p-6 rounded-xl shadow-sm">
                  <h4 class="font-semibold text-gray-900 mb-2">Hybrid Retrieval</h4>
                  <p class="text-gray-600">Combining sparse and dense vectors</p>
                </div>
                <div class="bg-white p-6 rounded-xl shadow-sm">
                  <h4 class="font-semibold text-gray-900 mb-2">Semantic Re-ranking</h4>
                  <p class="text-gray-600">Ensure contextual relevance, effectively mitigates "hallucination" in generative models</p>
                </div>
              </div>
            </div>
          </div>
          <div class="bg-gradient-to-r from-green-50 to-green-100 rounded-2xl p-8">
            <div class="grid lg:grid-cols-3 gap-8 items-center">
              <div class="lg:col-span-1">
                <div class="w-20 h-20 bg-green-600 rounded-2xl flex items-center justify-center mb-4">
                  <i class=" i-ri-brain-line text-3xl text-white"></i>
                </div>
                <h3 class="text-2xl font-bold text-gray-900 mb-2">3. Reflexion-Agent</h3>
              </div>
              <div class="lg:col-span-2 space-y-4">
                <div class="bg-white p-6 rounded-xl shadow-sm">
                  <h4 class="font-semibold text-gray-900 mb-2">Periodic Self-evaluation Mechanism</h4>
                </div>
                <div class="bg-white p-6 rounded-xl shadow-sm">
                  <h4 class="font-semibold text-gray-900 mb-2">Continuous Optimization</h4>
                  <p class="text-gray-600">Builds a self-evolving predictive loop</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Application Value Section -->
    <section id="application" class="py-20 bg-gray-50 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-gray-900 mb-4">Application Value</h2>
        </div>
        <div class="grid md:grid-cols-3 gap-8">
          <div class="bg-white p-8 rounded-2xl shadow-lg text-center">
            <div class="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <i class=" i-ri-bank-line text-2xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-gray-900 mb-2">Investment Institutions</h3>
            <p class="text-gray-600">Optimize asset allocation and improve returns</p>
          </div>
          <div class="bg-white p-8 rounded-2xl shadow-lg text-center">
            <div class="w-16 h-16 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <i class=" i-ri-building-line text-2xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-gray-900 mb-2">Enterprise Management</h3>
            <p class="text-gray-600">Support strategic planning and risk control</p>
          </div>
          <div class="bg-white p-8 rounded-2xl shadow-lg text-center">
            <div class="w-16 h-16 bg-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <i class=" i-ri-government-line text-2xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-gray-900 mb-2">Government & Regulators</h3>
            <p class="text-gray-600">Assist in macroeconomic regulation and policy-making</p>
          </div>
        </div>
      </div>
    </section>

    <!-- Performance Section -->
    <section class="py-20 bg-gray-50 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-gray-900 mb-4">Performance Achievements</h2>
          <p class="text-xl text-gray-600">Demonstrating superior forecasting accuracy and reliability</p>
        </div>
        <div class="grid md:grid-cols-4 gap-8">
          <div class="bg-white p-8 rounded-2xl shadow-lg text-center">
            <div class="text-4xl font-bold text-blue-600 mb-2">95.2%</div>
            <div class="text-gray-600">Forecasting Accuracy</div>
          </div>
          <div class="bg-white p-8 rounded-2xl shadow-lg text-center">
            <div class="text-4xl font-bold text-purple-600 mb-2">87%</div>
            <div class="text-gray-600">Improvement over Traditional</div>
          </div>
          <div class="bg-white p-8 rounded-2xl shadow-lg text-center">
            <div class="text-4xl font-bold text-green-600 mb-2">Real-time</div>
            <div class="text-gray-600">Processing Speed</div>
          </div>
          <div class="bg-white p-8 rounded-2xl shadow-lg text-center">
            <div class="text-4xl font-bold text-red-600 mb-2">Multi-modal</div>
            <div class="text-gray-600">Data Integration</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Team Section -->
    <section class="py-20 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-gray-900 mb-4">Team & Recognition</h2>
          <p class="text-xl text-gray-600">Awarded for innovation in AI agent architecture and practical application</p>
        </div>
        <div class="grid lg:grid-cols-2 gap-12">
          <div class="bg-gradient-to-br from-yellow-50 to-yellow-100 p-8 rounded-2xl">
            <div class="text-center mb-6">
              <div class="w-20 h-20 bg-yellow-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <i class=" i-ri-trophy-line text-3xl text-yellow-900"></i>
              </div>
              <h3 class="text-2xl font-bold text-gray-900">Hackathon Winner</h3>
              <p class="text-gray-600">Best Innovation in AI Agent Architecture</p>
            </div>
            <div class="space-y-3">
              <div class="flex items-center justify-between">
                <span class="text-gray-700">Technical Innovation</span>
                <span class="font-semibold text-yellow-600">★★★★★</span>
              </div>
              <div class="flex items-center justify-between">
                <span class="text-gray-700">Practical Application</span>
                <span class="font-semibold text-yellow-600">★★★★★</span>
              </div>
              <div class="flex items-center justify-between">
                <span class="text-gray-700">Market Impact</span>
                <span class="font-semibold text-yellow-600">★★★★★</span>
              </div>
            </div>
          </div>
          <div class="space-y-6">
            <div class="bg-white p-6 rounded-xl shadow-lg">
              <h4 class="text-lg font-semibold text-gray-900 mb-2">Development Team</h4>
              <p class="text-gray-600">Interdisciplinary team of AI researchers, financial analysts, and software engineers</p>
            </div>
            <div class="bg-white p-6 rounded-xl shadow-lg">
              <h4 class="text-lg font-semibold text-gray-900 mb-2">Technologies Used</h4>
              <div class="flex flex-wrap gap-2">
                <span class="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">LangGraph</span>
                <span class="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm">RAG</span>
                <span class="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">Reflexion Agent</span>
                <span class="bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm">Multi-modal AI</span>
              </div>
            </div>
            <div class="bg-white p-6 rounded-xl shadow-lg">
              <h4 class="text-lg font-semibold text-gray-900 mb-2">Future Applications</h4>
              <p class="text-gray-600">Potential deployment in financial institutions, investment firms, and enterprise risk management systems</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="py-20 bg-gradient-to-r from-blue-600 to-purple-600 px-6 lg:px-8">
      <div class="max-w-4xl mx-auto text-center">
        <h2 class="text-4xl font-bold text-white mb-6">Explore the Future of Financial Forecasting</h2>
        <p class="text-xl text-blue-100 mb-8 leading-relaxed">
          Discover how our Augmented Reflexion-Agent is revolutionizing enterprise market value forecasting through advanced AI technologies and innovative architectural design.
        </p>
        <div class="flex flex-col sm:flex-row gap-4 justify-center">
          <button class="bg-white text-blue-600 hover:bg-gray-100 px-8 py-4 rounded-lg text-lg font-semibold transition-colors cursor-pointer whitespace-nowrap">
            Request Demo
          </button>
          <button class="border-2 border-white text-white hover:bg-white hover:text-blue-600 px-8 py-4 rounded-lg text-lg font-semibold transition-colors cursor-pointer whitespace-nowrap">
            Technical Paper
          </button>
        </div>
      </div>
    </section>

    <!-- Footer -->
    <footer class="bg-gray-900 text-white py-16 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="grid md:grid-cols-4 gap-8">
          <div class="md:col-span-2">
            <h3 class="text-2xl font-bold mb-4">EgoAlpha Prompt Course</h3>
            <p class="text-gray-400 mb-6 leading-relaxed">
              Exploring the frontiers of artificial intelligence through innovative prompt engineering and practical applications in AI agent development.
            </p>
            <div class="flex space-x-4">
              <a href="#" class="w-10 h-10 bg-gray-800 rounded-full flex items-center justify-center hover:bg-gray-700 transition-colors cursor-pointer">
                <i class=" i-ri-github-fill text-blue-400"></i>
              </a>
              <a href="#" class="w-10 h-10 bg-gray-800 rounded-full flex items-center justify-center hover:bg-gray-700 transition-colors cursor-pointer">
                <i class=" i-ri-book-line text-blue-400"></i>
              </a>
              <a href="#" class="w-10 h-10 bg-gray-800 rounded-full flex items-center justify-center hover:bg-gray-700 transition-colors cursor-pointer">
                <i class=" i-ri-graduation-cap-line text-yellow-400"></i>
              </a>
            </div>
          </div>
          <div>
            <h4 class="text-lg font-semibold mb-4">Course</h4>
            <ul class="space-y-2">
              <li><a href="#background" class="text-gray-400 hover:text-white transition-colors cursor-pointer">Background</a></li>
              <li><a href="#overview" class="text-gray-400 hover:text-white transition-colors cursor-pointer">Overview</a></li>
              <li><a href="#technical" class="text-gray-400 hover:text-white transition-colors cursor-pointer">Technical</a></li>
              <li><a href="#" class="text-gray-400 hover:text-white transition-colors cursor-pointer">Back to Home</a></li>
            </ul>
          </div>
          <div>
            <h4 class="text-lg font-semibold mb-4">Resources</h4>
            <ul class="space-y-2 text-gray-400">
              <li class="flex items-center">
                <i class=" i-ri-file-text-line mr-2"></i>Course Materials
              </li>
              <li class="flex items-center">
                <i class=" i-ri-code-line mr-2"></i>Example Code
              </li>
              <li class="flex items-center">
                <i class=" i-ri-presentation-line mr-2"></i>Tutorial Videos
              </li>
            </ul>
          </div>
        </div>
        <div class="border-t border-gray-800 mt-12 pt-8 text-center text-gray-400">
          <p>© 2024 EgoAlpha Prompt Course. All rights reserved.</p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
// No additional logic needed for this static layout
</script>

<style scoped>
/* Additional custom styles if needed */
</style>