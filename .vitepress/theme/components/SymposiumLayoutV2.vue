<template>
  <div class="min-h-screen bg-white">
    <!-- Hero Section -->
    <section class="relative min-h-screen bg-cover bg-center bg-no-repeat flex items-center" 
             style="background-image: linear-gradient(rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.6)), url('/bg.jpg');">
      <div class="w-full px-6 lg:px-8">
        <div class="max-w-4xl">
          <div class="text-white space-y-8">
            <div class="space-y-4">
              <h1 class="text-5xl lg:text-6xl font-bold leading-tight">
                Symposium: Frontiers in
                <span class="block text-blue-400">Natural Language Processing</span>
                <span class="block text-purple-400">and AI Agents</span>
              </h1>
              <p class="text-xl lg:text-2xl text-gray-300 max-w-3xl leading-relaxed">
                Exploring the cutting-edge developments in NLP and AI agent technologies through expert insights and innovative research.
              </p>
            </div>
            <!-- <div class="flex flex-col sm:flex-row gap-4">
              <button class="bg-blue-500 hover:bg-blue-600 text-white px-8 py-4 rounded-lg text-lg font-semibold transition-colors cursor-pointer whitespace-nowrap">
                Join Symposium
              </button>
              <button class="border-2 border-white text-white hover:bg-white hover:text-gray-900 px-8 py-4 rounded-lg text-lg font-semibold transition-colors cursor-pointer whitespace-nowrap">
                View Schedule
              </button>
            </div> -->
          </div>
        </div>
      </div>
    </section>

    <!-- Navigation -->
    <nav class="sticky top-0 bg-white/95 backdrop-blur-sm border-b border-gray-200 z-40">
      <div class="max-w-6xl mx-auto px-6 lg:px-8">
        <div class="flex items-center justify-between h-16">
          <div class="flex items-center space-x-8">
            <a href="/" class="text-gray-600 hover:text-gray-900 transition-colors cursor-pointer">← Back to Course</a>
          </div>
          <div class="flex items-center space-x-6">
            <a href="#speakers" class="text-gray-600 hover:text-gray-900 transition-colors cursor-pointer">Speakers</a>
            <a href="#schedule" class="text-gray-600 hover:text-gray-900 transition-colors cursor-pointer">Schedule</a>
            <a href="#gallery" class="text-gray-600 hover:text-gray-900 transition-colors cursor-pointer">Gallery</a>
          </div>
        </div>
      </div>
    </nav>

    <!-- About Section -->
    <section class="py-20 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="grid lg:grid-cols-2 gap-12 items-center">
          <div class="space-y-6">
            <h2 class="text-4xl font-bold text-gray-900">About the Symposium</h2>
            <p class="text-lg text-gray-700 leading-relaxed">
              As advances in cognitive architectures, language technologies, and large-scale models accelerate, new opportunities and challenges are emerging for building intelligent systems that can understand, reason, and interact in complex environments.
            </p>
            <p class="text-lg text-gray-700 leading-relaxed">
              The symposium aims to provide a platform for exchanging ideas, fostering collaboration, and exploring future directions that bridge theory and practice in NLP and AI agent research.
            </p>
            <div class="grid grid-cols-2 gap-6 pt-6">
              <div class="text-center">
                <div class="text-3xl font-bold text-blue-600">50+</div>
                <div class="text-gray-600">Researchers</div>
              </div>
              <div class="text-center">
                <div class="text-3xl font-bold text-purple-600">2</div>
                <div class="text-gray-600">Keynote Speakers</div>
              </div>
            </div>
          </div>
          <div class="relative">
            <img alt="AI Research" class="rounded-2xl shadow-2xl object-cover w-full h-96" 
                 src="/symposium-hero.png">
          </div>
        </div>
      </div>
    </section>

    <!-- Keynote Speakers Section -->
    <section id="speakers" class="py-20 bg-gray-50 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-gray-900 mb-4">Keynote Speakers</h2>
          <p class="text-xl text-gray-600">Distinguished experts sharing cutting-edge insights</p>
        </div>
        <div class="space-y-16">
          <!-- Speaker 1 -->
          <div class="bg-white rounded-2xl p-8 shadow-lg">
            <div class="grid lg:grid-cols-3 gap-8 items-start">
              <div class="lg:col-span-1 text-center">
                <img alt="Prof. Yu Liu" class="w-48 h-64 mx-auto rounded-xl object-cover object-top mb-4" 
                     src="/prof-yu-liu.png">
                <h3 class="text-2xl font-bold text-gray-900">Prof. Yu Liu</h3>
                <p class="text-gray-600 mb-2">Dalian University of Technology</p>
                <p class="text-sm text-gray-500">Professor in Software School</p>
                <p class="text-sm text-gray-500">Head of Institute of Machine Intelligence and Informetrics</p>
              </div>
              <div class="lg:col-span-2 space-y-4">
                <div>
                  <h4 class="text-2xl font-semibold text-blue-600 mb-3">
                    Advancing AI Agents: From Theory to Practice with LangGraph
                  </h4>
                  <div class="space-y-3 text-gray-700 leading-relaxed">
                    <p>AI agents are increasingly deployed in domains ranging from personal assistants to large-scale autonomous systems. Their performance largely depends on the cognitive architectures that support perception, reasoning, planning, and action in complex environments.</p>
                    <p>In this talk, I will present a comparative overview of existing cognitive architectures for AI agents, outlining their core features, strengths, and limitations. I will then discuss how LangGraph, an open-source framework, can be leveraged to implement and evaluate diverse cognitive architectures in practice.</p>
                    <p>Through demonstrations, I will showcase how LangGraph facilitates the development of AI agents and enables systematic performance comparisons across different cognitive models.</p>
                  </div>
                </div>
                <div class="flex flex-wrap gap-2 pt-4">
                  <span class="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">AI Agent</span>
                  <span class="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm">Knowledge Graph</span>
                  <span class="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">Swarm Intelligence</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Speaker 2 -->
          <div class="bg-white rounded-2xl p-8 shadow-lg">
            <div class="grid lg:grid-cols-3 gap-8 items-start">
              <div class="lg:col-span-1 text-center">
                <img alt="Dr. Zheng Yuan" class="w-48 h-64 mx-auto rounded-xl object-cover object-top mb-4" 
                     src="/dr-zheng-yuan.png">
                <h3 class="text-2xl font-bold text-gray-900">Dr. Zheng Yuan</h3>
                <p class="text-gray-600 mb-2">King's College London</p>
                <p class="text-sm text-gray-500">Assistant Professor</p>
                <p class="text-sm text-gray-500">Visiting Researcher at University of Cambridge</p>
                <p class="text-sm text-gray-500">Fellow at Trinity College</p>
              </div>
              <div class="lg:col-span-2 space-y-4">
                <div>
                  <h4 class="text-2xl font-semibold text-purple-600 mb-3">
                    On the application of NLP in language education
                  </h4>
                  <div class="space-y-3 text-gray-700 leading-relaxed">
                    <p>Language education is the process and practice of teaching or learning a second or foreign language. Natural Language Processing (NLP) technologies have been widely applied to help with language teaching and assessment.</p>
                    <p>In this talk, I will focus on three educational NLP tasks: generating AI teacher responses in educational dialogues, automated assessment, and grammatical error correction.</p>
                    <p>Towards the end, I'll showcase some potential applications of Large Language Models (LLMs) for language education, and discuss the risks and ethical considerations surrounding generative AI in education technology for language learners.</p>
                  </div>
                </div>
                <div class="flex flex-wrap gap-2 pt-4">
                  <span class="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">Machine Learning</span>
                  <span class="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm">Deep Learning</span>
                  <span class="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">Language Education</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Schedule Section -->
    <section id="schedule" class="py-20 bg-gray-50 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-gray-900 mb-4">Symposium Schedule</h2>
          <p class="text-xl text-gray-600">A comprehensive program covering the latest in NLP and AI agent research</p>
        </div>
        <div class="space-y-8">
          <div class="bg-white rounded-2xl p-8 shadow-lg">
            <div class="grid md:grid-cols-4 gap-6 items-center">
              <div class="text-center">
                <div class="text-2xl font-bold text-blue-600">09:00</div>
                <div class="text-gray-600">Opening</div>
              </div>
              <div class="md:col-span-3">
                <h3 class="text-xl font-semibold text-gray-900 mb-2">Welcome & Introduction</h3>
                <p class="text-gray-600">Opening remarks and symposium overview</p>
              </div>
            </div>
          </div>
          <div class="bg-white rounded-2xl p-8 shadow-lg">
            <div class="grid md:grid-cols-4 gap-6 items-center">
              <div class="text-center">
                <div class="text-2xl font-bold text-blue-600">09:30</div>
                <div class="text-gray-600">Keynote 1</div>
              </div>
              <div class="md:col-span-3">
                <h3 class="text-xl font-semibold text-gray-900 mb-2">Advancing AI Agents: From Theory to Practice with LangGraph</h3>
                <p class="text-gray-600">Prof. Yu Liu - Dalian University of Technology</p>
              </div>
            </div>
          </div>
          <div class="bg-white rounded-2xl p-8 shadow-lg">
            <div class="grid md:grid-cols-4 gap-6 items-center">
              <div class="text-center">
                <div class="text-2xl font-bold text-purple-600">11:00</div>
                <div class="text-gray-600">Keynote 2</div>
              </div>
              <div class="md:col-span-3">
                <h3 class="text-xl font-semibold text-gray-900 mb-2">On the application of NLP in language education</h3>
                <p class="text-gray-600">Dr. Zheng Yuan - King's College London</p>
              </div>
            </div>
          </div>
          <div class="bg-white rounded-2xl p-8 shadow-lg">
            <div class="grid md:grid-cols-4 gap-6 items-center">
              <div class="text-center">
                <div class="text-2xl font-bold text-green-600">14:00</div>
                <div class="text-gray-600">Panel</div>
              </div>
              <div class="md:col-span-3">
                <h3 class="text-xl font-semibold text-gray-900 mb-2">Future Directions in AI Research</h3>
                <p class="text-gray-600">Interactive discussion with all speakers and audience Q&A</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>


    <!-- Research Topics Section -->
    <section class="py-20 bg-gray-50 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-gray-900 mb-4">Research Focus Areas</h2>
          <p class="text-xl text-gray-600">Key topics covered in our symposium</p>
        </div>
        <div class="grid md:grid-cols-3 gap-8">
          <div class="bg-white p-8 rounded-2xl shadow-lg text-center">
            <div class="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <i class=" i-ri-brain-line text-2xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-gray-900 mb-2">Multimodal AI</h3>
            <p class="text-gray-600">Integration of vision, language, and other modalities in AI systems</p>
          </div>
          <div class="bg-white p-8 rounded-2xl shadow-lg text-center">
            <div class="w-16 h-16 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <i class=" i-ri-robot-line text-2xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-gray-900 mb-2">Autonomous Agents</h3>
            <p class="text-gray-600">Self-directed AI systems capable of independent decision-making</p>
          </div>
          <div class="bg-white p-8 rounded-2xl shadow-lg text-center">
            <div class="w-16 h-16 bg-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <i class=" i-ri-chat-3-line text-2xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-gray-900 mb-2">Language Understanding</h3>
            <p class="text-gray-600">Advanced natural language processing and comprehension techniques</p>
          </div>
        </div>
      </div>
    </section>

    <!-- Event Details Section -->
    <!-- <section class="py-20 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="grid lg:grid-cols-3 gap-8">
          <div class="bg-gradient-to-br from-blue-50 to-blue-100 p-8 rounded-2xl text-center">
            <div class="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <i class=" i-ri-calendar-line text-2xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-gray-900 mb-2">Date & Time</h3>
            <p class="text-gray-700">March 15-16, 2024</p>
            <p class="text-gray-700">9:00 AM - 6:00 PM</p>
          </div>
          <div class="bg-gradient-to-br from-purple-50 to-purple-100 p-8 rounded-2xl text-center">
            <div class="w-16 h-16 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <i class=" i-ri-map-pin-line text-2xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-gray-900 mb-2">Location</h3>
            <p class="text-gray-700">Academic Conference Center</p>
            <p class="text-gray-700">University Campus</p>
          </div>
          <div class="bg-gradient-to-br from-green-50 to-green-100 p-8 rounded-2xl text-center">
            <div class="w-16 h-16 bg-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <i class=" i-ri-group-line text-2xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-gray-900 mb-2">Attendees</h3>
            <p class="text-gray-700">Researchers</p>
            <p class="text-gray-700">Practitioners</p>
          </div>
        </div>
      </div>
    </section> -->

    <!-- Research Topics Section -->
    <!-- <section class="py-20 bg-gray-900 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-white mb-4">Research Topics</h2>
          <p class="text-xl text-gray-300">Key areas of discussion and exploration</p>
        </div>
        <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div class="bg-gray-800 p-6 rounded-xl hover:bg-gray-700 transition-colors cursor-pointer">
            <div class="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center mb-4">
              <i class=" i-ri-brain-line text-xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-white mb-2">Cognitive Architectures</h3>
            <p class="text-gray-400">Advanced frameworks for AI reasoning and decision-making</p>
          </div>
          <div class="bg-gray-800 p-6 rounded-xl hover:bg-gray-700 transition-colors cursor-pointer">
            <div class="w-12 h-12 bg-purple-600 rounded-lg flex items-center justify-center mb-4">
              <i class=" i-ri-translate-line text-xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-white mb-2">Language Technologies</h3>
            <p class="text-gray-400">Cutting-edge NLP methods and applications</p>
          </div>
          <div class="bg-gray-800 p-6 rounded-xl hover:bg-gray-700 transition-colors cursor-pointer">
            <div class="w-12 h-12 bg-green-600 rounded-lg flex items-center justify-center mb-4">
              <i class=" i-ri-robot-line text-xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-white mb-2">AI Agents</h3>
            <p class="text-gray-400">Intelligent systems for complex environments</p>
          </div>
          <div class="bg-gray-800 p-6 rounded-xl hover:bg-gray-700 transition-colors cursor-pointer">
            <div class="w-12 h-12 bg-red-600 rounded-lg flex items-center justify-center mb-4">
              <i class=" i-ri-database-line text-xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-white mb-2">Large-Scale Models</h3>
            <p class="text-gray-400">Training and deployment of massive AI systems</p>
          </div>
          <div class="bg-gray-800 p-6 rounded-xl hover:bg-gray-700 transition-colors cursor-pointer">
            <div class="w-12 h-12 bg-yellow-600 rounded-lg flex items-center justify-center mb-4">
              <i class=" i-ri-graduation-cap-line text-xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-white mb-2">Educational Applications</h3>
            <p class="text-gray-400">NLP technologies in language learning</p>
          </div>
          <div class="bg-gray-800 p-6 rounded-xl hover:bg-gray-700 transition-colors cursor-pointer">
            <div class="w-12 h-12 bg-indigo-600 rounded-lg flex items-center justify-center mb-4">
              <i class=" i-ri-code-line text-xl text-white"></i>
            </div>
            <h3 class="text-xl font-semibold text-white mb-2">Practical Implementation</h3>
            <p class="text-gray-400">Bridging theory and real-world applications</p>
          </div>
        </div>
      </div>
    </section> -->

    
    <!-- Gallery Section -->
    <section id="gallery" class="py-20 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-gray-900 mb-4">Symposium Gallery</h2>
          <p class="text-xl text-gray-600">Highlights from our symposium on NLP and AI agents</p>
        </div>
        <div class="grid md:grid-cols-2 gap-8">
          <div class="relative group">
            <img alt="Symposium Conference Photo 1" class="rounded-2xl shadow-lg object-cover w-full h-80 group-hover:scale-105 transition-transform duration-300" 
                 src="/symposium-conference1.png">
            <div class="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 rounded-2xl transition-all duration-300"></div>
          </div>
          <div class="relative group">
            <img alt="Symposium Conference Photo 2" class="rounded-2xl shadow-lg object-cover w-full h-80 group-hover:scale-105 transition-transform duration-300" 
                 src="/symposium-conference2.png">
            <div class="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 rounded-2xl transition-all duration-300"></div>
          </div>
        </div>
      </div>
    </section>
    <!-- CTA Section -->
    <!-- <section class="py-20 bg-gradient-to-r from-blue-600 to-purple-600 px-6 lg:px-8">
      <div class="max-w-4xl mx-auto text-center">
        <h2 class="text-4xl font-bold text-white mb-6">Join the Future of AI Research</h2>
        <p class="text-xl text-blue-100 mb-8 leading-relaxed">
          Be part of groundbreaking discussions that will shape the future of natural language processing and AI agents. Connect with leading researchers and discover the latest innovations.
        </p>
        <div class="flex flex-col sm:flex-row gap-4 justify-center">
          <button class="bg-white text-blue-600 hover:bg-gray-100 px-8 py-4 rounded-lg text-lg font-semibold transition-colors cursor-pointer whitespace-nowrap">
            Register Today
          </button>
          <button class="border-2 border-white text-white hover:bg-white hover:text-blue-600 px-8 py-4 rounded-lg text-lg font-semibold transition-colors cursor-pointer whitespace-nowrap">
            Download Program
          </button>
        </div>
      </div>
    </section> -->

    <!-- Footer -->
    <!-- <footer class="bg-gray-900 text-white py-16 px-6 lg:px-8">
      <div class="max-w-6xl mx-auto">
        <div class="grid md:grid-cols-4 gap-8">
          <div class="md:col-span-2">
            <h3 class="text-2xl font-bold mb-4">EgoAlpha Prompt Course</h3>
            <p class="text-gray-400 mb-6 leading-relaxed">
              Advancing the frontiers of natural language processing and artificial intelligence through comprehensive prompt engineering education and research.
            </p>
            <div class="flex space-x-4">
              <a href="#" class="w-10 h-10 bg-gray-800 rounded-full flex items-center justify-center hover:bg-gray-700 transition-colors cursor-pointer">
                <i class=" i-ri-book-line text-blue-400"></i>
              </a>
              <a href="#" class="w-10 h-10 bg-gray-800 rounded-full flex items-center justify-center hover:bg-gray-700 transition-colors cursor-pointer">
                <i class=" i-ri-graduation-cap-line text-blue-400"></i>
              </a>
              <a href="#" class="w-10 h-10 bg-gray-800 rounded-full flex items-center justify-center hover:bg-gray-700 transition-colors cursor-pointer">
                <i class=" i-ri-github-fill text-blue-400"></i>
              </a>
            </div>
          </div>
          <div>
            <h4 class="text-lg font-semibold mb-4">Course</h4>
            <ul class="space-y-2">
              <li><a href="#" class="text-gray-400 hover:text-white transition-colors cursor-pointer">Curriculum</a></li>
              <li><a href="#" class="text-gray-400 hover:text-white transition-colors cursor-pointer">Instructors</a></li>
              <li><a href="#" class="text-gray-400 hover:text-white transition-colors cursor-pointer">Enrollment</a></li>
              <li><a href="#" class="text-gray-400 hover:text-white transition-colors cursor-pointer">Resources</a></li>
            </ul>
          </div>
          <div>
            <h4 class="text-lg font-semibold mb-4">Contact</h4>
            <ul class="space-y-2 text-gray-400">
              <li class="flex items-center">
                <i class=" i-ri-mail-line mr-2"></i>info@egoalpha.com
              </li>
              <li class="flex items-center">
                <i class=" i-ri-book-open-line mr-2"></i>Course Materials
              </li>
              <li class="flex items-center">
                <i class=" i-ri-question-line mr-2"></i>Support Center
              </li>
            </ul>
          </div>
        </div>
        <div class="border-t border-gray-800 mt-12 pt-8 text-center text-gray-400">
          <p>© 2024 EgoAlpha Prompt Course. All rights reserved.</p>
        </div>
      </div>
    </footer> -->
  </div>
</template>

<script setup>
// No additional logic needed for this static layout
</script>

<style scoped>
/* Additional custom styles if needed */
</style>