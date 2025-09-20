<template>
  <FullPageLayout>
    <div class="symposium-page">
      <!-- 动态渐变背景 -->
      <div class="animated-background"></div>
      
      <!-- 浮动粒子效果 -->
      <div class="particles">
        <div 
          v-for="i in 50" 
          :key="i" 
          class="particle"
          :style="getParticleStyle(i)"
        ></div>
      </div>

      <!-- 英雄区域 -->
      <section class="hero-section">
        <div class="hero-content">
          <h1 class="main-title">
            <span class="title-gradient">AI Agent &</span>
            <span class="title-highlight">LangGraph 研讨会</span>
          </h1>
          <p class="hero-subtitle">探索人工智能的未来，构建智能化的明天</p>
          <div class="hero-badges">
            <span class="badge pulse">🚀 前沿技术</span>
            <span class="badge">🎯 实战案例</span>
            <span class="badge">🌟 顶级专家</span>
            <span class="badge">🔥 免费参会</span>
          </div>
        </div>
      </section>

      <!-- 内容容器 -->
      <div class="content-container">
        <!-- 研讨会介绍 -->
        <section class="glass-section">
          <div class="section-header">
            <h2 class="section-title">✨ 研讨会简介</h2>
          </div>
          <p class="intro-text">
            欢迎参加AI Agent & LangGraph研讨会！本次研讨会汇聚了人工智能领域的顶尖专家，
            共同探讨AI Agent技术的最新发展和LangGraph框架的创新应用。我们将深入探索
            多智能体系统、知识图谱推理、智能对话系统等前沿技术，为您带来最具价值的
            技术分享和实践经验。
          </p>
        </section>

        <!-- 演讲者介绍 -->
        <section class="glass-section">
          <div class="section-header">
            <h2 class="section-title">🎤 主讲嘉宾</h2>
          </div>
          <div class="speaker-grid">
            <div class="speaker-card">
              <div class="speaker-avatar">
                <div class="avatar-glow"></div>
                <span class="avatar-circle">👨‍🎓</span>
              </div>
              <h3 class="speaker-name">张伟博士</h3>
              <p class="speaker-title">清华大学人工智能研究院 教授</p>
              <p class="speaker-research">专注于多智能体系统和强化学习研究</p>
              <h4 class="talk-title">《多智能体协作的理论基础与实践》</h4>
              <p class="talk-abstract">
                本次演讲将深入探讨多智能体系统的协作机制，包括通信协议、任务分配算法，
                以及在实际应用中的挑战与解决方案。我们将通过具体案例展示如何构建高效的
                多智能体协作系统。
              </p>
            </div>

            <div class="speaker-card">
              <div class="speaker-avatar">
                <div class="avatar-glow"></div>
                <span class="avatar-circle">👨‍💻</span>
              </div>
              <h3 class="speaker-name">李明教授</h3>
              <p class="speaker-title">北京大学计算机科学技术系 副教授</p>
              <p class="speaker-research">自然语言处理和知识图谱专家</p>
              <h4 class="talk-title">《LangGraph在知识推理中的应用》</h4>
              <p class="talk-abstract">
                LangGraph作为新兴的图神经网络框架，在知识推理任务中展现出巨大潜力。
                本演讲将介绍LangGraph的核心原理，以及在问答系统、推荐系统等场景中的
                具体应用案例。
              </p>
            </div>

            <div class="speaker-card">
              <div class="speaker-avatar">
                <div class="avatar-glow"></div>
                <span class="avatar-circle">👩‍🔬</span>
              </div>
              <h3 class="speaker-name">王芳博士</h3>
              <p class="speaker-title">微软亚洲研究院 首席研究员</p>
              <p class="speaker-research">对话系统和智能助手技术专家</p>
              <h4 class="talk-title">《构建下一代智能对话Agent》</h4>
              <p class="talk-abstract">
                随着大语言模型的快速发展，智能对话Agent正在向更加自然、智能的方向演进。
                本演讲将分享微软在构建智能对话系统方面的最新研究成果和技术突破。
              </p>
            </div>

            <div class="speaker-card">
              <div class="speaker-avatar">
                <div class="avatar-glow"></div>
                <span class="avatar-circle">👨‍💼</span>
              </div>
              <h3 class="speaker-name">陈浩先生</h3>
              <p class="speaker-title">字节跳动AI Lab 技术总监</p>
              <p class="speaker-research">大规模AI系统架构和工程化专家</p>
              <h4 class="talk-title">《AI Agent的工程化实践与挑战》</h4>
              <p class="talk-abstract">
                从实验室到生产环境，AI Agent面临着诸多工程化挑战。本演讲将分享字节跳动
                在大规模部署AI Agent系统方面的实践经验，包括系统架构设计、性能优化、
                可靠性保障等关键技术。
              </p>
            </div>
          </div>
        </section>

        <!-- 会议信息 -->
        <section class="glass-section">
          <div class="section-header">
            <h2 class="section-title">📋 会议信息</h2>
          </div>
          <div class="info-grid">
            <div class="info-card">
              <span class="info-icon">📅</span>
              <h3>时间安排</h3>
              <p>2024年3月15日<br>上午9:00 - 下午5:00</p>
            </div>
            <div class="info-card">
              <span class="info-icon">📍</span>
              <h3>会议地点</h3>
              <p>北京国际会议中心<br>第一会议厅</p>
            </div>
            <div class="info-card">
              <span class="info-icon">👥</span>
              <h3>参会人数</h3>
              <p>限额200人<br>先到先得</p>
            </div>
            <div class="info-card">
              <span class="info-icon">💰</span>
              <h3>参会费用</h3>
              <p>免费参会<br>提供茶歇和午餐</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  </FullPageLayout>
</template>

<script setup>
import FullPageLayout from './FullPageLayout.vue'

// 生成粒子样式
const getParticleStyle = (index) => {
  const size = Math.random() * 4 + 2
  const left = Math.random() * 100
  const animationDelay = Math.random() * 20
  const animationDuration = Math.random() * 10 + 10
  
  return {
    width: `${size}px`,
    height: `${size}px`,
    left: `${left}%`,
    animationDelay: `${animationDelay}s`,
    animationDuration: `${animationDuration}s`
  }
}
</script>

<style scoped>
.symposium-page {
  position: relative;
  min-height: 100vh;
  overflow-x: hidden;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* 动态渐变背景 */
.animated-background {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
  background-size: 400% 400%;
  animation: gradientShift 15s ease infinite;
  z-index: -2;
}

@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* 浮动粒子效果 */
.particles {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: -1;
}

.particle {
  position: absolute;
  background: rgba(255, 255, 255, 0.6);
  border-radius: 50%;
  animation: float linear infinite;
}

@keyframes float {
  0% {
    transform: translateY(100vh) rotate(0deg);
    opacity: 0;
  }
  10% {
    opacity: 1;
  }
  90% {
    opacity: 1;
  }
  100% {
    transform: translateY(-100px) rotate(360deg);
    opacity: 0;
  }
}

/* 英雄区域 */
.hero-section {
  position: relative;
  padding: 6rem 2rem;
  text-align: center;
  color: white;
}

.hero-content {
  max-width: 1000px;
  margin: 0 auto;
}

.main-title {
  font-size: 4rem;
  font-weight: 800;
  margin-bottom: 1.5rem;
  line-height: 1.2;
}

.title-gradient {
  background: linear-gradient(45deg, #fff, #f0f0f0, #e0e0e0);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  display: block;
}

.title-highlight {
  background: linear-gradient(45deg, #ffd700, #ffed4e, #fff59d);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  display: block;
  margin-top: 0.5rem;
}

.hero-subtitle {
  font-size: 1.5rem;
  margin-bottom: 2rem;
  opacity: 0.9;
  font-weight: 300;
}

.hero-badges {
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
}

.badge {
  background: rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.3);
  padding: 0.8rem 1.5rem;
  border-radius: 30px;
  font-weight: 600;
  font-size: 1rem;
}

.pulse {
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}

/* 内容容器 */
.content-container {
  position: relative;
  max-width: 1200px;
  margin: 0 auto;
  padding: 3rem 2rem;
  z-index: 1;
}

/* 玻璃拟态区域 */
.glass-section {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  margin-bottom: 3rem;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease;
}

.glass-section:hover {
  transform: translateY(-5px);
}

.section-header {
  background: rgba(255, 255, 255, 0.1);
  padding: 2rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.section-title {
  font-size: 2.5rem;
  font-weight: 700;
  color: white;
  margin: 0;
  text-align: center;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

/* 介绍区域 */
.intro-text {
  font-size: 1.3rem;
  color: rgba(255, 255, 255, 0.9);
  line-height: 1.8;
  text-align: center;
  padding: 2rem;
  margin: 0;
}

/* 演讲者网格 */
.speaker-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 2rem;
  padding: 2rem;
}

.speaker-card {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(15px);
  border-radius: 20px;
  padding: 2rem;
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.speaker-card:hover {
  transform: translateY(-10px);
  background: rgba(255, 255, 255, 0.15);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
}

.speaker-avatar {
  position: relative;
  text-align: center;
  margin-bottom: 1.5rem;
}

.avatar-circle {
  font-size: 4rem;
  display: inline-block;
  position: relative;
  z-index: 2;
}

.avatar-glow {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 80px;
  height: 80px;
  background: radial-gradient(circle, rgba(255, 255, 255, 0.3), transparent);
  border-radius: 50%;
  z-index: 1;
  animation: glow 3s ease-in-out infinite alternate;
}

@keyframes glow {
  0% { opacity: 0.5; transform: translate(-50%, -50%) scale(1); }
  100% { opacity: 1; transform: translate(-50%, -50%) scale(1.2); }
}

.speaker-name {
  font-size: 1.8rem;
  font-weight: 700;
  color: white;
  margin-bottom: 0.5rem;
  text-align: center;
}

.speaker-title {
  font-size: 1.1rem;
  color: #4facfe;
  margin-bottom: 0.5rem;
  text-align: center;
  font-weight: 600;
}

.speaker-research {
  font-size: 1rem;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 1.5rem;
  text-align: center;
  font-style: italic;
}

.talk-title {
  font-size: 1.4rem;
  font-weight: 600;
  color: white;
  margin-bottom: 1rem;
  line-height: 1.4;
}

.talk-abstract {
  font-size: 1rem;
  color: rgba(255, 255, 255, 0.8);
  line-height: 1.7;
  margin: 0;
}

/* 会议信息 */
.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  padding: 2rem;
}

.info-card {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 2rem;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
}

.info-card:hover {
  transform: translateY(-5px);
  background: rgba(255, 255, 255, 0.15);
}

.info-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  display: block;
}

.info-card h3 {
  font-size: 1.4rem;
  font-weight: 600;
  color: white;
  margin-bottom: 1rem;
}

.info-card p {
  color: rgba(255, 255, 255, 0.8);
  font-size: 1rem;
  line-height: 1.6;
  margin: 0;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .main-title {
    font-size: 2.5rem;
  }
  
  .hero-section {
    padding: 4rem 1rem;
  }
  
  .content-container {
    padding: 2rem 1rem;
  }
  
  .speaker-grid {
    grid-template-columns: 1fr;
    padding: 1rem;
  }
  
  .info-grid {
    grid-template-columns: 1fr;
    padding: 1rem;
  }
  
  .hero-badges {
    flex-direction: column;
    align-items: center;
  }
  
  .section-header {
    padding: 1.5rem;
  }
  
  .section-title {
    font-size: 2rem;
  }
}

@media (max-width: 480px) {
  .main-title {
    font-size: 2rem;
  }
  
  .hero-subtitle {
    font-size: 1.2rem;
  }
  
  .speaker-card {
    padding: 1.5rem;
  }
  
  .info-card {
    padding: 1.5rem;
  }
}
</style>