<template>
  <FullPageLayout>
    <div class="hackthon-page">
      <!-- 科技感动态背景 -->
      <div class="tech-background">
        <div class="matrix-rain"></div>
        <div class="circuit-pattern"></div>
        <div class="floating-elements">
          <div v-for="i in 30" :key="i" class="floating-element" :style="getFloatingStyle(i)">
            {{ getRandomTechIcon() }}
          </div>
        </div>
      </div>

      <!-- 英雄区域 -->
      <section class="hero-section">
        <div class="hero-content">
          <h1 class="main-title">
            <span class="title-tech">AI Agent &</span>
            <span class="title-highlight">LangGraph</span>
            <span class="title-event">Hackathon</span>
          </h1>
          <p class="hero-subtitle">48小时编程马拉松 · 创新无界限 · 智能新纪元</p>
          <div class="hero-badges">
            <span class="badge neon-blue">⚡ 48小时挑战</span>
            <span class="badge neon-purple">🏆 $50K奖金池</span>
            <span class="badge neon-green">🚀 前沿技术</span>
            <span class="badge neon-orange">💡 创新思维</span>
          </div>
        </div>
      </section>

      <!-- 内容容器 -->
      <div class="content-container">
        <!-- 获奖者展示 -->
        <section class="glass-section winner-section">
          <div class="section-header">
            <div class="winner-crown">👑</div>
            <h2 class="section-title">🏆 冠军项目</h2>
          </div>
          
          <div class="winner-project">
            <h3 class="project-title">
              基于增强反思Agent的企业市值预测系统
            </h3>
            
            <div class="project-showcase">
              <div class="project-intro">
                <p>
                  我们的获奖项目通过结合先进的AI智能体与反思学习机制，革命性地改变了企业市值预测方式，
                  实现了对复杂金融市场的深度理解和精准预测。
                </p>
              </div>
              
              <div class="tech-highlights">
                <div class="highlight-card">
                  <div class="highlight-icon">🧠</div>
                  <h4>增强反思Agent</h4>
                  <p>自我改进的AI系统，从预测错误中持续学习</p>
                </div>
                <div class="highlight-card">
                  <div class="highlight-icon">📊</div>
                  <h4>多源数据融合</h4>
                  <p>整合财务数据、新闻情感和市场指标</p>
                </div>
                <div class="highlight-card">
                  <div class="highlight-icon">⚡</div>
                  <h4>实时分析</h4>
                  <p>持续监控和更新市场预测</p>
                </div>
                <div class="highlight-card">
                  <div class="highlight-icon">🔍</div>
                  <h4>可解释AI</h4>
                  <p>透明的推理过程支持投资决策</p>
                </div>
              </div>
              
              <div class="results-showcase">
                <div class="result-card">
                  <div class="result-number">87%</div>
                  <div class="result-label">预测准确率</div>
                  <div class="result-trend">↗️ +34%</div>
                </div>
                <div class="result-card">
                  <div class="result-number">15min</div>
                  <div class="result-label">实时更新频率</div>
                  <div class="result-trend">⚡ 超快</div>
                </div>
                <div class="result-card">
                  <div class="result-number">500+</div>
                  <div class="result-label">分析企业数量</div>
                  <div class="result-trend">📈 规模化</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- 黑客松信息 -->
        <section class="glass-section hackthon-info">
          <div class="section-header">
            <h2 class="section-title">🚀 黑客松概览</h2>
          </div>
          
          <div class="info-content">
            <p class="hackthon-description">
              AI Agent & LangGraph黑客松汇聚了来自全球的创新开发者、研究人员和企业家，
              在48小时内探索AI智能体系统和图基语言模型的前沿可能性。参与者们夜以继日地工作，
              创造出突破人工智能边界的革命性解决方案。
            </p>
            
            <div class="stats-grid">
              <div class="stat-card">
                <div class="stat-icon">👥</div>
                <div class="stat-number">150+</div>
                <div class="stat-label">参赛选手</div>
              </div>
              <div class="stat-card">
                <div class="stat-icon">🏢</div>
                <div class="stat-number">35</div>
                <div class="stat-label">参赛团队</div>
              </div>
              <div class="stat-card">
                <div class="stat-icon">⏰</div>
                <div class="stat-number">48</div>
                <div class="stat-label">编程小时</div>
              </div>
              <div class="stat-card">
                <div class="stat-icon">💰</div>
                <div class="stat-number">$50K</div>
                <div class="stat-label">奖金池</div>
              </div>
            </div>
            
            <div class="themes-section">
              <h3 class="themes-title">🎯 竞赛主题</h3>
              <div class="themes-grid">
                <div class="theme-card">
                  <div class="theme-icon">🤖</div>
                  <h4>智能Agent系统</h4>
                  <p>构建能够推理、规划和执行复杂任务的自主AI系统</p>
                </div>
                <div class="theme-card">
                  <div class="theme-icon">🔗</div>
                  <h4>LangGraph应用</h4>
                  <p>利用图基语言模型增强理解和生成能力</p>
                </div>
                <div class="theme-card">
                  <div class="theme-icon">💼</div>
                  <h4>企业级解决方案</h4>
                  <p>面向商业自动化和决策支持的实用AI应用</p>
                </div>
                <div class="theme-card">
                  <div class="theme-icon">🌐</div>
                  <h4>多Agent协作</h4>
                  <p>多个专业化智能体协同工作的协作AI系统</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  </FullPageLayout>
</template>

<script setup>
import FullPageLayout from './FullPageLayout.vue'

// 生成浮动元素样式
const getFloatingStyle = (index) => {
  const size = Math.random() * 30 + 20
  const left = Math.random() * 100
  const animationDelay = Math.random() * 10
  const animationDuration = Math.random() * 20 + 15
  
  return {
    fontSize: `${size}px`,
    left: `${left}%`,
    animationDelay: `${animationDelay}s`,
    animationDuration: `${animationDuration}s`
  }
}

// 随机科技图标
const getRandomTechIcon = () => {
  const icons = ['⚡', '🔬', '🧬', '🔭', '🛸', '🤖', '💻', '📡', '🔋', '⚙️', '🧠', '💾', '🔌', '📊', '🎯']
  return icons[Math.floor(Math.random() * icons.length)]
}
</script>

<style scoped>
.hackthon-page {
  position: relative;
  min-height: 100vh;
  overflow-x: hidden;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* 科技感动态背景 */
.tech-background {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533483 100%);
  z-index: -3;
}

.matrix-rain {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-image: 
    linear-gradient(90deg, transparent 98%, rgba(0, 255, 0, 0.03) 100%),
    linear-gradient(180deg, transparent 98%, rgba(0, 255, 0, 0.03) 100%);
  background-size: 50px 50px;
  animation: matrix-scroll 20s linear infinite;
  z-index: -2;
}

@keyframes matrix-scroll {
  0% { transform: translateY(0); }
  100% { transform: translateY(50px); }
}

.circuit-pattern {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-image: 
    radial-gradient(circle at 25% 25%, rgba(0, 255, 255, 0.1) 0%, transparent 50%),
    radial-gradient(circle at 75% 75%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
    radial-gradient(circle at 50% 50%, rgba(255, 255, 0, 0.05) 0%, transparent 50%);
  animation: circuit-pulse 8s ease-in-out infinite;
  z-index: -1;
}

@keyframes circuit-pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 0.7; }
}

.floating-elements {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: -1;
}

.floating-element {
  position: absolute;
  color: rgba(0, 255, 255, 0.6);
  animation: float-tech linear infinite;
  text-shadow: 0 0 10px currentColor;
}

@keyframes float-tech {
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
  padding: 8rem 2rem 6rem;
  text-align: center;
  color: white;
  z-index: 1;
}

.hero-content {
  max-width: 1000px;
  margin: 0 auto;
}

.main-title {
  font-size: 4.5rem;
  font-weight: 900;
  margin-bottom: 2rem;
  line-height: 1.1;
}

.title-tech {
  display: block;
  background: linear-gradient(45deg, #00ffff, #0080ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
}

.title-highlight {
  display: block;
  background: linear-gradient(45deg, #ff00ff, #ff0080);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: 0 0 30px rgba(255, 0, 255, 0.5);
  margin: 0.5rem 0;
}

.title-event {
  display: block;
  background: linear-gradient(45deg, #ffff00, #ff8000);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: 0 0 30px rgba(255, 255, 0, 0.5);
}

.hero-subtitle {
  font-size: 1.4rem;
  margin-bottom: 3rem;
  opacity: 0.9;
  font-weight: 300;
  color: #e0e0e0;
}

.hero-badges {
  display: flex;
  justify-content: center;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.badge {
  padding: 1rem 2rem;
  border-radius: 50px;
  font-weight: 600;
  font-size: 1.1rem;
  border: 2px solid;
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
  cursor: pointer;
}

.badge:hover {
  transform: translateY(-3px) scale(1.05);
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
}

.neon-blue {
  border-color: #00ffff;
  background: rgba(0, 255, 255, 0.1);
  color: #00ffff;
  box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
}

.neon-purple {
  border-color: #ff00ff;
  background: rgba(255, 0, 255, 0.1);
  color: #ff00ff;
  box-shadow: 0 0 20px rgba(255, 0, 255, 0.3);
}

.neon-green {
  border-color: #00ff00;
  background: rgba(0, 255, 0, 0.1);
  color: #00ff00;
  box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
}

.neon-orange {
  border-color: #ff8000;
  background: rgba(255, 128, 0, 0.1);
  color: #ff8000;
  box-shadow: 0 0 20px rgba(255, 128, 0, 0.3);
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
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(20px);
  border-radius: 25px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin-bottom: 4rem;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  transition: transform 0.3s ease;
}

.glass-section:hover {
  transform: translateY(-5px);
}

.section-header {
  background: rgba(255, 255, 255, 0.05);
  padding: 2.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  text-align: center;
  position: relative;
}

.winner-crown {
  font-size: 4rem;
  margin-bottom: 1rem;
  animation: crown-glow 3s ease-in-out infinite alternate;
}

@keyframes crown-glow {
  0% { transform: scale(1); filter: drop-shadow(0 0 10px gold); }
  100% { transform: scale(1.1); filter: drop-shadow(0 0 20px gold); }
}

.section-title {
  font-size: 2.8rem;
  font-weight: 700;
  color: white;
  margin: 0;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}

/* 获奖项目展示 */
.winner-section {
  border: 2px solid rgba(255, 215, 0, 0.3);
  box-shadow: 0 0 40px rgba(255, 215, 0, 0.2);
}

.project-title {
  font-size: 2.2rem;
  font-weight: 700;
  color: #ffd700;
  text-align: center;
  margin: 2rem 0;
  line-height: 1.3;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}

.project-showcase {
  padding: 2rem;
}

.project-intro {
  font-size: 1.3rem;
  color: rgba(255, 255, 255, 0.9);
  text-align: center;
  margin-bottom: 3rem;
  line-height: 1.7;
  background: rgba(255, 255, 255, 0.05);
  padding: 2rem;
  border-radius: 15px;
}

.tech-highlights {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
}

.highlight-card {
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 2rem;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
}

.highlight-card:hover {
  transform: translateY(-10px);
  background: rgba(255, 255, 255, 0.12);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

.highlight-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  display: block;
}

.highlight-card h4 {
  font-size: 1.4rem;
  font-weight: 600;
  color: white;
  margin-bottom: 1rem;
}

.highlight-card p {
  color: rgba(255, 255, 255, 0.8);
  line-height: 1.6;
  margin: 0;
}

.results-showcase {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 2rem;
}

.result-card {
  background: linear-gradient(135deg, rgba(0, 255, 255, 0.1), rgba(255, 0, 255, 0.1));
  backdrop-filter: blur(15px);
  border-radius: 20px;
  padding: 2rem;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
}

.result-card:hover {
  transform: translateY(-5px) scale(1.05);
  box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
}

.result-number {
  font-size: 2.5rem;
  font-weight: 900;
  color: #00ffff;
  margin-bottom: 0.5rem;
  text-shadow: 0 0 10px currentColor;
}

.result-label {
  font-size: 1rem;
  color: rgba(255, 255, 255, 0.9);
  margin-bottom: 0.5rem;
}

.result-trend {
  font-size: 1.1rem;
  color: #00ff00;
  font-weight: 600;
}

/* 黑客松信息 */
.hackthon-description {
  font-size: 1.3rem;
  color: rgba(255, 255, 255, 0.9);
  text-align: center;
  margin-bottom: 3rem;
  line-height: 1.7;
  padding: 2rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 2rem;
  margin-bottom: 4rem;
}

.stat-card {
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(15px);
  border-radius: 20px;
  padding: 2rem;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-10px);
  background: rgba(255, 255, 255, 0.12);
}

.stat-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  display: block;
}

.stat-number {
  font-size: 2.5rem;
  font-weight: 900;
  color: #ff00ff;
  margin-bottom: 0.5rem;
  text-shadow: 0 0 10px currentColor;
}

.stat-label {
  font-size: 1.1rem;
  color: rgba(255, 255, 255, 0.9);
  font-weight: 500;
}

.themes-section {
  padding: 2rem;
}

.themes-title {
  font-size: 2rem;
  font-weight: 700;
  color: white;
  text-align: center;
  margin-bottom: 2rem;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}

.themes-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 2rem;
}

.theme-card {
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(15px);
  border-radius: 20px;
  padding: 2rem;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
}

.theme-card:hover {
  transform: translateY(-10px);
  background: rgba(255, 255, 255, 0.12);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

.theme-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  display: block;
}

.theme-card h4 {
  font-size: 1.4rem;
  font-weight: 600;
  color: white;
  margin-bottom: 1rem;
}

.theme-card p {
  color: rgba(255, 255, 255, 0.8);
  line-height: 1.6;
  margin: 0;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .main-title {
    font-size: 3rem;
  }
  
  .hero-section {
    padding: 6rem 1rem 4rem;
  }
  
  .content-container {
    padding: 2rem 1rem;
  }
  
  .tech-highlights,
  .results-showcase,
  .stats-grid,
  .themes-grid {
    grid-template-columns: 1fr;
  }
  
  .hero-badges {
    flex-direction: column;
    align-items: center;
  }
  
  .section-header {
    padding: 2rem 1rem;
  }
  
  .section-title {
    font-size: 2.2rem;
  }
  
  .project-showcase {
    padding: 1rem;
  }
}

@media (max-width: 480px) {
  .main-title {
    font-size: 2.2rem;
  }
  
  .hero-subtitle {
    font-size: 1.1rem;
  }
  
  .badge {
    padding: 0.8rem 1.5rem;
    font-size: 1rem;
  }
  
  .highlight-card,
  .result-card,
  .stat-card,
  .theme-card {
    padding: 1.5rem;
  }
  
  .project-title {
    font-size: 1.8rem;
  }
}
</style>