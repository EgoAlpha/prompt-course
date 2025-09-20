<template>
  <div class="full-page-layout">
    <!-- 顶部导航栏 -->
    <VPNav>
      <template #nav-bar-title-before><slot name="nav-bar-title-before" /></template>
      <template #nav-bar-title-after><slot name="nav-bar-title-after" /></template>
      <template #nav-bar-content-before><slot name="nav-bar-content-before" /></template>
      <template #nav-bar-content-after><slot name="nav-bar-content-after" /></template>
    </VPNav>
    
    <!-- 主要内容区域，无侧边栏 -->
    <div class="full-page-content">
      <slot />
    </div>
  </div>
</template>

<script setup>
import VPNav from '../base/components/VPNav.vue'
</script>

<style scoped>
.full-page-layout {
  min-height: 100vh;
  background: var(--vp-c-bg);
}

.full-page-content {
  padding-top: var(--vp-nav-height);
  width: 100%;
  max-width: none;
  margin: 0;
  padding-left: 0;
  padding-right: 0;
}

/* 确保内容占满整个宽度 */
.full-page-content :deep(*) {
  max-width: none;
}
</style>