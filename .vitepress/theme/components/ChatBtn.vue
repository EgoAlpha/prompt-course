<script setup>
let handler = null
// 3. onMount()
onMounted(() => {
  handler = setInterval(() => {
    if (window.__MINI_CHAT_MOUNT) {
      window.__MINI_CHAT_MOUNT('#miniBtn', {
        textHtmlSelector: '#VPContent',
        // questions: ['卡罗拉外文名是什么'],
      })
      clearInterval(handler)
    }
  }, 500)
})
// 4. onUnMount()
onUnmounted(() => {
  window.__MINI_CHAT_UNMOUNT()
  handler && clearInterval(handler)
})
</script>

<template lang="pug">
div.mini.z-30(id="miniBtn")
</template>

<style scoped lang="scss">
.mini {
  position: fixed;
  bottom: 20px;
  right: 20px;
}
</style>
