<script setup async>
const ready = ref(false)
if (!window.__MINI_CHAT_MOUNT) {
// 1. 读取trust/mini.html, 取得url地址
  const { data } = await useFetch('./trust/mini.html').get().text()
  const parser = new DOMParser()
  const dom = parser.parseFromString(data.value.split('\n').splice(1).join(''), 'text/html')
  const selectMap = (selectors, attrName) => {
    const result = []
    dom.querySelectorAll(selectors).forEach(item => {
      result.push(item.getAttribute(attrName))
    })
    return result
  }
  const jsList = [
    ...selectMap('[type=module]', 'src'),
    ...selectMap('[rel=modulepreload]', 'href'),
  ]

// 2. 读取js地址，并运行
  for (let i = 0; i < jsList.length; i++) {
    const url = jsList[i]
    const { load } = useScriptTag(url, () => {}, {
      noModule: false,
      attrs: {
        crossorigin: '',
        type: 'module',
      },
    })
    await load()
  }
  const cssList = [
    ...selectMap('[rel=stylesheet]', 'href'),
  ]
  for (let i = 0; i < cssList.length; i++) {
    const url = cssList[i]
    const { data } = await useFetch(url).get().text()
    const el = document.createElement('style')
    el.type = 'text/css'
    el.textContent = data.value
    document.head.appendChild(el)
  }
}
ready.value = true
</script>

<template lang="pug">
chat-btn(v-if="ready")
</template>

<style scoped lang="scss">

</style>
