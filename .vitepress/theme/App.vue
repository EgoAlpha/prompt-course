<script setup>
import { useData } from 'vitepress'
import Layout from './base/Layout.vue'
import SymposiumLayout from './components/SymposiumLayout.vue'
import HackthonLayout from './components/HackthonLayout.vue'
import SymposiumLayoutV2 from './components/SymposiumLayoutV2.vue'
import HackthonLayoutV2 from './components/HackthonLayoutV2.vue'

const { page } = useData()

const layoutComponents = {
  SymposiumLayout,
  HackthonLayout,
  SymposiumLayoutV2,
  HackthonLayoutV2
}

const currentLayout = computed(() => {
  const layoutName = page.value.frontmatter?.layout
  return layoutComponents[layoutName] || Layout
})
</script>

<template lang="pug">
component(:is="currentLayout")
//  template(#sidebar-nav-after)
//    teleport(to="body"): client-only: Suspense: ChatBtnLoader
</template>

<style scoped lang="scss">

</style>
