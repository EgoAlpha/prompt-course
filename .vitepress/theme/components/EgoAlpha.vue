<script setup>
const props = defineProps({
  id: String,
})
const url = computed(() => {
  return `https://agentpilots.egoalpha.com/share/${props.id}`
})
</script>

<template>
  <iframe :src="url" />
  <p><a :href="url" target="_blank" rel="noreferrer">Open in new window</a></p>
</template>
