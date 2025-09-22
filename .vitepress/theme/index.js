// import './default/styles/fonts.css';
// import { createPinia } from 'pinia'
import Theme from './base/without-fonts'
import './styles/main.scss'
import 'uno.css'
import App from './App.vue'
import EgoAlpha from './components/EgoAlpha.vue'

export default {
  ...Theme,
  // Layout: () => {
  //   return h(Theme.Layout, null, {
  //     // https://vitepress.dev/guide/extending-default-theme#layout-slots
  //     'nav-screen-content-after':()=>{
  //       return "AAA"
  //     }
  //   })
  // },
  Layout: App,
  enhanceApp({ app, router, siteData }) {
    app.component('EgoAlpha', EgoAlpha)
    // const pinia = createPinia()
    // app.use(pinia)
    // pinia.state.value = siteData.value.themeConfig.pinia || {}
  },
}
