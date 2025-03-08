import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/Hassane_Portfolio_With_Chatbot/',
  server: {
    port: 3000,
    open: true
  },
  css: {
    postcss: './postcss.config.js'
  }
})
