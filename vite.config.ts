import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Relative base so the build works under GitHub Pages' /gpu-visualizer/ path
export default defineConfig({
  base: './',
  plugins: [react()],
})
