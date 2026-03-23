/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  
  server: {
    port: 5173,
    strictPort: true,
  },

  build: {
    target: 'esnext',
    minify: 'esbuild',
    rollupOptions: {
      output: {
        manualChunks: {
          'three': ['three'],
          'spark': ['@sparkjsdev/spark'],
          'rapier': ['@dimforge/rapier3d-compat'],
          'react-vendor': ['react', 'react-dom'],
        },
      },
    },
  },

  test: {
    globals: true,
    environment: 'happy-dom',
    include: ['src/**/*.{test,spec}.ts', 'src/**/*.{test,spec}.tsx'],
    exclude: ['node_modules', 'dist'],
    testTimeout: 120_000,
    coverage: {
      reporter: ['text', 'json', 'html'],
    },
  },
})
