import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '../', '');
  return {
    plugins: [react()],
    envDir: '../',
    define: {
      'import.meta.env.VITE_INSTITUCION': JSON.stringify(env.INSTITUCION || ''),
    },
    server: {
      proxy: {
        '/auth': 'http://localhost:8000',
        '/api': 'http://localhost:8000',
        '/uploads': 'http://localhost:8000',
      },
      watch: {
        ignored: ['**/.venv/**', '**/.git/**', '**/__pycache__/**']
      }
    },
  };
})
