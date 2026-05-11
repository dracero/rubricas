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
      host: '0.0.0.0',  // Listen on all interfaces
      allowedHosts: [
        'localhost',
        '127.0.0.1',
        'asistiag.udc.es',
        '.udc.es',  // Allow all subdomains of udc.es
      ],
      proxy: {
        '/auth': 'http://localhost:8080',
        '/api': 'http://localhost:8080',
        '/uploads': 'http://localhost:8080',
      },
      watch: {
        ignored: ['**/.venv/**', '**/.git/**', '**/__pycache__/**']
      }
    },
  };
})
