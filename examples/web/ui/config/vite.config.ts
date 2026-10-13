import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react-swc';
import path from 'path';
import {fileURLToPath} from 'url';

const currentDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  base: '/',
  plugins: [react()],
  resolve: {
    alias: {
      '@magenta-rt/common': path.resolve(currentDir, '../../../common/react_ui/index.ts'),
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {target: 'http://127.0.0.1:8000', ws: true},
    },
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: {
    outDir: path.resolve(currentDir, '../dist'),
    emptyOutDir: true,
  },
  publicDir: path.resolve(currentDir, '../public'),
});
