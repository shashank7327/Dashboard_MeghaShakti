import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' so `npm run build` produces a dist/ that also works when opened
// straight off the filesystem, not only when served from a web root.
export default defineConfig({
  plugins: [react()],
  base: './',
  // PORT from the environment wins, 5180 is only the fallback.
  //
  // A hard-coded port makes the dev server refuse to start whenever anything
  // else already holds it -- another copy of this very dashboard, most
  // often. Nothing here needs a fixed port: it is a static React app reading
  // local JSON, with no OAuth callback, webhook or CORS origin pinned to a
  // number. strictPort stays false so Vite steps to the next free port
  // instead of failing outright.
  server: {
    port: Number(process.env.PORT) || 5180,
    strictPort: false,
    open: false,
  },
  build: {
    outDir: 'dist',
    // the district geometry alone is ~3 MB; the warning is expected, not a
    // problem to chase, so raise the threshold rather than leave a red herring
    chunkSizeWarningLimit: 8000,
  },
})
