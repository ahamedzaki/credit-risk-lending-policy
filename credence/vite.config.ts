import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // GitHub Pages serves this repo's Pages site at /<repo-name>/, not the domain root —
  // asset URLs need that prefix baked in at build time. Routing itself doesn't care
  // (HashRouter), only the JS/CSS <script>/<link> src paths do.
  base: process.env.GITHUB_PAGES ? "/credit-risk-lending-policy/" : "/",
  server: { port: 5188, strictPort: false },
  build: { outDir: "dist", sourcemap: false },
});
