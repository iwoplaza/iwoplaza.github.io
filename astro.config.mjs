// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';

import tailwindcss from '@tailwindcss/vite';
import typegpu from 'unplugin-typegpu/vite';

// https://astro.build/config
export default defineConfig({
  site: 'https://iwoplaza.dev',
  integrations: [mdx(), sitemap()],

  vite: {
    plugins: [tailwindcss(), typegpu({})],
  },
});