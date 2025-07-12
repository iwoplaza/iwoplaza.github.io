# Agent Guidelines for iwoplaza.dev

## Build/Development Commands
- `pnpm dev` - Start development server
- `pnpm build` - Build for production
- `pnpm preview` - Preview production build
- No test commands configured

## Code Style & Formatting
- Uses Biome for linting and formatting
- JavaScript: single quotes, 2-space indentation
- JSON: 2-space indentation
- Organize imports automatically enabled
- Remove unused imports (error level)

## Project Structure
- Astro-based static site with TypeScript
- Uses TailwindCSS v4 for styling
- Package manager: pnpm (required)
- Components in `src/components/`
- Pages in `src/pages/`
- Content in `src/content/`
- Shared logic in `src/lib/`

## Conventions
- Astro components use `.astro` extension
- Import statements: relative paths for local files
- CSS classes: TailwindCSS utility-first approach
- Dark mode: `dark:` prefix classes
- No test framework configured