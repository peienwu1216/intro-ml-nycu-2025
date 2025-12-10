# aesthetic-web — Local development README

This repo is a small React + Vite app for evaluating image aesthetics. The instructions below help you get the project running locally, including Tailwind CSS setup and a fallback for when the backend is not available.

## Prerequisites
- Node.js (recommended >= 18)
- npm (bundled with Node) or yarn

Confirm installation:
```bash
node -v
npm -v
```

## Install dependencies
From the project root, run:
```bash
npm install
```

If you add new dev packages (for example `@tailwindcss/postcss`), re-run `npm install`.

## Tailwind CSS notes
This project uses Tailwind CSS. The repository includes `tailwind.config.cjs` and `postcss.config.cjs`, and the main CSS file `src/index.css` contains `@tailwind` directives.

If you see PostCSS/Tailwind errors during `npm run dev`, ensure the `@tailwindcss/postcss` package is installed (Tailwind v4+ requires the separate PostCSS plugin):
```bash
npm install -D @tailwindcss/postcss
```

Also make sure `src/index.css` begins with any `@import` rules for external fonts followed by:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

## Run the dev server
Start the Vite dev server:
```bash
npm run dev
```

Open the URL shown in the terminal (usually `http://localhost:5173`).

## Backend (optional)
The client expects an API at `http://localhost:5000/score` for image scoring. If the backend is not running, the frontend will automatically fall back to mock data after a 5s timeout (or if the network request fails). You can still test the UI without the backend.

If you do have a backend, point it to `http://localhost:5000` (or update the URL in `src/api/ScoreApi.js`).

## Useful commands
- `npm run dev` — start dev server
- `npm run build` — build for production
- `npm run preview` — preview production build locally

## Project structure (important files)
- `src/main.jsx` — app entry, imports `src/index.css`
- `src/index.css` — global CSS, Tailwind directives and custom helpers
- `src/components/Hero.jsx` — hero section / upload area
- `src/components/UploadArea.jsx` — file input and upload handling
- `src/api/ScoreApi.js` — uploads image and returns score (includes 5s timeout and mock fallback)
- `src/components/RadarScore.jsx` — radar chart visualization (uses `recharts`)

## Next steps / suggestions
- If you want fully consistent Tailwind spacing, consider replacing custom `.hero-spacing` values with Tailwind utilities or extend `tailwind.config.cjs` spacing scale.
- For responsive radar charts, replace fixed-size `RadarChart` with `ResponsiveContainer` from `recharts`.

If you want, I can add a short script or npm task to ensure the required optional plugin (`@tailwindcss/postcss`) is installed automatically.

---
If anything fails locally, paste the terminal error and I can help debug further.
