# React Frontend Setup

The frontend has been rebuilt with React! 🎉

## What's New

- ⚛️ **React 18** with modern hooks
- 🎨 **Framer Motion** for smooth animations
- 🚀 **Vite** for fast development and builds
- 💎 **Modern Design** with glassmorphism effects
- 📱 **Fully Responsive**

## Development

To work on the frontend:

```bash
cd frontend
npm install
npm run dev
```

This starts a development server on `http://localhost:3000` with hot reload.

## Building for Production

The frontend is automatically built to `src/api/static/` for FastAPI to serve:

```bash
cd frontend
npm run build
```

After building, restart the FastAPI server to serve the new frontend.

## Features

✨ Animated gradient background with floating orbs
🎨 Glassmorphism cards with backdrop blur
💫 Smooth Framer Motion animations
📊 Beautiful result display with animated confidence bars
🎯 Interactive example cards
📱 Fully responsive design
🔔 Real-time API status indicator
⚡ Fast and optimized with Vite

## Project Structure

```
frontend/
├── src/
│   ├── components/     # React components
│   ├── styles/         # CSS files
│   ├── utils/          # API utilities
│   ├── App.jsx         # Main app component
│   └── main.jsx        # Entry point
├── package.json
└── vite.config.js
```

The built files go to `src/api/static/` for FastAPI to serve.

