/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Deliberately not FPL purple-and-green: you will have both open at once.
        base: "#0B0D10",
        surface: "#14171C",
        raised: "#1C2027",
        line: "#262B33",
        muted: "#8A93A0",
        pos: "#4ADE80",
        neg: "#F87171",
        warn: "#FBBF24",
      },
      fontFamily: {
        sans: ["Inter", "Geist", "system-ui", "-apple-system", "sans-serif"],
      },
      fontVariantNumeric: { tabular: "tabular-nums" },
    },
  },
  plugins: [],
};
