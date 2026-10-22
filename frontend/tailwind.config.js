import colors from "tailwindcss/colors";

/** Semantic theme tokens — values live as CSS vars in index.css :root so the
 *  whole app re-themes by editing one block (and a future light theme is a
 *  token swap, not a code sweep). */
const v = (name) => `rgb(var(${name}) / <alpha-value>)`;

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: v("--c-bg"), // app background
        panel: v("--c-panel"), // cards, sidebar
        panel2: v("--c-panel-2"), // inputs, hover, active tabs
        edge: v("--c-edge"), // borders
        edge2: v("--c-edge-2"), // stronger borders / hover
        fg: v("--c-fg"), // primary text
        muted: v("--c-muted"), // secondary text
        faint: v("--c-faint"), // placeholder / dimmest text
        // CADX accent — single point to re-theme the whole app
        brand: colors.blue,
        accent: colors.blue,
      },
    },
  },
  plugins: [],
};
