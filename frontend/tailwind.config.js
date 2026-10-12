import colors from "tailwindcss/colors";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // CADX accent — single point to re-theme the whole app
        brand: colors.blue,
      },
    },
  },
  plugins: [],
};
