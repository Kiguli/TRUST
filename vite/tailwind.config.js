import colors from "tailwindcss/colors.js"

/** @type {import('tailwindcss').Config} */
export default {
  content: [
      './index.html',
      './js/**/*.{vue,js}',
  ],
  theme: {
    extend: {
        colors: {
            'primary': {
                'light': '#ADBED3',
                'DEFAULT': '#6B86AA',
            },
            'secondary': {
                'light': '#E2B051',
                'DEFAULT': '#A23C26',
            },
            gray: colors.slate
        }
    },
  },
  plugins: [],
}

