// Minimal flat config: TypeScript baseline + react-hooks systemic guard.
// Kept intentionally small — no stylistic/formatting rules.
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "node_modules"] },
  {
    files: ["src/**/*.{ts,tsx}"],
    extends: [tseslint.configs.recommended],
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // explicit, per the guard's intent even if plugin defaults shift
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
    },
  }
);
