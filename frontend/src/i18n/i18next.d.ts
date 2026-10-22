import "i18next";

// Deliberately no `resources` typing: strict key unions would force every
// component into per-namespace useTranslation() calls. The parity test
// (lib/i18nParity.test.ts) is the real correctness gate — it validates key
// existence across all locales and scans every t() call site.
declare module "i18next" {
  interface CustomTypeOptions {
    defaultNS: "common";
  }
}
