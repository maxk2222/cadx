import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import enCommon from "./locales/en/common.json";
import enNav from "./locales/en/nav.json";
import enWizard from "./locales/en/wizard.json";
import enProjects from "./locales/en/projects.json";
import enVersion from "./locales/en/version.json";
import enJobs from "./locales/en/jobs.json";
import enChat from "./locales/en/chat.json";
import enStats from "./locales/en/stats.json";
import enLibrary from "./locales/en/library.json";
import enSettings from "./locales/en/settings.json";
import enEditor from "./locales/en/editor.json";

import deCommon from "./locales/de/common.json";
import deNav from "./locales/de/nav.json";
import deWizard from "./locales/de/wizard.json";
import deProjects from "./locales/de/projects.json";
import deVersion from "./locales/de/version.json";
import deJobs from "./locales/de/jobs.json";
import deChat from "./locales/de/chat.json";
import deStats from "./locales/de/stats.json";
import deLibrary from "./locales/de/library.json";
import deSettings from "./locales/de/settings.json";
import deEditor from "./locales/de/editor.json";

import esCommon from "./locales/es/common.json";
import esNav from "./locales/es/nav.json";
import esWizard from "./locales/es/wizard.json";
import esProjects from "./locales/es/projects.json";
import esVersion from "./locales/es/version.json";
import esJobs from "./locales/es/jobs.json";
import esChat from "./locales/es/chat.json";
import esStats from "./locales/es/stats.json";
import esLibrary from "./locales/es/library.json";
import esSettings from "./locales/es/settings.json";
import esEditor from "./locales/es/editor.json";

import frCommon from "./locales/fr/common.json";
import frNav from "./locales/fr/nav.json";
import frWizard from "./locales/fr/wizard.json";
import frProjects from "./locales/fr/projects.json";
import frVersion from "./locales/fr/version.json";
import frJobs from "./locales/fr/jobs.json";
import frChat from "./locales/fr/chat.json";
import frStats from "./locales/fr/stats.json";
import frLibrary from "./locales/fr/library.json";
import frSettings from "./locales/fr/settings.json";
import frEditor from "./locales/fr/editor.json";

export const SUPPORTED = ["en", "de", "es", "fr"] as const;
export type Lang = (typeof SUPPORTED)[number];
export const LANG_LABEL: Record<Lang, string> = {
  en: "English",
  de: "Deutsch",
  es: "Español",
  fr: "Français",
};

const LANG_KEY = "cadx:lang";

export function detectLang(): Lang {
  try {
    const stored =
      typeof localStorage !== "undefined" ? localStorage.getItem(LANG_KEY) : null;
    if (stored && (SUPPORTED as readonly string[]).includes(stored))
      return stored as Lang;
    const nav = (
      typeof navigator !== "undefined" ? navigator.language : "en"
    ).slice(0, 2).toLowerCase();
    return (SUPPORTED as readonly string[]).includes(nav) ? (nav as Lang) : "en";
  } catch {
    return "en";
  }
}

export const resources = {
  en: {
    common: enCommon,
    nav: enNav,
    wizard: enWizard,
    projects: enProjects,
    version: enVersion,
    jobs: enJobs,
    chat: enChat,
    stats: enStats,
    library: enLibrary,
    settings: enSettings,
    editor: enEditor,
  },
  de: {
    common: deCommon,
    nav: deNav,
    wizard: deWizard,
    projects: deProjects,
    version: deVersion,
    jobs: deJobs,
    chat: deChat,
    stats: deStats,
    library: deLibrary,
    settings: deSettings,
    editor: deEditor,
  },
  es: {
    common: esCommon,
    nav: esNav,
    wizard: esWizard,
    projects: esProjects,
    version: esVersion,
    jobs: esJobs,
    chat: esChat,
    stats: esStats,
    library: esLibrary,
    settings: esSettings,
    editor: esEditor,
  },
  fr: {
    common: frCommon,
    nav: frNav,
    wizard: frWizard,
    projects: frProjects,
    version: frVersion,
    jobs: frJobs,
    chat: frChat,
    stats: frStats,
    library: frLibrary,
    settings: frSettings,
    editor: frEditor,
  },
} as const;

const initial = detectLang();
if (typeof document !== "undefined") document.documentElement.lang = initial;

void i18n.use(initReactI18next).init({
  resources,
  lng: initial,
  fallbackLng: "en",
  defaultNS: "common",
  interpolation: { escapeValue: false },
  returnNull: false,
});

i18n.on("languageChanged", (lng) => {
  try {
    if (typeof localStorage !== "undefined") localStorage.setItem(LANG_KEY, lng);
  } catch {
    /* private mode */
  }
  if (typeof document !== "undefined") document.documentElement.lang = lng;
});

export function setLanguage(lng: Lang) {
  void i18n.changeLanguage(lng);
}

export default i18n;
