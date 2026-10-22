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

import zhCommon from "./locales/zh/common.json";
import zhNav from "./locales/zh/nav.json";
import zhWizard from "./locales/zh/wizard.json";
import zhProjects from "./locales/zh/projects.json";
import zhVersion from "./locales/zh/version.json";
import zhJobs from "./locales/zh/jobs.json";
import zhChat from "./locales/zh/chat.json";
import zhStats from "./locales/zh/stats.json";
import zhLibrary from "./locales/zh/library.json";
import zhSettings from "./locales/zh/settings.json";
import zhEditor from "./locales/zh/editor.json";

import ptCommon from "./locales/pt/common.json";
import ptNav from "./locales/pt/nav.json";
import ptWizard from "./locales/pt/wizard.json";
import ptProjects from "./locales/pt/projects.json";
import ptVersion from "./locales/pt/version.json";
import ptJobs from "./locales/pt/jobs.json";
import ptChat from "./locales/pt/chat.json";
import ptStats from "./locales/pt/stats.json";
import ptLibrary from "./locales/pt/library.json";
import ptSettings from "./locales/pt/settings.json";
import ptEditor from "./locales/pt/editor.json";

import ruCommon from "./locales/ru/common.json";
import ruNav from "./locales/ru/nav.json";
import ruWizard from "./locales/ru/wizard.json";
import ruProjects from "./locales/ru/projects.json";
import ruVersion from "./locales/ru/version.json";
import ruJobs from "./locales/ru/jobs.json";
import ruChat from "./locales/ru/chat.json";
import ruStats from "./locales/ru/stats.json";
import ruLibrary from "./locales/ru/library.json";
import ruSettings from "./locales/ru/settings.json";
import ruEditor from "./locales/ru/editor.json";

import plCommon from "./locales/pl/common.json";
import plNav from "./locales/pl/nav.json";
import plWizard from "./locales/pl/wizard.json";
import plProjects from "./locales/pl/projects.json";
import plVersion from "./locales/pl/version.json";
import plJobs from "./locales/pl/jobs.json";
import plChat from "./locales/pl/chat.json";
import plStats from "./locales/pl/stats.json";
import plLibrary from "./locales/pl/library.json";
import plSettings from "./locales/pl/settings.json";
import plEditor from "./locales/pl/editor.json";

import jaCommon from "./locales/ja/common.json";
import jaNav from "./locales/ja/nav.json";
import jaWizard from "./locales/ja/wizard.json";
import jaProjects from "./locales/ja/projects.json";
import jaVersion from "./locales/ja/version.json";
import jaJobs from "./locales/ja/jobs.json";
import jaChat from "./locales/ja/chat.json";
import jaStats from "./locales/ja/stats.json";
import jaLibrary from "./locales/ja/library.json";
import jaSettings from "./locales/ja/settings.json";
import jaEditor from "./locales/ja/editor.json";

export const SUPPORTED = ["en", "de", "es", "fr", "zh", "pt", "ru", "pl", "ja"] as const;
export type Lang = (typeof SUPPORTED)[number];
export const LANG_LABEL: Record<Lang, string> = {
  en: "English",
  de: "Deutsch",
  es: "Español",
  fr: "Français",
  zh: "中文",
  pt: "Português",
  ru: "Русский",
  pl: "Polski",
  ja: "日本語",
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
  zh: {
    common: zhCommon,
    nav: zhNav,
    wizard: zhWizard,
    projects: zhProjects,
    version: zhVersion,
    jobs: zhJobs,
    chat: zhChat,
    stats: zhStats,
    library: zhLibrary,
    settings: zhSettings,
    editor: zhEditor,
  },
  pt: {
    common: ptCommon,
    nav: ptNav,
    wizard: ptWizard,
    projects: ptProjects,
    version: ptVersion,
    jobs: ptJobs,
    chat: ptChat,
    stats: ptStats,
    library: ptLibrary,
    settings: ptSettings,
    editor: ptEditor,
  },
  ru: {
    common: ruCommon,
    nav: ruNav,
    wizard: ruWizard,
    projects: ruProjects,
    version: ruVersion,
    jobs: ruJobs,
    chat: ruChat,
    stats: ruStats,
    library: ruLibrary,
    settings: ruSettings,
    editor: ruEditor,
  },
  pl: {
    common: plCommon,
    nav: plNav,
    wizard: plWizard,
    projects: plProjects,
    version: plVersion,
    jobs: plJobs,
    chat: plChat,
    stats: plStats,
    library: plLibrary,
    settings: plSettings,
    editor: plEditor,
  },
  ja: {
    common: jaCommon,
    nav: jaNav,
    wizard: jaWizard,
    projects: jaProjects,
    version: jaVersion,
    jobs: jaJobs,
    chat: jaChat,
    stats: jaStats,
    library: jaLibrary,
    settings: jaSettings,
    editor: jaEditor,
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
