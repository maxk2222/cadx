import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * i18n quality gate:
 *  1. every namespace exists in every supported language
 *  2. key sets are identical across languages (no drifting de/es)
 *  3. every `t("ns:key")` used in src resolves to a real en key
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const LOCALES = join(HERE, "..", "i18n", "locales");
const SRC = join(HERE, "..");
const REF = "en";
const LANGS = readdirSync(LOCALES, { withFileTypes: true })
  .filter((e) => e.isDirectory())
  .map((e) => e.name)
  .sort();

type Tree = Record<string, unknown>;

function load(lang: string, ns: string): Tree {
  return JSON.parse(readFileSync(join(LOCALES, lang, `${ns}.json`), "utf8"));
}

function keySet(tree: Tree, prefix = ""): Set<string> {
  const out = new Set<string>();
  for (const [k, v] of Object.entries(tree)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object") {
      for (const s of keySet(v as Tree, key)) out.add(s);
    } else {
      out.add(key);
    }
  }
  return out;
}

function* walk(dir: string): Generator<string> {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) {
      if (!e.name.startsWith(".") && e.name !== "node_modules") yield* walk(p);
    } else if (/\.tsx?$/.test(e.name) && !e.name.endsWith(".test.ts")) {
      yield p;
    }
  }
}

const namespaces = readdirSync(join(LOCALES, REF))
  .filter((f) => f.endsWith(".json"))
  .map((f) => f.replace(/\.json$/, ""));

describe("i18n locale parity", () => {
  it("all namespaces exist in all languages", () => {
    for (const lang of LANGS) {
      const files = readdirSync(join(LOCALES, lang))
        .filter((f) => f.endsWith(".json"))
        .map((f) => f.replace(/\.json$/, ""));
      expect(files.sort(), `missing namespaces in ${lang}`).toEqual(
        [...namespaces].sort()
      );
    }
  });

  for (const lang of LANGS.filter((l) => l !== REF)) {
    it(`key set of ${lang} matches ${REF}`, () => {
      for (const ns of namespaces) {
        const enKeys = keySet(load(REF, ns));
        const lngKeys = keySet(load(lang, ns));
        const missing = [...enKeys].filter((k) => !lngKeys.has(k));
        const extra = [...lngKeys].filter((k) => !enKeys.has(k));
        expect(
          missing,
          `${lang}/${ns} missing keys: ${missing.join(", ")}`
        ).toEqual([]);
        expect(extra, `${lang}/${ns} extra keys: ${extra.join(", ")}`).toEqual(
          []
        );
      }
    });
  }

  it("every t() key in src exists in en", () => {
    const enKeys = new Map<string, Set<string>>();
    for (const ns of namespaces) enKeys.set(ns, keySet(load("en", ns)));
    const bad: string[] = [];
    const re = /\bt\(\s*["'`]((?:[a-z]+:)?[a-zA-Z0-9_.]+)["'`]/g;
    // useTranslation("ns") / useTranslation(["a","b"]) binds bare t() keys —
    // the first namespace in the array is the default
    const nsRe = /useTranslation\(\s*(?:\[\s*["']([a-z]+)["']|["']([a-z]+)["'])/;
    for (const file of walk(SRC)) {
      const src = readFileSync(file, "utf8");
      const bound = nsRe.exec(src);
      const defaultNs = bound ? (bound[1] ?? bound[2]) : "common";
      for (const m of src.matchAll(re)) {
        const raw = m[1];
        const hasNs = raw.includes(":");
        const ns = hasNs ? raw.split(":")[0] : defaultNs;
        const key = hasNs ? raw.slice(ns.length + 1) : raw;
        // plural forms: t("x", {count}) resolves to x_zero/one/two/few/many/other
        const keys = enKeys.get(ns);
        const hit =
          keys?.has(key) ||
          ["_zero", "_one", "_two", "_few", "_many", "_other"].some((suf) =>
            keys?.has(key + suf)
          );
        if (!hit) bad.push(`${file.split("/src/")[1]}: t("${raw}")`);
      }
    }
    expect(bad).toEqual([]);
  });
});
