#!/usr/bin/env node
// i18n parity gate — every locale must cover `en` completely.
//
// Locales live in src/i18n/locales/<lang>/<namespace>.json; keys are flattened
// to dot-paths per namespace ("ns:key.sub"). `en` is the source of truth.
//
// Two rules are enforced:
//   1. verbatim parity — every key present in en must exist in the locale.
//   2. plural coverage — for each en plural base (>=2 keys sharing a base and
//      ending in _zero/_one/_two/_few/_many/_other, e.g. `updates_one` +
//      `updates_other`), the locale must define every category its own
//      Intl.PluralRules produces (ru/pl need _few+_many, es/fr/pt need _many,
//      zh/ja need only _other).
//
// Keys a locale has that en lacks are only warnings (e.g. legit _few/_many).
//
// Usage: node scripts/check_i18n.mjs [--verbose]   (exit 1 on missing keys)

import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const LOCALES = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "i18n", "locales");
const SOURCE = "en";
const PLURAL_SUFFIX = /_(zero|one|two|few|many|other)$/;

const verbose = process.argv.includes("--verbose");

/** Flatten nested objects into { "a.b.c": leafValue } (arrays are leaves). */
function flatten(obj, prefix = "", out = {}) {
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === "object" && !Array.isArray(v)) {
      flatten(v, key, out);
    } else {
      out[key] = v;
    }
  }
  return out;
}

/** Load one locale dir → { "ns:key.path": value } across all its json files. */
function loadLocale(dir) {
  const flat = {};
  let files = [];
  try {
    files = readdirSync(dir).filter((f) => f.endsWith(".json"));
  } catch {
    return null; // locale dir missing entirely
  }
  for (const f of files) {
    const ns = f.replace(/\.json$/, "");
    let doc;
    try {
      doc = JSON.parse(readFileSync(join(dir, f), "utf8"));
    } catch (e) {
      console.error(`✗ ${dir}/${f}: invalid JSON — ${e.message}`);
      process.exitCode = 1;
      continue;
    }
    for (const [k, v] of Object.entries(flatten(doc))) flat[`${ns}:${k}`] = v;
  }
  return flat;
}

const langs = readdirSync(LOCALES, { withFileTypes: true })
  .filter((d) => d.isDirectory())
  .map((d) => d.name)
  .sort();

const ref = loadLocale(join(LOCALES, SOURCE));
if (!ref) {
  console.error(`✗ source locale '${SOURCE}' not found under ${LOCALES}`);
  process.exit(1);
}
const refKeys = new Set(Object.keys(ref));
const refNamespaces = new Set([...refKeys].map((k) => k.split(":", 1)[0]));

// plural bases in en: a suffix-matching leaf with >=2 suffixed siblings —
// single `_other` literals ("other" as a plain label) are NOT plural forms
const suffixOf = (k) => (k.split(":", 2)[1] || "").match(PLURAL_SUFFIX)?.[1] ?? null;
const baseOf = (k) => k.replace(PLURAL_SUFFIX, "");
const byBase = {};
for (const k of refKeys) {
  const cat = suffixOf(k);
  if (cat) (byBase[baseOf(k)] ||= []).push(cat);
}
const pluralBases = new Set(Object.keys(byBase).filter((b) => byBase[b].length >= 2));

let failed = false;
for (const lang of langs) {
  if (lang === SOURCE) continue;
  const flat = loadLocale(join(LOCALES, lang));
  if (!flat) {
    console.error(`✗ ${lang}: locale directory missing`);
    failed = true;
    continue;
  }
  const keys = new Set(Object.keys(flat));
  const cats = new Intl.PluralRules(lang).resolvedOptions().pluralCategories;

  const missingFiles = [...refNamespaces].filter(
    (ns) => ![...keys].some((k) => k.startsWith(`${ns}:`)),
  );

  const missing = new Set();
  for (const k of refKeys) {
    if (keys.has(k)) continue;
    const cat = suffixOf(k);
    // plural keys are covered by the category rule below — a missing *_one
    // in a language without a "one" category is not required by rule 2, but
    // verbatim parity (rule 1) still demands it; keep them required.
    missing.add(k);
  }
  // rule 2: every needed plural category per base (adds non-en categories
  // like _few/_many that rule 1 can never see)
  for (const b of pluralBases)
    for (const c of cats) {
      const k = `${b}_${c}`;
      if (!keys.has(k)) missing.add(k);
    }

  const extra = [...keys].filter((k) => !refKeys.has(k) && !pluralBases.has(baseOf(k)));
  const extraPlural = [...keys].filter((k) => !refKeys.has(k) && pluralBases.has(baseOf(k)));

  if (missingFiles.length)
    console.error(`✗ ${lang}: missing namespace file(s): ${missingFiles.join(", ")}`);
  for (const k of [...missing].sort())
    console.error(`✗ ${lang}: missing key ${k}   (en: ${JSON.stringify(ref[k] ?? "<plural form>").slice(0, 80)})`);
  if (verbose) {
    for (const k of extraPlural)
      console.warn(`~ ${lang}: locale-specific plural form ${k} (not in en)`);
    for (const k of extra) console.warn(`~ ${lang}: extra key not in en: ${k}`);
  }

  if (missingFiles.length || missing.size) {
    failed = true;
    console.error(
      `✗ ${lang}: ${missing.size} missing key(s)` +
        (missingFiles.length ? ` + ${missingFiles.length} file(s)` : ""),
    );
  } else {
    console.log(`✓ ${lang}: ${keys.size}/${refKeys.size} keys`);
  }
}

if (failed) {
  console.error("\ni18n parity check FAILED — add the missing keys listed above.");
  process.exit(1);
}
console.log("\ni18n parity check OK");
