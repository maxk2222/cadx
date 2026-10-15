import type { GenMode } from "./types";

/** Auto project name from the part description: first ~6 words, trimmed,
 *  capitalized. Falls back to a dated placeholder. */
export function deriveProjectName(prompt: string): string {
  const words = prompt.trim().replace(/\s+/g, " ").split(" ").filter(Boolean);
  if (!words.length) {
    const d = new Date();
    return `Project ${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
      d.getDate()
    ).padStart(2, "0")}`;
  }
  let name = words.slice(0, 6).join(" ");
  if (name.length > 48) name = name.slice(0, 48).trimEnd();
  return name.charAt(0).toUpperCase() + name.slice(1);
}

const TOPOPT_RE = /\b(topopt|topology|optimiz|lightweight|leichtbau|stiffen|load.?case|tragf)/i;
const MESH_RE = /\b(organic|sculpt|figur|statue|ornament|griff|handle|grip|character|dragon|toy|artistic|freeform|skulptur|tier|animal)\b/i;
const CAD_RE = /\b(hole|bohr|thread|gewinde|bracket|plate|platte|mount|halter|enclosure|gehäuse|din\s?\d|iso\s?\d|m\d{1,2}\b|flange|flansch|spacer|adapter|countersink|senk)/i;

/** Heuristic mode suggestion shown as a hint in the wizard — the user always
 *  keeps the final say ("auto" defers to the planner). */
export function suggestMode(prompt: string, hasImages: boolean, meshOk: boolean): GenMode {
  const p = prompt.trim();
  if (TOPOPT_RE.test(p)) return "topopt";
  if (hasImages && meshOk) {
    if (MESH_RE.test(p)) return "mesh";
    if (CAD_RE.test(p)) return "hybrid";
    return "hybrid";
  }
  if (CAD_RE.test(p) || !hasImages) return "cad";
  return "auto";
}
