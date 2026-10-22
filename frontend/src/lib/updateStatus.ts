/** Pure helpers for the UpdateCard — kept logic-only so they're unit-testable. */

import i18n from "../i18n";
import type { UpdateStatus } from "./types";

const PHASES = new Set([
  "starting",
  "fetch",
  "check",
  "pull",
  "reexec",
  "deps",
  "restart",
  "verify",
  "rollback",
  "done",
]);

/** Human label for an update phase/state combination. */
export function updatePhaseLabel(s: Pick<UpdateStatus, "state" | "phase">): string {
  if (s.state === "running" || s.state === "checking" || s.state === "rolling-back") {
    return s.phase && PHASES.has(s.phase)
      ? i18n.t(`settings:upd_phase_${s.phase}`)
      : (s.phase ?? s.state);
  }
  if (s.state === "done") return i18n.t("settings:upd_phase_state_done");
  if (s.state === "failed") return i18n.t("settings:upd_phase_state_failed");
  return i18n.t("settings:upd_phase_state_idle");
}

/** True while a status poll should keep running (update in flight). */
export function isUpdateActive(s: Pick<UpdateStatus, "state"> | null | undefined): boolean {
  return !!s && (s.state === "running" || s.state === "checking" || s.state === "rolling-back");
}

/** Short badge text for the nav item — behind-count or null when nothing pending. */
export function behindBadge(behind: number | null | undefined): number | undefined {
  return behind && behind > 0 ? behind : undefined;
}

/** Tail of the update log for display — already normalized, newest last. */
export function logTail(lines: string[] | undefined, max = 12): string {
  return (lines ?? []).slice(-max).join("\n");
}
