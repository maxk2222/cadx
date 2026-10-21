/** Pure helpers for the UpdateCard — kept logic-only so they're unit-testable. */

import type { UpdateStatus } from "./types";

const PHASE_LABELS: Record<string, string> = {
  starting: "starting…",
  fetch: "fetching origin",
  check: "checking",
  pull: "pulling changes",
  reexec: "restarting updater",
  deps: "installing dependencies",
  restart: "restarting the stack",
  verify: "verifying health",
  rollback: "rolling back",
  done: "finished",
};

/** Human label for an update phase/state combination. */
export function updatePhaseLabel(s: Pick<UpdateStatus, "state" | "phase">): string {
  if (s.state === "running" || s.state === "checking" || s.state === "rolling-back") {
    return PHASE_LABELS[s.phase ?? ""] ?? s.phase ?? s.state;
  }
  if (s.state === "done") return "done";
  if (s.state === "failed") return "failed";
  return "idle";
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
