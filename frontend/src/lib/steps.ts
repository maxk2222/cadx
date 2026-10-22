import i18n from "../i18n";

/** Pipeline step ids → display labels (shared by JobLog, JobsPage, VersionTree).
 *  Translated via the jobs:step.* keys; unknown ids pass through. */
export function stepLabel(s: string | null | undefined): string {
  if (!s) return "";
  return i18n.t(`jobs:step.${s}`, { defaultValue: s });
}
