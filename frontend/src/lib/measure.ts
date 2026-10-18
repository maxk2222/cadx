/** Pure helpers for the pick-two-points measure tool (mm, model space). */

export type Pt3 = [number, number, number];

/** Euclidean distance between two surface points. */
export function distMm(a: Pt3, b: Pt3): number {
  return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
}

/** Append a picked point; once two are placed the next pick starts over
 *  (same behaviour as EditPage's measure mode). */
export function nextMeasurePts(pts: readonly Pt3[], p: Pt3): Pt3[] {
  return pts.length >= 2 ? [p] : [...pts, p];
}
