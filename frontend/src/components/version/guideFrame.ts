import * as THREE from "three";

/** Shared solver→display frame for every guide 3D view (GuideScene and
 *  GuideEditor must show the model the same way Viewer3D does).
 *
 *  The solver world is Z-up mm; three.js renders Y-up. One -90° X turn
 *  presents a Z-up world upright — instance transforms stay in solver
 *  frame, so gizmo commits read back the same matrix.
 *
 *  Operator view: the solver's working face is +Y, which the tilt alone
 *  maps to -Z — facing AWAY from the default camera. An extra 180° turn
 *  about display-Y (outermost wrapper, applied after the tilt) brings the
 *  working face to +Z: the camera sees the front, and operator-left
 *  (+X world) lands screen-left like the manual figure.
 *
 *  Apply as nested groups — yaw OUTSIDE tilt — and with solverToDisplayVec
 *  for any point converted manually (camera focus, arrow directions). */
export const ZUP_DISPLAY_ROT: [number, number, number] = [-Math.PI / 2, 0, 0];
export const FRONT_YAW_ROT: [number, number, number] = [0, Math.PI, 0];

export const ZUP_DISPLAY_EULER = new THREE.Euler(-Math.PI / 2, 0, 0);
export const FRONT_YAW_EULER = new THREE.Euler(0, Math.PI, 0);

/** solver coords -> display coords, mirroring the nested wrappers
 *  (tilt first, then the front-yaw about display-Y). */
export function solverToDisplayVec(p: THREE.Vector3, zUp: boolean): THREE.Vector3 {
  if (!zUp) return p;
  return p.applyEuler(ZUP_DISPLAY_EULER).applyEuler(FRONT_YAW_EULER);
}
