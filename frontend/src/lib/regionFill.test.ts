import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { positionsToWire } from "./meshArrays";
import { fillVertexColors, flattenPoints, runFillRequest } from "./regionFill";

const IDENTITY = new THREE.Matrix4().elements.slice();
const BASE: [number, number, number] = [0.5, 0.5, 0.5];

const attr = (pts: number[]) =>
  new THREE.BufferAttribute(new Float32Array(pts), 3);

describe("flattenPoints", () => {
  it("flattens xyz triples and drops malformed points", () => {
    const f = flattenPoints([
      [1, 2, 3],
      [4, 5],
      [6, 7, 8],
      [],
    ]);
    expect([...f]).toEqual([1, 2, 3, 6, 7, 8]);
  });
});

describe("fillVertexColors", () => {
  it("colours vertices within radius, keeps the base colour elsewhere", () => {
    const out = fillVertexColors(
      attr([0, 0, 0, 10, 0, 0]),
      IDENTITY,
      [{ points: new Float32Array([0.5, 0, 0]), radius: 2, color: [1, 0, 0] }],
      BASE
    );
    expect([...out]).toEqual([1, 0, 0, 0.5, 0.5, 0.5]);
  });

  it("first matching entry wins", () => {
    const out = fillVertexColors(
      attr([0, 0, 0]),
      IDENTITY,
      [
        { points: new Float32Array([0, 0, 0]), radius: 1, color: [1, 0, 0] },
        { points: new Float32Array([0, 0, 0]), radius: 1, color: [0, 1, 0] },
      ],
      BASE
    );
    expect([...out]).toEqual([1, 0, 0]);
  });

  it("applies the mesh→model matrix before distance checks", () => {
    const m = new THREE.Matrix4().makeTranslation(0, 100, 0);
    const out = fillVertexColors(
      attr([0, 0, 0, 10, 0, 0]),
      m.elements.slice(),
      [{ points: new Float32Array([0, 100, 0]), radius: 1, color: [0, 0, 1] }],
      BASE
    );
    expect([...out]).toEqual([0, 0, 1, 0.5, 0.5, 0.5]);
  });
});

describe("worker wire path", () => {
  it("positionsToWire → runFillRequest matches the direct call", () => {
    const a = attr([0, 0, 0, 5, 0, 0, 10, 0, 0]);
    const entries = [
      {
        points: new Float32Array([0, 0, 0]),
        radius: 1.5,
        color: [1, 0, 0] as [number, number, number],
      },
    ];
    const direct = fillVertexColors(a, IDENTITY, entries, BASE);
    const [viaWire] = runFillRequest({
      entries,
      meshes: [{ pos: positionsToWire(a)!, matrix: IDENTITY, base: BASE }],
    });
    expect([...viaWire!]).toEqual([...direct]);
  });
});
