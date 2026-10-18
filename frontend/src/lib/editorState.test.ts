import { describe, expect, it } from "vitest";
import { num, readStoredAnchors, readUiEditor, recordedMeshOps, savedOps } from "./editorState";

describe("readUiEditor", () => {
  it("returns the persisted draft blob", () => {
    const v = { options: { ui_editor: { load_cases: [] } } };
    expect(readUiEditor(v)).toEqual({ load_cases: [] });
  });

  it("returns null when no draft exists or options is malformed", () => {
    expect(readUiEditor({ options: {} })).toBeNull();
    expect(readUiEditor({ options: null })).toBeNull();
    expect(readUiEditor({ options: { ui_editor: "junk" } })).toBeNull();
    expect(readUiEditor(null)).toBeNull();
    expect(readUiEditor(undefined)).toBeNull();
  });
});

describe("savedOps", () => {
  const meshParent = {
    kind: "generate",
    resolved_mode: "mesh",
    params: { ops: [{ op: "rotate", args: { axis: "z", deg: 90 } }] },
  };

  it("restores a persisted ops draft", () => {
    const v = { ...meshParent, options: { ui_editor: { ops: [{ op: "mirror", args: {} }] } } };
    expect(savedOps(v)).toEqual([{ op: "mirror", args: {} }]);
  });

  it("honours a persisted EMPTY ops draft — the user dropped the stack", () => {
    const v = { ...meshParent, options: { ui_editor: { ops: [] } } };
    expect(savedOps(v)).toEqual([]);
  });

  it("falls back to recorded parent ops for mesh-mode versions", () => {
    expect(savedOps(meshParent)).toEqual(meshParent.params.ops);
    // a draft written by another editor (no ops key) must not shadow them
    const v = { ...meshParent, options: { ui_editor: { load_cases: [] } } };
    expect(savedOps(v)).toEqual(meshParent.params.ops);
  });

  it("mesh_edit versions never replay recorded ops (already baked)", () => {
    const v = { ...meshParent, kind: "mesh_edit" };
    expect(savedOps(v)).toEqual([]);
  });

  it("returns [] for non-mesh versions and missing input", () => {
    expect(savedOps({ kind: "generate", resolved_mode: "cad" })).toEqual([]);
    expect(savedOps(null)).toEqual([]);
  });
});

describe("readStoredAnchors", () => {
  it("reads analysis.anchors.holes", () => {
    const holes = [{ index: 1, kind: "hole", axis: "+z", x: 1, y: 2, dia: 4 }];
    expect(readStoredAnchors({ analysis: { anchors: { holes } } })).toEqual(holes);
  });

  it("returns [] when absent or malformed", () => {
    expect(readStoredAnchors({ analysis: null })).toEqual([]);
    expect(readStoredAnchors({ analysis: { anchors: {} } })).toEqual([]);
    expect(readStoredAnchors({ analysis: { anchors: { holes: "x" } } })).toEqual([]);
    expect(readStoredAnchors(undefined)).toEqual([]);
  });
});

describe("num", () => {
  it("parses finite numbers, falls back otherwise", () => {
    expect(num("4.5")).toBe(4.5);
    expect(num("-2", 9)).toBe(-2);
    expect(num("abc", 7)).toBe(7);
    expect(num("", 3)).toBe(0); // Number("") === 0
    expect(num("Infinity", 1)).toBe(1);
  });
});
