import { describe, expect, it } from "vitest";
import {
  fmtDlBytes,
  fmtGb,
  fmtMemGb,
  modelState,
  modelStateLabel,
  modelStateTone,
  urlHost,
} from "./models";
import type { MeshModelInfo } from "./types";

function mk(p: Partial<MeshModelInfo>): MeshModelInfo {
  return {
    id: "m1",
    name: "Model 1",
    device: "cpu",
    backend: "auto",
    kind: "llm",
    supports_multiview: false,
    max_images: 0,
    license_note: "",
    loaded: false,
    available: false,
    last_error: null,
    default_size: null,
    max_count: null,
    supports_edit: false,
    requires_image: false,
    downloaded: false,
    downloading: false,
    size_gb: null,
    est_mem_mb: null,
    supports_vision: false,
    supports_tools: false,
    context_len: null,
    pinned: false,
    custom: false,
    ...p,
  };
}

describe("modelState / modelStateLabel", () => {
  it("reports not installed when weights are absent", () => {
    const m = mk({});
    expect(modelState(m)).toBe("not_installed");
    expect(modelStateLabel(m)).toBe("not installed");
    expect(modelStateTone(m)).toBe("zinc");
  });

  it("falls back to legacy `available` when `downloaded` is unset", () => {
    const m = mk({ available: true });
    // `downloaded` defaults false here — emulate the old payload shape
    (m as { downloaded?: boolean }).downloaded = undefined;
    expect(modelState(m)).toBe("installed");
  });

  it("reports installed / loaded / pinned", () => {
    expect(modelState(mk({ downloaded: true }))).toBe("installed");
    expect(modelState(mk({ downloaded: true, loaded: true }))).toBe("loaded");
    expect(modelState(mk({ downloaded: true, loaded: true, pinned: true }))).toBe(
      "pinned"
    );
  });

  it("downloading wins over everything else", () => {
    const m = mk({ downloaded: true, loaded: true, pinned: true, downloading: true });
    expect(modelState(m)).toBe("downloading");
    expect(modelStateLabel(m)).toBe("downloading");
    expect(modelStateTone(m)).toBe("blue");
  });
});

describe("formatting helpers", () => {
  it("fmtMemGb renders ~GB estimates", () => {
    expect(fmtMemGb(null)).toBe("—");
    expect(fmtMemGb(12288)).toBe("~12 GB");
    expect(fmtMemGb(4096)).toBe("~4.0 GB");
  });

  it("fmtGb renders GiB with one decimal", () => {
    expect(fmtGb(null)).toBe("—");
    expect(fmtGb(16.5)).toBe("16.5 GB");
  });

  it("fmtDlBytes scales download progress", () => {
    expect(fmtDlBytes(0)).toBe("0 KB");
    expect(fmtDlBytes(5_200_000)).toBe("5 MB");
    expect(fmtDlBytes(4_500_000_000)).toBe("4.5 GB");
  });
});

describe("urlHost", () => {
  it("extracts host with port", () => {
    expect(urlHost("https://api.openai.com/v1")).toBe("api.openai.com");
    expect(urlHost("http://localhost:8765")).toBe("localhost:8765");
  });

  it("returns empty for missing or invalid URLs", () => {
    expect(urlHost("")).toBe("");
    expect(urlHost(null)).toBe("");
    expect(urlHost("not a url")).toBe("");
  });
});
