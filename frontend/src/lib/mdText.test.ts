import { describe, expect, it } from "vitest";
import { splitMd } from "./mdText";

describe("splitMd", () => {
  it("returns inline-only text as a single segment", () => {
    expect(splitMd("hello **world**")).toEqual([
      { type: "text", content: "hello **world**" },
    ]);
    expect(splitMd("")).toEqual([]);
  });

  it("splits a fenced block and preserves the language", () => {
    expect(splitMd("intro\n```py\nx = 1\n```\ntail")).toEqual([
      { type: "text", content: "intro\n" },
      { type: "code", lang: "py", content: "x = 1" },
      { type: "text", content: "\ntail" },
    ]);
  });

  it("preserves text after the last code block", () => {
    const segs = splitMd("```\ncode\n```\nand now the explanation");
    expect(segs).toEqual([
      { type: "code", lang: undefined, content: "code" },
      { type: "text", content: "\nand now the explanation" },
    ]);
  });

  it("handles multiple blocks with interleaved text", () => {
    const segs = splitMd(
      "a\n```py\np\n```\nb\n```js\nj\n```\nc",
    );
    expect(segs).toEqual([
      { type: "text", content: "a\n" },
      { type: "code", lang: "py", content: "p" },
      { type: "text", content: "\nb\n" },
      { type: "code", lang: "js", content: "j" },
      { type: "text", content: "\nc" },
    ]);
  });

  it("treats an unclosed fence as a trailing code segment", () => {
    expect(splitMd("look:\n```py\npartial = True")).toEqual([
      { type: "text", content: "look:\n" },
      { type: "code", lang: "py", content: "partial = True" },
    ]);
  });
});
