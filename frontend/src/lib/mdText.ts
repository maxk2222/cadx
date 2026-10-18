/**
 * Splits assistant text into interleaved text/code segments on ``` fences.
 * Unlike a naive split() it keeps text AFTER the last block, handles several
 * blocks and degrades an unclosed fence into a trailing code segment.
 */
export type MdSegment =
  | { type: "text"; content: string }
  | { type: "code"; lang?: string; content: string };

const FENCE = /```(\w*)\n?([\s\S]*?)(?:```|$)/g;

export function splitMd(text: string): MdSegment[] {
  const out: MdSegment[] = [];
  let last = 0;
  for (const m of text.matchAll(FENCE)) {
    const pre = text.slice(last, m.index);
    if (pre) out.push({ type: "text", content: pre });
    out.push({
      type: "code",
      lang: m[1] || undefined,
      content: m[2].replace(/\n$/, ""),
    });
    last = m.index + m[0].length;
  }
  const tail = text.slice(last);
  if (tail) out.push({ type: "text", content: tail });
  return out;
}
