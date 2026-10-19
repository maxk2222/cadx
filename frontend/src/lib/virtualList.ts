import type { ChatMessage } from "./types";

/**
 * Dependency-free windowing for the chat message list. Rows keep their real
 * (variable) heights: unmeasured rows use `estimateMessageHeight`, rendered
 * rows report their offsetHeight via ResizeObserver, and two spacer divs
 * stand in for the out-of-window ranges.
 */

/** below this count the list renders plainly — windowing is skipped. */
export const VIRTUALIZE_MIN_ITEMS = 40;
/** row gap in px — must match the `gap-3` on the message column. */
export const ROW_GAP = 12;
/** render margin beyond both viewport edges (px). */
export const OVERSCAN_PX = 600;

/**
 * Rough row height (px, gap excluded) used until a row has been measured
 * once. Deliberately simple — measurements replace it, so it only ever
 * needs to be close enough to keep the scrollbar sane.
 */
export function estimateMessageHeight(m: ChatMessage): number {
  if (m.role === "system") return 24;
  // ~90 chars/line in the max-w-3xl column at text-sm, plus bubble padding
  let h = 26 + Math.max(1, Math.ceil((m.content?.length ?? 0) / 90)) * 20;
  h += (m.meta?.tool_log?.length ?? 0) * 28;
  h += (m.meta?.spawned?.length ?? 0) * 110;
  if (m.asset_ids?.length) h += 92;
  if (m.meta?.streaming) h += 12;
  return Math.min(Math.max(h, 44), 800);
}

export interface VirtualWindow {
  /** first rendered index */
  start: number;
  /** one past the last rendered index */
  end: number;
  /** px spacer before the window */
  topPad: number;
  /** px spacer after the window */
  bottomPad: number;
}

/**
 * Window over `slots` (per-row heights INCLUDING each row's trailing gap)
 * covering [scrollTop - overscan, scrollTop + viewportH + overscan].
 *
 * The spacers live inside the same flex container, so each still consumes
 * one `gap` of its own — topPad/bottomPad subtract it back so rendered rows
 * land on exactly the same offsets as in the full list.
 */
export function computeWindow(
  slots: number[],
  scrollTop: number,
  viewportH: number,
  overscan = OVERSCAN_PX,
  gap = ROW_GAP
): VirtualWindow {
  const n = slots.length;
  if (!n) return { start: 0, end: 0, topPad: 0, bottomPad: 0 };

  const lo = scrollTop - overscan;
  const hi = scrollTop + viewportH + overscan;
  let acc = 0;
  let start = 0;
  while (start < n && acc + slots[start] <= lo) {
    acc += slots[start];
    start++;
  }
  const topSlots = acc; // sum of slots[0..start)
  let end = start;
  while (end < n && acc <= hi) {
    acc += slots[end];
    end++;
  }
  let tail = 0;
  for (let i = end; i < n; i++) tail += slots[i];

  return {
    start,
    end,
    topPad: start > 0 ? Math.max(0, topSlots - gap) : 0,
    bottomPad: end < n ? Math.max(0, tail - gap) : 0,
  };
}

/**
 * Stand-in window for the very first render, before the scroll element is
 * measurable: enough trailing rows to cover a typical viewport plus
 * overscan. The real window replaces it on the first scroll/resize/measure
 * pass — chats open pinned to the bottom, so the tail is what shows.
 */
export function tailWindow(
  slots: number[],
  minRows = 20,
  px = 1800,
  gap = ROW_GAP
): VirtualWindow {
  const n = slots.length;
  let start = n;
  let acc = 0;
  while (start > 0 && (n - start < minRows || acc < px)) {
    start--;
    acc += slots[start];
  }
  let top = 0;
  for (let i = 0; i < start; i++) top += slots[i];
  return {
    start,
    end: n,
    topPad: start > 0 ? Math.max(0, top - gap) : 0,
    bottomPad: 0,
  };
}
