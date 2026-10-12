import { useEffect } from "react";

/** Sets document.title to `<text> — CADX` (or bare "CADX" when empty)
 *  and restores the base title on unmount. */
export function useTitle(text?: string | null) {
  useEffect(() => {
    document.title = text ? `${text} — CADX` : "CADX";
    return () => {
      document.title = "CADX";
    };
  }, [text]);
}
