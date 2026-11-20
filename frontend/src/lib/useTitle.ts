import { useEffect } from "react";

/** Sets document.title to `<text> — KIARA` (or bare "KIARA" when empty)
 *  and restores the base title on unmount. */
export function useTitle(text?: string | null) {
  useEffect(() => {
    document.title = text ? `${text} — KIARA` : "KIARA";
    return () => {
      document.title = "KIARA";
    };
  }, [text]);
}
