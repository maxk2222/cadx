import { useState } from "react";

/**
 * <img> that never shows the browser's broken-image glyph: renders `fallback`
 * until `onLoad` fires, and falls back to it again on `onError`. Use for
 * artifact URLs that may exist in the DB before the file is flushed.
 */
export default function SafeImg({
  src,
  alt,
  className,
  fallback = null,
  loading,
}: {
  src: string | null | undefined;
  alt: string;
  className?: string;
  fallback?: React.ReactNode;
  loading?: "lazy" | "eager";
}) {
  const [failed, setFailed] = useState(false);
  if (!src || failed) return <>{fallback}</>;
  return (
    <img
      src={src}
      alt={alt}
      className={className}
      loading={loading}
      onError={() => setFailed(true)}
    />
  );
}
