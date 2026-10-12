import { useId } from "react";

/** CADX brand mark — two crossing extruded bars forming an "X".
 *  Mirrors frontend/public/logo.svg; inlined so it needs no asset request
 *  and can be sized via props. */
export default function Logo({
  size = 20,
  wordmark = false,
  className = "",
}: {
  size?: number;
  wordmark?: boolean;
  className?: string;
}) {
  const uid = useId();
  const mark = (
    <svg
      viewBox="0 0 64 64"
      width={size}
      height={size}
      className="shrink-0"
      aria-hidden="true"
      fill="none"
    >
      <defs>
        <linearGradient
          id={`${uid}-a`}
          x1="10"
          y1="10"
          x2="54"
          y2="54"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0" stopColor="#60a5fa" />
          <stop offset="1" stopColor="#2563eb" />
        </linearGradient>
        <linearGradient
          id={`${uid}-b`}
          x1="54"
          y1="10"
          x2="10"
          y2="54"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0" stopColor="#3b82f6" />
          <stop offset="1" stopColor="#1e40af" />
        </linearGradient>
      </defs>
      <path
        d="M45.8 9.8 54.2 18.2 18.2 54.2 9.8 45.8Z"
        transform="translate(0 5)"
        fill="#172554"
      />
      <path
        d="M9.8 18.2 18.2 9.8 54.2 45.8 45.8 54.2Z"
        transform="translate(0 5)"
        fill="#1e3a8a"
      />
      <path d="M45.8 9.8 54.2 18.2 18.2 54.2 9.8 45.8Z" fill={`url(#${uid}-b)`} />
      <path d="M9.8 18.2 18.2 9.8 54.2 45.8 45.8 54.2Z" fill={`url(#${uid}-a)`} />
    </svg>
  );
  if (!wordmark) return <span className={className}>{mark}</span>;
  // "CAD" + the X mark — the logo doubles as the final letter
  return (
    <span className={`flex items-center gap-1 ${className}`}>
      <span className="text-lg font-bold tracking-tight">CAD</span>
      {mark}
    </span>
  );
}
