import { Check, CheckCircle2, Copy, Info, Loader2, X, XCircle } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { fmtDur, sparkPoints, stepRows } from "../lib/liveStats";
import type { JobLive } from "../lib/types";

/* ---------- primitives ---------- */

export function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return <Loader2 className={`animate-spin ${className}`} />;
}

export function Card({
  className = "",
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className={`rounded-lg border border-zinc-800 bg-zinc-900/60 ${className}`}>
      {children}
    </div>
  );
}

export function SectionTitle({
  children,
  right,
}: {
  children: ReactNode;
  right?: ReactNode;
}) {
  return (
    <div className="mb-2 flex items-center justify-between gap-2">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
        {children}
      </h3>
      {right}
    </div>
  );
}

type Tone = "zinc" | "blue" | "green" | "amber" | "red" | "purple" | "teal";

const CHIP_TONES: Record<Tone, string> = {
  zinc: "border-zinc-700 bg-zinc-800 text-zinc-300",
  blue: "border-brand-800 bg-brand-950/60 text-brand-300",
  green: "border-emerald-800 bg-emerald-950/60 text-emerald-300",
  amber: "border-amber-800 bg-amber-950/60 text-amber-300",
  red: "border-red-800 bg-red-950/60 text-red-300",
  purple: "border-purple-800 bg-purple-950/60 text-purple-300",
  teal: "border-teal-800 bg-teal-950/60 text-teal-300",
};

export function Chip({
  tone = "zinc",
  className = "",
  title,
  children,
}: {
  tone?: Tone;
  className?: string;
  title?: string;
  children: ReactNode;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[11px] font-medium ${CHIP_TONES[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

/* ---------- buttons ---------- */

type BtnVariant = "primary" | "ghost" | "danger" | "success" | "subtle";

const BTN: Record<BtnVariant, string> = {
  primary: "bg-brand-600 text-white hover:bg-brand-500",
  success: "bg-emerald-600 text-white hover:bg-emerald-500",
  danger: "bg-red-700 text-white hover:bg-red-600",
  ghost: "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100",
  subtle: "bg-transparent text-brand-400 hover:bg-brand-950/40 hover:text-brand-300",
};

const BTN_SIZE: Record<"sm" | "md", string> = {
  sm: "px-2 py-1 text-xs",
  md: "px-3 py-1.5 text-sm",
};

export function Btn({
  variant = "primary",
  size = "md",
  className = "",
  busy,
  disabled,
  children,
  ...rest
}: {
  variant?: BtnVariant;
  size?: "sm" | "md";
  busy?: boolean;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 disabled:cursor-not-allowed disabled:opacity-50 ${BTN_SIZE[size]} ${BTN[variant]} ${className}`}
      disabled={disabled || busy}
      {...rest}
    >
      {busy && <Spinner className="h-3.5 w-3.5" />}
      {children}
    </button>
  );
}

export function IconBtn({
  title,
  className = "",
  danger,
  children,
  ...rest
}: {
  title: string;
  danger?: boolean;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      title={title}
      aria-label={title}
      className={`inline-flex h-7 w-7 items-center justify-center rounded-md text-zinc-400 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 disabled:opacity-40 ${
        danger ? "hover:bg-red-950 hover:text-red-300" : "hover:bg-zinc-800 hover:text-zinc-100"
      } ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

export function CopyBtn({ text, className = "" }: { text: string; className?: string }) {
  const [ok, setOk] = useState(false);
  return (
    <IconBtn
      title="copy to clipboard"
      className={className}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setOk(true);
          setTimeout(() => setOk(false), 1500);
        } catch {
          /* clipboard unavailable */
        }
      }}
    >
      {ok ? <Check className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4" />}
    </IconBtn>
  );
}

/* ---------- feedback ---------- */

export function EmptyState({
  icon,
  title,
  hint,
}: {
  icon?: ReactNode;
  title: string;
  hint?: string;
}) {
  return (
    <div className="flex flex-col items-center gap-1.5 rounded-lg border border-dashed border-zinc-800 px-4 py-8 text-center">
      {icon && <div className="text-zinc-500">{icon}</div>}
      <div className="text-sm text-zinc-400">{title}</div>
      {hint && <div className="text-xs text-zinc-500">{hint}</div>}
    </div>
  );
}

export function ProgressBar({
  value,
  className = "",
  tone = "blue",
}: {
  value: number;
  className?: string;
  tone?: "blue" | "green" | "amber" | "red";
}) {
  const colors = {
    blue: "bg-brand-500",
    green: "bg-emerald-500",
    amber: "bg-amber-500",
    red: "bg-red-500",
  };
  return (
    <div className={`h-1.5 w-full overflow-hidden rounded-full bg-zinc-800 ${className}`}>
      <div
        className={`h-full rounded-full transition-[width] duration-500 ${colors[tone]}`}
        style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }}
      />
    </div>
  );
}

/* ---------- overlay components ---------- */

export function Modal({
  open,
  onClose,
  title,
  children,
  wide,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  wide?: boolean;
}) {
  const dlgRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const prev = document.activeElement as HTMLElement | null;
    dlgRef.current?.focus();
    const h = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      // minimal focus trap — keep Tab cycling inside the dialog
      if (e.key !== "Tab" || !dlgRef.current) return;
      const els = dlgRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      const list = [...els].filter((el) => !el.hasAttribute("disabled"));
      if (!list.length) return;
      const first = list[0];
      const last = list[list.length - 1];
      if (!dlgRef.current.contains(document.activeElement)) {
        e.preventDefault();
        (e.shiftKey ? last : first).focus();
      } else if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", h);
    return () => {
      window.removeEventListener("keydown", h);
      prev?.focus?.();
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        ref={dlgRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        className={`w-full ${wide ? "max-w-3xl" : "max-w-md"} rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl outline-none`}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
          <h2 className="text-sm font-semibold">{title}</h2>
          <IconBtn title="close" onClick={onClose}>
            <X className="h-4 w-4" />
          </IconBtn>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}

export function ConfirmModal({
  open,
  onClose,
  onConfirm,
  title,
  body,
  confirmLabel = "Delete",
  busy,
}: {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  body: ReactNode;
  confirmLabel?: string;
  busy?: boolean;
}) {
  return (
    <Modal open={open} onClose={onClose} title={title}>
      <div className="mb-4 text-sm text-zinc-400">{body}</div>
      <div className="flex justify-end gap-2">
        <Btn variant="ghost" onClick={onClose}>
          Cancel
        </Btn>
        <Btn variant="danger" onClick={onConfirm} busy={busy}>
          {confirmLabel}
        </Btn>
      </div>
    </Modal>
  );
}

export function Lightbox({ src, onClose }: { src: string | null; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!src) return;
    const prev = document.activeElement as HTMLElement | null;
    ref.current?.focus();
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => {
      window.removeEventListener("keydown", h);
      prev?.focus?.();
    };
  }, [src, onClose]);
  if (!src) return null;
  return (
    <div
      ref={ref}
      role="dialog"
      aria-modal="true"
      aria-label="image preview"
      tabIndex={-1}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-8 outline-none"
      onClick={onClose}
    >
      <img src={src} className="max-h-full max-w-full rounded shadow-2xl" alt="" />
    </div>
  );
}

/* ---------- tabs ---------- */

export function Tabs<T extends string>({
  tabs,
  active,
  onChange,
  className = "",
}: {
  tabs: { id: T; label: ReactNode }[];
  active: T;
  onChange: (t: T) => void;
  className?: string;
}) {
  return (
    <div
      role="tablist"
      className={`flex gap-0.5 rounded-lg border border-zinc-800 bg-zinc-900 p-1 ${className}`}
      onKeyDown={(e) => {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        e.preventDefault();
        const i = tabs.findIndex((t) => t.id === active);
        if (i < 0) return;
        const n = (i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
        onChange(tabs[n].id);
        e.currentTarget
          .querySelectorAll<HTMLElement>('[role="tab"]')
          [n]?.focus();
      }}
    >
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={active === t.id}
          tabIndex={active === t.id ? 0 : -1}
          onClick={() => onChange(t.id)}
          className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 ${
            active === t.id
              ? "bg-zinc-800 text-zinc-100 shadow-sm"
              : "text-zinc-500 hover:text-zinc-300"
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

/* ---------- form ---------- */

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="mb-3 block">
      <span className="text-sm text-zinc-400">{label}</span>
      <div className="mt-1">{children}</div>
      {hint && <span className="mt-0.5 block text-xs text-zinc-500">{hint}</span>}
    </label>
  );
}

export const inputCls =
  "w-full rounded-md border border-zinc-700 bg-zinc-800/80 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 outline-none transition-colors focus:border-brand-600 focus:ring-1 focus:ring-brand-600/40";

/* ---------- crumbs ---------- */

export function Crumbs({ items }: { items: ReactNode[] }) {
  return (
    <nav className="flex items-center gap-1.5 text-sm text-zinc-500">
      {items.map((it, i) => (
        <span key={i} className="flex items-center gap-1.5">
          {i > 0 && <span className="text-zinc-700">/</span>}
          {it}
        </span>
      ))}
    </nav>
  );
}

/* ---------- toasts ---------- */

type ToastTone = "success" | "error" | "info";
interface Toast {
  id: number;
  tone: ToastTone;
  text: string;
}

const ToastCtx = createContext<(text: string, tone?: ToastTone) => void>(() => {});

/** Push a transient notification. Usage: `const toast = useToast(); toast("saved", "success")` */
export function useToast() {
  return useContext(ToastCtx);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);

  const push = useCallback((text: string, tone: ToastTone = "info") => {
    const id = nextId.current++;
    setToasts((t) => [...t.slice(-4), { id, tone, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4500);
  }, []);

  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-80 flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`pointer-events-auto flex items-start gap-2 rounded-lg border px-3 py-2 text-sm shadow-xl backdrop-blur ${
              t.tone === "success"
                ? "border-emerald-800 bg-emerald-950/90 text-emerald-200"
                : t.tone === "error"
                  ? "border-red-800 bg-red-950/90 text-red-200"
                  : "border-zinc-700 bg-zinc-900/90 text-zinc-300"
            }`}
          >
            {t.tone === "success" ? (
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
            ) : t.tone === "error" ? (
              <XCircle className="mt-0.5 h-4 w-4 shrink-0" />
            ) : (
              <Info className="mt-0.5 h-4 w-4 shrink-0" />
            )}
            <span className="min-w-0 flex-1 break-words">{t.text}</span>
            <button
              className="shrink-0 text-current opacity-50 hover:opacity-100"
              aria-label="dismiss"
              onClick={() => setToasts((ts) => ts.filter((x) => x.id !== t.id))}
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

/* ---------- skeletons ---------- */

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-zinc-800 ${className}`} />;
}

/* ---------- misc ---------- */

export function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

export function useStickyBottom(dep: unknown) {
  const boxRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const onScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };
  useEffect(() => {
    // scroll the log container directly — scrollIntoView would also scroll the window
    if (stick.current && boxRef.current)
      boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [dep]);
  return { boxRef, onScroll, stick };
}

/** Numeric input that commits on blur/Enter and keeps the raw text while
 *  editing (typing "1." must not snap back to "1"). Shared by the editors
 *  and param panels — keeps behaviour identical everywhere. */
export function NumInput({
  value,
  onChange,
  step = 1,
  className = "",
}: {
  value: number;
  onChange: (n: number) => void;
  step?: number;
  className?: string;
}) {
  const [txt, setTxt] = useState(String(value));
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setTxt(String(value));
  }, [value, editing]);
  const commit = () => {
    setEditing(false);
    const n = Number(txt);
    if (Number.isFinite(n)) onChange(n);
    else setTxt(String(value));
  };
  return (
    <input
      type="number"
      step={step}
      className={`${inputCls} px-1.5 py-0.5 text-[11px] ${className}`}
      value={editing ? txt : String(value)}
      onFocus={() => {
        setEditing(true);
        setTxt(String(value));
      }}
      onChange={(e) => setTxt(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && commit()}
    />
  );
}

/* ---------- live stats ---------- */

/** Date.now() ticking every intervalMs while active; with active=false no
 *  interval runs and the last value is kept. */
export function useNow(intervalMs: number, active = true): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, active]);
  return now;
}

/** Min/max-normalized SVG sparkline with a dezent area fill and an optional
 *  dashed reference line (value given in data units). */
export function Sparkline({
  values,
  width = 80,
  height = 24,
  refLine,
  className = "",
}: {
  values: number[];
  width?: number;
  height?: number;
  refLine?: number;
  className?: string;
}) {
  const gid = useId();
  const pts = sparkPoints(values, width, height);
  if (!pts) return <svg width={width} height={height} className={className} aria-hidden="true" />;
  const min = Math.min(...values);
  const span = Math.max(...values) - min;
  const refY =
    refLine != null && span > 0 ? height - ((refLine - min) / span) * height : null;
  return (
    <svg width={width} height={height} className={className} aria-hidden="true">
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgb(96 165 250 / 0.25)" />
          <stop offset="100%" stopColor="rgb(96 165 250 / 0)" />
        </linearGradient>
      </defs>
      <polygon
        points={`${pts} ${width},${height} 0,${height}`}
        fill={`url(#${gid})`}
        stroke="none"
      />
      {refY != null && (
        <line
          x1={0}
          x2={width}
          y1={refY}
          y2={refY}
          stroke="rgb(82 82 91)"
          strokeWidth={1}
          strokeDasharray="3 2"
        />
      )}
      <polyline
        points={pts}
        fill="none"
        stroke="rgb(96 165 250)"
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Circular progress: zinc-800 track, brand-400→emerald-400 gradient arc,
 *  centered percentage (or custom label). */
export function ProgressRing({
  value,
  size = 48,
  label,
  className = "",
}: {
  value: number;
  size?: number;
  label?: string;
  className?: string;
}) {
  const gid = useId();
  const v = Math.min(1, Math.max(0, value));
  const sw = 4;
  const r = (size - sw) / 2;
  const c = 2 * Math.PI * r;
  return (
    <div
      className={`relative inline-flex items-center justify-center ${className}`}
      style={{ width: size, height: size }}
    >
      <svg width={size} height={size} className="-rotate-90">
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#60a5fa" />
            <stop offset="100%" stopColor="#34d399" />
          </linearGradient>
        </defs>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#27272a" strokeWidth={sw} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={`url(#${gid})`}
          strokeWidth={sw}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - v)}
          className="transition-[stroke-dashoffset] duration-500"
        />
      </svg>
      <span className="absolute font-mono text-xs text-zinc-200">
        {label ?? `${Math.round(v * 100)}%`}
      </span>
    </div>
  );
}

/** Horizontal flex-wrap pills of completed + running pipeline steps (ticks
 *  every second). Done: emerald check + duration below; active: pulsing dot,
 *  brand border + shimmer; compact drops the duration row. */
export function PipelineStepper({
  live,
  compact,
  className = "",
}: {
  live: JobLive;
  compact?: boolean;
  className?: string;
}) {
  const now = useNow(1000, true);
  const rows = stepRows(live, now);
  if (!rows.length) return null;
  return (
    <div className={`flex flex-wrap items-center gap-1 ${className}`}>
      {rows.map((r, i) => (
        <span
          key={`${r.step}-${i}`}
          className={`inline-flex flex-col rounded border px-1.5 py-0.5 text-[10px] font-medium ${
            r.active
              ? "shimmer border-brand-700 bg-brand-950/50 text-brand-300"
              : "border-zinc-800 bg-zinc-900 text-zinc-400"
          }`}
        >
          <span className="flex items-center gap-1">
            {r.active ? (
              <span className="pulse-dot inline-block h-1.5 w-1.5 rounded-full bg-brand-400" />
            ) : (
              <Check className="h-3 w-3 text-emerald-400" />
            )}
            {r.label}
          </span>
          {!compact && (
            <span className="pl-4 font-mono text-[9px] opacity-60">{fmtDur(r.dt_s)}</span>
          )}
        </span>
      ))}
    </div>
  );
}

/** Per-step duration bars (width ∝ dt_s/max); the active step's bar grows
 *  live. Shows the last maxRows steps. */
export function StepWaterfall({
  live,
  maxRows = 12,
  className = "",
}: {
  live: JobLive;
  maxRows?: number;
  className?: string;
}) {
  const now = useNow(1000, true);
  const rows = stepRows(live, now).slice(-maxRows);
  if (!rows.length) return null;
  const maxDt = Math.max(1, ...rows.map((r) => r.dt_s));
  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {rows.map((r, i) => (
        <div key={`${r.step}-${i}`} className="flex items-center gap-2">
          <span
            className={`w-20 shrink-0 truncate text-xs ${
              r.active ? "text-brand-300" : "text-zinc-400"
            }`}
            title={r.label}
          >
            {r.label}
          </span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-800/60">
            <div
              className={`h-full rounded-full transition-[width] duration-500 ${
                r.active ? "bg-brand-500/70" : "bg-zinc-600"
              }`}
              style={{ width: `${(r.dt_s / maxDt) * 100}%` }}
            />
          </div>
          <span className="w-10 shrink-0 text-right font-mono text-[10px] text-zinc-500">
            {fmtDur(r.dt_s)}
          </span>
        </div>
      ))}
    </div>
  );
}

const STAT_TONES = {
  default: "text-zinc-100",
  ok: "text-emerald-300",
  warn: "text-amber-300",
  error: "text-red-300",
} as const;

/** Compact metric tile: small uppercase label, mono value, optional hint. */
export function StatChip({
  label,
  value,
  hint,
  tone = "default",
  className = "",
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: keyof typeof STAT_TONES;
  className?: string;
}) {
  return (
    <div
      className={`min-w-0 rounded-md border border-zinc-800 bg-zinc-900/60 px-2 py-1.5 ${className}`}
    >
      <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">{label}</div>
      <div className={`truncate font-mono text-sm ${STAT_TONES[tone]}`}>{value}</div>
      {hint && <div className="truncate text-[10px] text-zinc-500">{hint}</div>}
    </div>
  );
}
