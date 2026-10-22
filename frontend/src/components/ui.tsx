import { Check, CheckCircle2, ChevronDown, Copy, EllipsisVertical, Ghost, Info, Loader2, X, XCircle } from "lucide-react";
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
import { useTranslation } from "react-i18next";
import i18n from "../i18n";
import { fmtDur, sparkPoints, stepRows } from "../lib/liveStats";
import type { JobLive } from "../lib/types";

/* ---------- primitives ---------- */

export function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return <Loader2 className={`animate-spin ${className}`} />;
}

export function Card({
  className = "",
  interactive,
  children,
}: {
  className?: string;
  /** hover lift + border brightening — for clickable cards */
  interactive?: boolean;
  children: ReactNode;
}) {
  return (
    <div className={`panel ${interactive ? "panel-hover" : ""} ${className}`}>
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
      <h3 className="text-xs font-semibold uppercase tracking-wider text-faint">
        {children}
      </h3>
      {right}
    </div>
  );
}

/** collapsible section — SectionTitle-styled header with a chevron toggle.
 *  Animates via the grid-rows 0fr→1fr trick (no height JS). `persistKey`
 *  remembers the open state in localStorage. */
export function Collapsible({
  title,
  right,
  count,
  defaultOpen = true,
  persistKey,
  className = "",
  children,
}: {
  title: ReactNode;
  right?: ReactNode;
  /** optional item count shown as a muted suffix */
  count?: number;
  defaultOpen?: boolean;
  /** localStorage key — persists the open/closed state across reloads */
  persistKey?: string;
  className?: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(() => {
    if (persistKey) {
      try {
        const s = localStorage.getItem(`collapse:${persistKey}`);
        if (s != null) return s === "1";
      } catch {
        /* private mode */
      }
    }
    return defaultOpen;
  });
  const toggle = () => {
    setOpen((o) => {
      const n = !o;
      if (persistKey) {
        try {
          localStorage.setItem(`collapse:${persistKey}`, n ? "1" : "0");
        } catch {
          /* private mode */
        }
      }
      return n;
    });
  };
  return (
    <div className={className}>
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="group mb-2 flex w-full items-center justify-between gap-2 text-left"
      >
        <h3 className="text-xs font-semibold uppercase tracking-wider text-faint transition-colors group-hover:text-muted">
          {title}
          {count != null && (
            <span className="ml-1.5 font-normal normal-case tracking-normal text-faint/70">
              {count}
            </span>
          )}
        </h3>
        <span className="flex items-center gap-2">
          {right}
          <ChevronDown
            className={`h-3.5 w-3.5 text-faint transition-transform duration-200 group-hover:text-muted ${
              open ? "" : "-rotate-90"
            }`}
          />
        </span>
      </button>
      <div
        className={`grid transition-[grid-template-rows] duration-200 ease-out ${
          open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
        }`}
      >
        <div className="min-h-0 overflow-hidden">{children}</div>
      </div>
    </div>
  );
}

/** Card-wrapped Collapsible — the inspector/panel section pattern. */
export function SectionCollapsible({
  title,
  right,
  count,
  defaultOpen = true,
  persistKey,
  className = "",
  pad = "p-3",
  children,
}: {
  title: ReactNode;
  right?: ReactNode;
  count?: number;
  defaultOpen?: boolean;
  persistKey?: string;
  className?: string;
  pad?: string;
  children: ReactNode;
}) {
  return (
    <Card className={`${pad} ${className}`}>
      <Collapsible
        title={title}
        right={right}
        count={count}
        defaultOpen={defaultOpen}
        persistKey={persistKey}
      >
        {children}
      </Collapsible>
    </Card>
  );
}

type Tone = "zinc" | "blue" | "green" | "amber" | "red" | "purple" | "teal";

const CHIP_TONES: Record<Tone, string> = {
  zinc: "border-edge2 bg-panel2 text-zinc-300",
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
  primary:
    "bg-gradient-to-b from-brand-600 to-brand-700 text-white shadow-sm shadow-brand-950/50 hover:from-brand-500 hover:to-brand-600",
  success:
    "bg-gradient-to-b from-emerald-600 to-emerald-700 text-white shadow-sm shadow-emerald-950/50 hover:from-emerald-500 hover:to-emerald-600",
  danger: "bg-red-700 text-white hover:bg-red-600",
  ghost: "bg-panel2 text-zinc-300 hover:bg-edge hover:text-zinc-100",
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
      className={`inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 active:scale-[.98] disabled:cursor-not-allowed disabled:opacity-50 disabled:active:scale-100 ${BTN_SIZE[size]} ${BTN[variant]} ${className}`}
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
      className={`inline-flex h-7 w-7 items-center justify-center rounded-md text-muted transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 disabled:opacity-40 ${
        danger ? "hover:bg-red-950 hover:text-red-300" : "hover:bg-panel2 hover:text-fg"
      } ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

/** compact model-transparency slider for the 3D viewers' toolbars —
 *  value 0.1–1; 1 = opaque */
export function OpacitySlider({
  value,
  onChange,
  className = "",
}: {
  value: number;
  onChange: (v: number) => void;
  className?: string;
}) {
  const { t } = useTranslation();
  return (
    <label
      title={t("common:ui.model_opacity_hint")}
      className={`flex items-center gap-1.5 rounded-md bg-panel/80 px-2 py-1 text-xs text-muted backdrop-blur ${className}`}
    >
      <Ghost className="h-3.5 w-3.5" />
      <input
        type="range"
        min={0.1}
        max={1}
        step={0.05}
        className="w-24 accent-brand-500"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="w-8 tabular-nums text-faint">{Math.round(value * 100)}%</span>
    </label>
  );
}

export function CopyBtn({ text, className = "" }: { text: string; className?: string }) {
  const { t } = useTranslation();
  const [ok, setOk] = useState(false);
  return (
    <IconBtn
      title={t("common:action.copy_clipboard")}
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
    <div className="flex flex-col items-center gap-1.5 rounded-lg border border-dashed border-edge px-4 py-8 text-center">
      {icon && <div className="text-faint">{icon}</div>}
      <div className="text-sm text-muted">{title}</div>
      {hint && <div className="text-xs text-faint">{hint}</div>}
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
    blue: "bg-gradient-to-r from-brand-600 to-brand-400",
    green: "bg-gradient-to-r from-emerald-600 to-emerald-400",
    amber: "bg-gradient-to-r from-amber-600 to-amber-400",
    red: "bg-gradient-to-r from-red-600 to-red-400",
  };
  return (
    <div className={`h-1.5 w-full overflow-hidden rounded-full bg-panel2 ${className}`}>
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
  const { t } = useTranslation();
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
      className="anim-fade-in fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        ref={dlgRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        className={`anim-scale-in w-full ${wide ? "max-w-3xl" : "max-w-md"} rounded-xl border border-edge2 bg-panel shadow-2xl shadow-black/50 outline-none`}
      >
        <div className="flex items-center justify-between border-b border-edge px-4 py-3">
          <h2 className="text-sm font-semibold">{title}</h2>
          <IconBtn title={t("common:action.close")} onClick={onClose}>
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
  confirmLabel,
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
  const { t } = useTranslation();
  return (
    <Modal open={open} onClose={onClose} title={title}>
      <div className="mb-4 text-sm text-muted">{body}</div>
      <div className="flex justify-end gap-2">
        <Btn variant="ghost" onClick={onClose}>
          {t("common:action.cancel")}
        </Btn>
        <Btn variant="danger" onClick={onConfirm} busy={busy}>
          {confirmLabel ?? t("common:action.delete")}
        </Btn>
      </div>
    </Modal>
  );
}

export function Lightbox({ src, onClose }: { src: string | null; onClose: () => void }) {
  const { t } = useTranslation();
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
      aria-label={t("common:dialog.image_preview")}
      tabIndex={-1}
      className="anim-fade-in fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-8 outline-none"
      onClick={onClose}
    >
      <img src={src} className="anim-scale-in max-h-full max-w-full rounded shadow-2xl" alt="" />
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
      className={`flex gap-0.5 rounded-lg border border-edge bg-panel p-1 ${className}`}
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
          className={`relative flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 ${
            active === t.id
              ? "bg-panel2 text-fg shadow-sm after:absolute after:inset-x-3 after:bottom-0.5 after:h-px after:rounded-full after:bg-brand-500/70"
              : "text-faint hover:text-muted"
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
      <span className="text-sm text-muted">{label}</span>
      <div className="mt-1">{children}</div>
      {hint && <span className="mt-0.5 block text-xs text-faint">{hint}</span>}
    </label>
  );
}

/** @deprecated alias for the `.input` css class — kept for existing imports */
export const inputCls = "input";

/* ---------- crumbs ---------- */

export function Crumbs({ items }: { items: ReactNode[] }) {
  return (
    <nav className="flex items-center gap-1.5 text-sm text-faint">
      {items.map((it, i) => (
        <span key={i} className="flex items-center gap-1.5">
          {i > 0 && <span className="text-edge2">/</span>}
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
            className={`anim-fade-in-up pointer-events-auto flex items-start gap-2 rounded-lg border px-3 py-2 text-sm shadow-xl shadow-black/40 backdrop-blur ${
              t.tone === "success"
                ? "border-emerald-800 bg-emerald-950/90 text-emerald-200"
                : t.tone === "error"
                  ? "border-red-800 bg-red-950/90 text-red-200"
                  : "border-edge2 bg-panel/90 text-zinc-300"
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
              aria-label={i18n.t("action.dismiss")}
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
  return <div className={`animate-pulse rounded bg-panel2 ${className}`} />;
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
  endMs,
  className = "",
}: {
  live: JobLive;
  compact?: boolean;
  /** job.finished_at — bounds the running step's clock on finished jobs */
  endMs?: number;
  className?: string;
}) {
  const now = useNow(1000, endMs == null);
  const rows = stepRows(live, now, endMs);
  if (!rows.length) return null;
  return (
    <div className={`flex flex-wrap items-center gap-1 ${className}`}>
      {rows.map((r, i) => (
        <span
          key={`${r.step}-${i}`}
          className={`inline-flex flex-col rounded border px-1.5 py-0.5 text-[10px] font-medium ${
            r.active
              ? "shimmer border-brand-700 bg-brand-950/50 text-brand-300"
              : "border-edge bg-panel text-muted"
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
  endMs,
  className = "",
}: {
  live: JobLive;
  maxRows?: number;
  /** job.finished_at — bounds the running step's clock on finished jobs */
  endMs?: number;
  className?: string;
}) {
  const now = useNow(1000, endMs == null);
  const rows = stepRows(live, now, endMs).slice(-maxRows);
  if (!rows.length) return null;
  const maxDt = Math.max(1, ...rows.map((r) => r.dt_s));
  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {rows.map((r, i) => (
        <div key={`${r.step}-${i}`} className="flex items-center gap-2">
          <span
            className={`w-20 shrink-0 truncate text-xs ${
              r.active ? "text-brand-300" : "text-muted"
            }`}
            title={r.label}
          >
            {r.label}
          </span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-panel2/60">
            <div
              className={`h-full rounded-full transition-[width] duration-500 ${
                r.active ? "bg-brand-500/70" : "bg-faint"
              }`}
              style={{ width: `${(r.dt_s / maxDt) * 100}%` }}
            />
          </div>
          <span className="w-10 shrink-0 text-right font-mono text-[10px] text-faint">
            {fmtDur(r.dt_s)}
          </span>
        </div>
      ))}
    </div>
  );
}

const STAT_TONES = {
  default: "text-fg",
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
      className={`min-w-0 rounded-md border border-edge bg-panel/60 px-2 py-1.5 ${className}`}
    >
      <div className="text-[10px] font-medium uppercase tracking-wider text-faint">{label}</div>
      <div className={`truncate font-mono text-sm ${STAT_TONES[tone]}`}>{value}</div>
      {hint && <div className="truncate text-[10px] text-faint">{hint}</div>}
    </div>
  );
}

/* ---------- page scaffolding ---------- */

/** Standard page wrapper — identical header rhythm on every page:
 *  crumbs → title + subtitle → actions, then content in a soft fade. */
export function PageShell({
  title,
  subtitle,
  crumbs,
  actions,
  maxW = "default",
  className = "",
  children,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  crumbs?: ReactNode;
  actions?: ReactNode;
  /** "list" → max-w-4xl · "default" → max-w-6xl · "wide" → max-w-7xl */
  maxW?: "list" | "default" | "wide";
  className?: string;
  children: ReactNode;
}) {
  const mw =
    maxW === "list" ? "max-w-4xl" : maxW === "wide" ? "max-w-7xl" : "max-w-6xl";
  return (
    <div className={`mx-auto w-full ${mw} p-6 lg:p-8 ${className}`}>
      <div className="anim-fade-in-up mb-5">
        {crumbs && <div className="mb-1.5">{crumbs}</div>}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <h1 className="text-xl font-semibold text-fg">{title}</h1>
          {actions && (
            <>
              <span className="flex-1" />
              <div className="flex flex-wrap items-center gap-2">{actions}</div>
            </>
          )}
        </div>
        {subtitle && <div className="mt-1 text-sm text-muted">{subtitle}</div>}
      </div>
      {children}
    </div>
  );
}

/** One-line metric strip — `it 5/8 · 3:42 · 45k tok` style. Lighter than a
 *  row of chips: muted text with middle-dot separators. */
export function StatRow({
  items,
  className = "",
}: {
  items: (ReactNode | null | undefined | false)[];
  className?: string;
}) {
  const shown = items.filter(Boolean);
  if (!shown.length) return null;
  return (
    <div
      className={`flex flex-wrap items-center gap-x-2 gap-y-1 text-xs tabular-nums text-muted ${className}`}
    >
      {shown.map((it, i) => (
        <span key={i} className="flex items-center gap-2">
          {i > 0 && <span className="text-faint/60">·</span>}
          {it}
        </span>
      ))}
    </div>
  );
}

/** ⋯ overflow dropdown for secondary actions (rename/delete/…). */
export function OverflowMenu({
  title,
  items,
  className = "",
}: {
  title: string;
  items: {
    icon?: ReactNode;
    label: ReactNode;
    danger?: boolean;
    disabled?: boolean;
    onClick: () => void;
  }[];
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const h = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", h);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", h);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);
  return (
    <div ref={ref} className={`relative ${className}`}>
      <IconBtn title={title} onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <EllipsisVertical className="h-4 w-4" />
      </IconBtn>
      {open && (
        <div className="anim-scale-in absolute right-0 top-8 z-40 min-w-44 origin-top-right rounded-lg border border-edge bg-panel p-1 shadow-xl shadow-black/40">
          {items.map((it, i) => (
            <button
              key={i}
              type="button"
              disabled={it.disabled}
              onClick={() => {
                setOpen(false);
                it.onClick();
              }}
              className={`flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm transition-colors disabled:opacity-40 ${
                it.danger
                  ? "text-red-400 hover:bg-red-950/50"
                  : "text-muted hover:bg-panel2 hover:text-fg"
              }`}
            >
              {it.icon}
              {it.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
