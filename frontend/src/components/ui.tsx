import { Check, CheckCircle2, Copy, Info, Loader2, X, XCircle } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

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

type BtnVariant = "primary" | "ghost" | "danger" | "success";

const BTN: Record<BtnVariant, string> = {
  primary: "bg-brand-600 text-white hover:bg-brand-500",
  success: "bg-emerald-600 text-white hover:bg-emerald-500",
  danger: "bg-red-700 text-white hover:bg-red-600",
  ghost: "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100",
};

export function Btn({
  variant = "primary",
  className = "",
  busy,
  disabled,
  children,
  ...rest
}: {
  variant?: BtnVariant;
  busy?: boolean;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${BTN[variant]} ${className}`}
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
      className={`inline-flex h-7 w-7 items-center justify-center rounded-md text-zinc-400 transition-colors disabled:opacity-40 ${
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
      {icon && <div className="text-zinc-600">{icon}</div>}
      <div className="text-sm text-zinc-400">{title}</div>
      {hint && <div className="text-xs text-zinc-600">{hint}</div>}
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
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className={`w-full ${wide ? "max-w-3xl" : "max-w-md"} rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl`}
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
  useEffect(() => {
    if (!src) return;
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [src, onClose]);
  if (!src) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-8"
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
    <div className={`flex gap-0.5 rounded-lg border border-zinc-800 bg-zinc-900 p-1 ${className}`}>
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
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
      {hint && <span className="mt-0.5 block text-xs text-zinc-600">{hint}</span>}
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
  return { boxRef, onScroll };
}
