const COLORS: Record<string, string> = {
  queued: "bg-zinc-600",
  running: "bg-blue-600 animate-pulse",
  waiting_for_input: "bg-amber-600",
  succeeded: "bg-emerald-600",
  needs_review: "bg-amber-700",
  failed: "bg-red-600",
  cancelled: "bg-zinc-700",
};

export default function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${COLORS[status] ?? "bg-zinc-600"}`}>
      {status.replaceAll("_", " ")}
    </span>
  );
}
