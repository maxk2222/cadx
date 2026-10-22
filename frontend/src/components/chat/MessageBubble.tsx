import { AlertTriangle, Check, Loader2, Wrench } from "lucide-react";
import { useTranslation } from "react-i18next";
import { memo, useState } from "react";
import { fmtDur } from "../../lib/liveStats";
import { splitMd } from "../../lib/mdText";
import { mergeToolLogs } from "../../lib/toolMerge";
import type { ToolEvent } from "../../lib/useChatEvents";
import type { Asset, ChatMessage, SpawnedRef } from "../../lib/types";
import { useNow } from "../ui";
import ChatJobCard from "./ChatJobCard";

/** Minimal markdown-ish renderer — fenced code blocks, inline code, bold,
 *  lists. Keeps assistant output readable without a markdown dependency. */
function MdText({ text }: { text: string }) {
  return (
    <div className="text-sm leading-relaxed">
      {splitMd(text).map((s, i) =>
        s.type === "code" ? (
          <pre
            key={i}
            className="my-2 overflow-x-auto rounded bg-zinc-950 p-2.5 text-xs text-zinc-200"
          >
            <code>{s.content}</code>
          </pre>
        ) : (
          <InlineMd key={i} text={s.content} />
        ),
      )}
    </div>
  );
}

function InlineMd({ text }: { text: string }) {
  const lines = text.split("\n");
  return (
    <>
      {lines.map((line, i) => {
        const bullet = /^\s*[-*•]\s+/.test(line);
        const content = bullet ? line.replace(/^\s*[-*•]\s+/, "") : line;
        const nodes = inlineNodes(content);
        return (
          <div key={i} className={bullet ? "flex gap-1.5" : "min-h-[1em]"}>
            {bullet && <span className="text-zinc-500">•</span>}
            <span>{nodes}</span>
          </div>
        );
      })}
    </>
  );
}

function inlineNodes(s: string): React.ReactNode[] {
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  return s.split(re).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**"))
      return (
        <strong key={i} className="font-semibold text-zinc-100">
          {part.slice(2, -2)}
        </strong>
      );
    if (part.startsWith("`") && part.endsWith("`"))
      return (
        <code key={i} className="rounded bg-zinc-800 px-1 text-xs text-brand-300">
          {part.slice(1, -1)}
        </code>
      );
    return <span key={i}>{part}</span>;
  });
}

function resultSummary(
  r: Record<string, unknown> | undefined,
  t: (k: string) => string,
): string {
  if (!r) return "";
  if (typeof r.error === "string") return r.error;
  if (r.status) return `→ ${r.status}`;
  if (r.ok === false) return t("tool_failed");
  return t("tool_ok");
}

function ToolChip({
  name,
  args,
  ok,
  result,
  done,
  t_ms,
  assetsById,
}: {
  name: string;
  args?: Record<string, unknown>;
  ok?: boolean;
  result?: Record<string, unknown>;
  done: boolean;
  t_ms?: number;
  assetsById: Map<string, Asset>;
}) {
  const { t } = useTranslation("chat");
  const [open, setOpen] = useState(false);
  // ticks only while the call is in flight — shows live elapsed next to spinner
  const now = useNow(1000, !done);
  const imgIds = (
    (result?.assets as { asset_id?: string }[] | undefined) ?? []
  )
    .map((a) => a.asset_id)
    .filter((x): x is string => !!x);
  const imgs = imgIds
    .map((id) => assetsById.get(id))
    .filter((a): a is Asset => !!a && !!a.url);
  const questions =
    (result?.questions as { question: string }[] | undefined) ?? [];

  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/60 text-xs">
      <button
        className="flex w-full items-center gap-1.5 px-2 py-1 text-left"
        onClick={() => setOpen((o) => !o)}
      >
        <Wrench className="h-3 w-3 text-zinc-500" />
        <span className="font-mono text-zinc-300">{name}</span>
        <span className="ml-auto flex items-center gap-1 text-zinc-500">
          {!done ? (
            <>
              <Loader2 className="h-3 w-3 animate-spin" />
              {t_ms != null && (
                <span className="font-mono text-[10px]">
                  {fmtDur((now - t_ms) / 1000)}
                </span>
              )}
            </>
          ) : ok === false ? (
            <AlertTriangle className="h-3 w-3 text-red-400" />
          ) : (
            <Check className="h-3 w-3 text-emerald-400" />
          )}
          {done && resultSummary(result, t)}
        </span>
      </button>
      {open && (
        <div className="border-t border-zinc-800 px-2 py-1.5">
          {args && Object.keys(args).length > 0 && (
            <pre className="mb-1 max-h-32 overflow-auto whitespace-pre-wrap text-zinc-500">
              {JSON.stringify(args, null, 1).slice(0, 1200)}
            </pre>
          )}
          {result && (
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap text-zinc-400">
              {JSON.stringify(result, null, 1).slice(0, 2000)}
            </pre>
          )}
        </div>
      )}
      {imgs.length > 0 && (
        <div className="flex gap-2 px-2 pb-2">
          {imgs.map((a) => (
            <a key={a.id} href={a.url} target="_blank" rel="noreferrer">
              <img
                src={a.url}
                alt={a.filename}
                className="h-24 rounded border border-zinc-700 object-cover"
              />
            </a>
          ))}
        </div>
      )}
      {questions.length > 0 && (
        <div className="mx-2 mb-2 rounded border border-amber-800/60 bg-amber-950/30 p-2 text-amber-200/90">
          <div className="mb-0.5 font-medium text-amber-300">
            {t("clarify_needed")}
          </div>
          <ul className="list-disc pl-4">
            {questions.map((q, i) => (
              <li key={i}>{q.question}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

const MessageBubble = memo(function MessageBubble({
  msg,
  assetsById,
  liveTools,
}: {
  msg: ChatMessage;
  assetsById: Map<string, Asset>;
  /** live tool events while this message is streaming */
  liveTools?: ToolEvent[];
}) {
  const { t } = useTranslation("chat");
  const streaming = !!msg.meta?.streaming;
  const att = (msg.asset_ids ?? [])
    .map((id) => assetsById.get(id))
    .filter((a): a is Asset => !!a);
  // persisted meta.tool_log (flushed per tool_result) + live events beyond
  // it, merged by call_idx — a reload mid-turn restores every chip and live
  // entries appended after reload never double up with persisted ones
  const log = mergeToolLogs(msg.meta?.tool_log, liveTools, streaming);
  const spawned: SpawnedRef[] = msg.meta?.spawned ?? [];

  if (msg.role === "user") {
    return (
      <div className="anim-fade-in-up flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-brand-900/60 px-3.5 py-2.5">
          {att.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {att.map((a) => (
                <a key={a.id} href={a.url} target="_blank" rel="noreferrer">
                  <img
                    src={a.url}
                    alt={a.filename}
                    className="h-20 rounded border border-brand-800 object-cover"
                  />
                </a>
              ))}
            </div>
          )}
          {msg.content && (
            <div className="whitespace-pre-wrap text-sm text-zinc-100">
              {msg.content}
            </div>
          )}
        </div>
      </div>
    );
  }

  if (msg.role === "system") {
    return (
      <div className="text-center text-xs italic text-zinc-500">{msg.content}</div>
    );
  }

  return (
    <div className="anim-fade-in-up flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-bl-sm border border-zinc-800 bg-zinc-900 px-3.5 py-2.5">
        {streaming && !msg.content && log.length === 0 && (
          <div className="flex items-center gap-2 text-sm text-zinc-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> {t("thinking")}
          </div>
        )}
        {log.length > 0 && (
          <div className="mb-2 flex flex-col gap-1">
            {log.map((t) => (
              <ToolChip key={t.seq} {...t} assetsById={assetsById} />
            ))}
          </div>
        )}
        {msg.content && <MdText text={msg.content} />}
        {streaming && msg.content && (
          <span className="anim-caret ml-1 inline-block h-3.5 w-1.5 bg-brand-400/70" />
        )}
        {spawned.length > 0 && (
          <div className="mt-2 flex flex-col gap-2">
            {spawned.map((s, i) => (
              <ChatJobCard key={s.job_id ?? s.version_id ?? i} spawned={s} />
            ))}
          </div>
        )}
        {msg.meta?.error && (
          <div className="mt-2 flex items-center gap-1.5 text-xs text-red-400">
            <AlertTriangle className="h-3 w-3" /> {msg.meta.error}
          </div>
        )}
        {msg.meta?.cancelled && (
          <div className="mt-1 text-xs text-zinc-500">{t("turn_cancelled")}</div>
        )}
      </div>
    </div>
  );
});

export default MessageBubble;
