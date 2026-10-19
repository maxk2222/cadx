import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, errMsg } from "../lib/api";
import { fmtSecs } from "../lib/format";
import {
  fmtDlBytes,
  fmtGb,
  fmtMemGb,
  modelStateLabel,
  modelStateTone,
} from "../lib/models";
import type { MeshModelInfo } from "../lib/types";
import {
  Btn,
  Card,
  Chip,
  ProgressBar,
  SectionTitle,
  inputCls,
  useToast,
} from "./ui";

const KIND_GROUPS: { kind: MeshModelInfo["kind"]; label: string }[] = [
  { kind: "llm", label: "LLM" },
  { kind: "mesh", label: "Mesh" },
  { kind: "image", label: "Image" },
];

/** Download state cell — polls the transfer status while the model reports
 *  `downloading`, otherwise renders the plain state chip. */
function StateCell({ m }: { m: MeshModelInfo }) {
  const st = useQuery({
    queryKey: ["meshModelDownload", m.id],
    queryFn: () => api.meshModelDownloadStatus(m.id),
    enabled: m.downloading,
    refetchInterval: 1500,
    retry: false,
  });
  if (!m.downloading)
    return <Chip tone={modelStateTone(m)}>{modelStateLabel(m)}</Chip>;
  const d = st.data;
  const pct =
    d && d.bytes_total ? Math.min(1, d.bytes_done / d.bytes_total) : null;
  return (
    <div className="w-28">
      <ProgressBar
        value={pct ?? 0.02}
        className={pct == null ? "animate-pulse" : ""}
      />
      <div className="mt-0.5 text-[10px] leading-tight text-zinc-500">
        {d?.status === "error"
          ? `failed: ${d.error ?? "unknown"}`
          : d
            ? `${fmtDlBytes(d.bytes_done)}${d.bytes_total ? ` / ${fmtDlBytes(d.bytes_total)}` : ""}`
            : "downloading…"}
      </div>
    </div>
  );
}

/** "Local models" settings tab — grouped model table with download
 *  management, a custom-HF-repo register form and the live sidecar queue. */
export default function ModelsTab({ models }: { models: MeshModelInfo[] }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [repo, setRepo] = useState("");
  const [backend, setBackend] = useState("auto");
  const [vision, setVision] = useState(false);

  // the tab is only mounted while visible — poll the queue at 5s
  const queue = useQuery({
    queryKey: ["meshQueue"],
    queryFn: api.meshQueue,
    refetchInterval: () =>
      typeof document !== "undefined" && document.hidden ? false : 5000,
    retry: false,
  });

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["meshModels"] });
    qc.invalidateQueries({ queryKey: ["meshHealth"] });
    qc.invalidateQueries({ queryKey: ["imageModels"] });
    qc.invalidateQueries({ queryKey: ["meshQueue"] });
  };
  const onErr = (e: unknown) => toast(errMsg(e), "error");

  const download = useMutation({
    mutationFn: api.meshModelDownload,
    onSuccess: invalidate,
    onError: onErr,
  });
  const cancelDownload = useMutation({
    mutationFn: api.meshModelDownloadCancel,
    onSuccess: invalidate,
    onError: onErr,
  });
  const deleteWeights = useMutation({
    mutationFn: api.meshModelDeleteWeights,
    onSuccess: (d) => {
      if (d.error) toast(d.error, "error");
      invalidate();
    },
    onError: onErr,
  });
  const unregister = useMutation({
    mutationFn: api.meshModelUnregister,
    onSuccess: invalidate,
    onError: onErr,
  });
  const register = useMutation({
    mutationFn: api.meshModelRegister,
    onSuccess: () => {
      toast("model registered", "success");
      setRepo("");
      invalidate();
    },
    onError: onErr,
  });

  const busy =
    download.isPending ||
    cancelDownload.isPending ||
    deleteWeights.isPending ||
    unregister.isPending;

  const flagChips = (m: MeshModelInfo) => (
    <div className="flex flex-wrap gap-1">
      {m.supports_vision && <Chip tone="blue">vision</Chip>}
      {m.supports_tools && <Chip tone="zinc">tools</Chip>}
      {m.supports_edit && <Chip tone="zinc">edit</Chip>}
      {m.supports_multiview && <Chip tone="zinc">multiview</Chip>}
    </div>
  );

  const row = (m: MeshModelInfo) => (
    <tr key={m.id} className="border-t border-zinc-800/60 align-top">
      <td className="py-2 pr-2">
        <div className="font-medium text-zinc-200">{m.name || m.id}</div>
        <div className="text-[10px] text-zinc-500">
          {m.id}
          {m.custom ? " · custom" : ""}
        </div>
        {m.license_note ? (
          <div className="text-[10px] text-zinc-500">license: {m.license_note}</div>
        ) : null}
        {m.last_error ? (
          <div className="text-[10px] text-red-400">{m.last_error}</div>
        ) : null}
      </td>
      <td className="py-2 pr-2 text-zinc-400" title={`${m.backend} · ${m.device}`}>
        {m.device}
      </td>
      <td className="py-2 pr-2 text-zinc-400">{fmtMemGb(m.est_mem_mb)}</td>
      <td className="py-2 pr-2 text-zinc-400">{fmtGb(m.size_gb)}</td>
      <td className="py-2 pr-2">
        <StateCell m={m} />
      </td>
      <td className="py-2 pr-2">{flagChips(m)}</td>
      <td className="py-2">
        <div className="flex flex-wrap justify-end gap-1">
          {!m.downloaded && (
            <Btn
              size="sm"
              variant="ghost"
              disabled={busy}
              onClick={() => download.mutate(m.id)}
            >
              download
            </Btn>
          )}
          {m.downloading && (
            <Btn
              size="sm"
              variant="ghost"
              disabled={busy}
              onClick={() => cancelDownload.mutate(m.id)}
            >
              cancel
            </Btn>
          )}
          {m.downloaded && !m.loaded && (
            <Btn
              size="sm"
              variant="ghost"
              disabled={busy}
              onClick={() => deleteWeights.mutate(m.id)}
            >
              delete weights
            </Btn>
          )}
          {m.custom && (
            <Btn
              size="sm"
              variant="ghost"
              disabled={busy}
              onClick={() => unregister.mutate(m.id)}
            >
              remove
            </Btn>
          )}
        </div>
      </td>
    </tr>
  );

  const registerForm = (
    <div className="mb-3 rounded-md border border-zinc-800 bg-zinc-900/40 p-2">
      <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
        register custom model
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <input
          className={`${inputCls} w-48 px-2 py-1 text-xs`}
          placeholder="org/name (Hugging Face repo)"
          value={repo}
          onChange={(e) => setRepo(e.target.value)}
        />
        <select
          className={`${inputCls} w-32 px-2 py-1 text-xs`}
          value={backend}
          onChange={(e) => setBackend(e.target.value)}
          title="serving backend"
        >
          <option value="auto">auto</option>
          <option value="mlx-lm">mlx-lm</option>
          <option value="llama.cpp">llama.cpp</option>
        </select>
        <label className="flex items-center gap-1.5 text-xs text-zinc-400">
          <input
            type="checkbox"
            checked={vision}
            onChange={(e) => setVision(e.target.checked)}
          />
          vision-capable
        </label>
        <Btn
          size="sm"
          disabled={!repo.trim() || register.isPending}
          busy={register.isPending}
          onClick={() =>
            register.mutate({
              repo: repo.trim(),
              kind: "llm",
              backend: backend === "auto" ? undefined : backend,
              vision: vision || undefined,
            })
          }
        >
          register
        </Btn>
      </div>
    </div>
  );

  const q = queue.data;

  return (
    <div className={q ? "grid gap-4 xl:grid-cols-[minmax(0,1fr)_20rem]" : ""}>
      <Card className="p-4">
        <SectionTitle>Local models</SectionTitle>
        {KIND_GROUPS.map((g) => {
          const rows = models.filter((m) => m.kind === g.kind);
          return (
            <div key={g.kind} className="mb-4 last:mb-0">
              <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                {g.label}
              </div>
              {g.kind === "llm" && registerForm}
              {rows.length ? (
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[34rem] text-left text-xs">
                    <thead>
                      <tr className="text-[10px] uppercase tracking-wider text-zinc-600">
                        <th className="pb-1 pr-2 font-medium">model</th>
                        <th className="pb-1 pr-2 font-medium">device</th>
                        <th className="pb-1 pr-2 font-medium">mem</th>
                        <th className="pb-1 pr-2 font-medium">size</th>
                        <th className="pb-1 pr-2 font-medium">state</th>
                        <th className="pb-1 pr-2 font-medium">flags</th>
                        <th className="pb-1 font-medium" />
                      </tr>
                    </thead>
                    <tbody>{rows.map(row)}</tbody>
                  </table>
                </div>
              ) : (
                <div className="text-xs text-zinc-500">
                  no {g.label.toLowerCase()} models known to the sidecar
                </div>
              )}
            </div>
          );
        })}
        {!models.length && (
          <div className="text-xs text-zinc-500">
            sidecar unreachable — start it with <code>make mesh-server</code>
          </div>
        )}
      </Card>

      {q && (
        <Card className="p-4">
          <SectionTitle
            right={
              q.free_mb != null ? (
                <Chip tone="zinc" title="free host RAM">
                  free {fmtMemGb(q.free_mb)}
                </Chip>
              ) : undefined
            }
          >
            Sidecar queue
          </SectionTitle>
          {q.jobs.length ? (
            <div className="mb-2 flex flex-col gap-1 text-xs">
              {q.jobs.map((j) => (
                <div
                  key={j.id}
                  className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-zinc-400"
                >
                  <Chip
                    tone={
                      j.status === "running"
                        ? "green"
                        : j.status === "queued"
                          ? "amber"
                          : "zinc"
                    }
                  >
                    {j.status}
                  </Chip>
                  <span className="text-zinc-300">
                    {j.kind} {j.model}
                    {j.resolved_model && j.resolved_model !== j.model
                      ? ` → ${j.resolved_model}`
                      : ""}
                  </span>
                  <span>
                    waited {fmtSecs(j.waited_s)}
                    {j.est_mem_mb != null ? ` · ${fmtMemGb(j.est_mem_mb)}` : ""}
                  </span>
                  {j.message ? (
                    <span className="text-zinc-500">{j.message}</span>
                  ) : null}
                </div>
              ))}
            </div>
          ) : (
            <div className="mb-2 text-xs text-zinc-500">no queued jobs</div>
          )}
          {q.residents.length > 0 && (
            <div className="text-xs text-zinc-500">
              resident:{" "}
              {q.residents.map((r, i) => (
                <span key={r.id}>
                  {i > 0 && " · "}
                  <b className="font-medium text-zinc-300">{r.id}</b>{" "}
                  {fmtMemGb(r.mb)} · {r.kind}
                  {r.pinned ? " · pinned" : ""}
                  {r.idle_s > 0 ? ` · idle ${fmtSecs(r.idle_s)}` : ""}
                </span>
              ))}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
