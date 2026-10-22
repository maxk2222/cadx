import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Boxes,
  Check,
  ChevronDown,
  ChevronRight,
  Download,
  Package,
  Pencil,
  Search,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import ModelCanvas from "../components/editor/ModelCanvas";
import {
  Btn,
  PageShell,
  Card,
  Chip,
  ConfirmModal,
  EmptyState,
  IconBtn,
  Modal,
  OpacitySlider,
  SectionTitle,
  Spinner,
  inputCls,
  useDebounced,
  useToast,
} from "../components/ui";
import { api, errMsg } from "../lib/api";
import { fmtDateTime, timeAgo } from "../lib/format";
import type { LibraryPart } from "../lib/types";
import { useTitle } from "../lib/useTitle";

const KIND_FILTERS = ["all", "step", "stl", "template"] as const;
type KindFilter = (typeof KIND_FILTERS)[number];

export default function LibraryPage() {
  const { t } = useTranslation("library");
  useTitle(t("title"));
  return (
    <PageShell maxW="list" title={t("title")} subtitle={t("subtitle")} className="pb-16">
      <PartsSection />
      <CatalogBrowser />
    </PageShell>
  );
}

/* ---------- user parts ---------- */

function PartsSection() {
  const { t } = useTranslation("library");
  const qc = useQueryClient();
  const toast = useToast();
  const fileRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [caption, setCaption] = useState("");
  const [q, setQ] = useState("");
  const needle = useDebounced(q.trim().toLowerCase(), 150);
  const [kindFilter, setKindFilter] = useState<KindFilter>("all");
  const [detail, setDetail] = useState<LibraryPart | null>(null);
  const [deleting, setDeleting] = useState<LibraryPart | null>(null);

  const parts = useQuery({
    queryKey: ["library"],
    queryFn: api.listLibrary,
    staleTime: 10_000,
  });

  const upload = useMutation({
    mutationFn: (files: File[]) => api.uploadLibrary(files, name, caption),
    onSuccess: () => {
      setName("");
      setCaption("");
      qc.invalidateQueries({ queryKey: ["library"] });
    },
    onError: (e) => toast(errMsg(e), "error"),
  });
  const del = useMutation({
    mutationFn: (id: string) => api.deleteLibraryPart(id),
    onSuccess: () => {
      setDeleting(null);
      setDetail(null);
      qc.invalidateQueries({ queryKey: ["library"] });
    },
    onError: (e) => toast(errMsg(e), "error"),
  });

  const filtered = useMemo(() => {
    // backend returns oldest first — show newest first
    let list = [...(parts.data ?? [])].reverse();
    if (kindFilter !== "all") list = list.filter((p) => p.kind === kindFilter);
    if (needle)
      list = list.filter((p) =>
        [p.name, p.filename, p.caption]
          .join(" ")
          .toLowerCase()
          .includes(needle),
      );
    return list;
  }, [parts.data, kindFilter, needle]);

  return (
    <Card className="p-4">
      <SectionTitle>{t("reusable", { count: parts.data?.length ?? 0 })}</SectionTitle>

      {/* upload + toolbar */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <input
          ref={fileRef}
          type="file"
          accept=".step,.stp,.stl"
          multiple
          className="hidden"
          onChange={(e) => {
            const fs = [...(e.target.files ?? [])];
            if (fs.length) upload.mutate(fs);
            e.target.value = "";
          }}
        />
        <input
          className={inputCls + " w-40 text-xs"}
          placeholder={t("name_ph")}
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <input
          className={inputCls + " w-56 text-xs"}
          placeholder={t("caption_ph")}
          value={caption}
          onChange={(e) => setCaption(e.target.value)}
        />
        <Btn
          variant="ghost"
          onClick={() => fileRef.current?.click()}
          busy={upload.isPending}
        >
          <Upload className="h-3.5 w-3.5" /> {t("upload")}
        </Btn>
        <div className="relative ml-auto">
          <Search className="pointer-events-none absolute left-2 top-2.5 h-3.5 w-3.5 text-faint" />
          <input
            className={inputCls + " w-52 pl-7 text-xs"}
            placeholder={t("search_ph")}
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>
      </div>
      <div className="mb-3 flex gap-1">
        {KIND_FILTERS.map((k) => (
          <button
            key={k}
            onClick={() => setKindFilter(k)}
            className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
              kindFilter === k
                ? "bg-zinc-700 font-medium text-zinc-100"
                : "text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
            }`}
          >
            {k === "all" ? t("all") : k}
          </button>
        ))}
      </div>

      {parts.isLoading ? (
        <div className="flex justify-center py-8 text-zinc-500">
          <Spinner />
        </div>
      ) : filtered.length === 0 ? (
        <EmptyState
          title={needle || kindFilter !== "all" ? t("empty_filtered") : t("empty")}
          hint={
            needle || kindFilter !== "all"
              ? t("empty_filtered_hint")
              : t("empty_hint")
          }
        />
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((p) => (
            <div
              key={p.id}
              role="button"
              tabIndex={0}
              onClick={() => setDetail(p)}
              onKeyDown={(e) => e.key === "Enter" && setDetail(p)}
              className="group flex cursor-pointer flex-col rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 text-left transition-colors hover:border-zinc-600 hover:bg-zinc-900"
            >
              <div className="mb-1.5 flex items-center gap-2">
                <Chip tone={p.kind === "template" ? "purple" : "blue"}>
                  {p.kind}
                </Chip>
                <span className="ml-auto text-[10px] text-zinc-500">
                  {timeAgo(p.created_at)}
                </span>
              </div>
              <div className="truncate text-sm font-medium text-zinc-200">
                {p.name}
              </div>
              <div className="truncate text-xs text-zinc-500">
                {p.filename || t("code_template")}
              </div>
              {p.caption && (
                <div className="mt-1 line-clamp-2 text-xs text-zinc-500">
                  {p.caption}
                </div>
              )}
              <div className="mt-2 flex items-center gap-0.5 border-t border-zinc-800/60 pt-2 opacity-60 transition-opacity group-hover:opacity-100">
                <IconBtn title={t("details")} onClick={(e) => { e.stopPropagation(); setDetail(p); }}>
                  <Boxes className="h-3.5 w-3.5" />
                </IconBtn>
                {p.url && (
                  <a
                    href={p.url}
                    title={t("common:action.download")}
                    aria-label={t("common:action.download")}
                    className="inline-flex h-7 w-7 items-center justify-center rounded-md text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Download className="h-3.5 w-3.5" />
                  </a>
                )}
                <span className="ml-auto" />
                <IconBtn
                  title={t("common:action.delete")}
                  danger
                  onClick={(e) => {
                    e.stopPropagation();
                    setDeleting(p);
                  }}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </IconBtn>
              </div>
            </div>
          ))}
        </div>
      )}

      <PartDetail
        key={detail?.id ?? "none"}
        part={detail}
        onClose={() => setDetail(null)}
        onDelete={(p) => setDeleting(p)}
      />
      <ConfirmModal
        open={!!deleting}
        onClose={() => setDeleting(null)}
        onConfirm={() => deleting && del.mutate(deleting.id)}
        title={t("delete_title")}
        body={
          <>
            {t("delete_body_pre")}<b className="text-fg">{deleting?.name}</b>{t("delete_body_post")}
          </>
        }
        busy={del.isPending}
      />
    </Card>
  );
}

/* ---------- part detail / preview ---------- */

function PartPreview({ part }: { part: LibraryPart }) {
  const { t } = useTranslation("library");
  const [opacity, setOpacity] = useState(1);
  const needSource = part.kind !== "stl" && !!part.source_version_id;
  const src = useQuery({
    queryKey: ["version", part.source_version_id],
    queryFn: () => api.getVersion(part.source_version_id!),
    enabled: needSource,
    retry: false,
    staleTime: 60_000,
  });
  const glb = src.data?.artifacts.find((a) => a.kind === "glb");

  const canvas =
    part.kind === "stl" && part.url ? (
      <ModelCanvas url={part.url} format="stl" opacity={opacity} />
    ) : glb ? (
      <ModelCanvas url={glb.url} opacity={opacity} />
    ) : null;
  if (canvas)
    return (
      <div className="relative h-full w-full">
        {canvas}
        <OpacitySlider
          value={opacity}
          onChange={setOpacity}
          className="absolute bottom-2 right-2"
        />
      </div>
    );
  if (needSource && src.isLoading)
    return (
      <div className="flex h-full items-center justify-center text-zinc-500">
        <Spinner />
      </div>
    );
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-sm text-zinc-500">
      <Boxes className="h-8 w-8 text-zinc-700" />
      {t("no_preview")}
    </div>
  );
}

function PartDetail({
  part,
  onClose,
  onDelete,
}: {
  part: LibraryPart | null;
  onClose: () => void;
  onDelete: (p: LibraryPart) => void;
}) {
  const { t } = useTranslation("library");
  const qc = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState("");
  const [editCaption, setEditCaption] = useState("");

  const update = useMutation({
    mutationFn: (id: string) =>
      api.updateLibraryPart(id, {
        name: editName.trim(),
        caption: editCaption.trim(),
      }),
    onSuccess: () => {
      setEditing(false);
      qc.invalidateQueries({ queryKey: ["library"] });
    },
  });

  if (!part) return null;

  return (
    <Modal open={!!part} onClose={onClose} title={part.name} wide>
      <div className="mb-3 h-64 overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900">
        <PartPreview part={part} />
      </div>

      {editing ? (
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <input
            className={inputCls + " w-44 text-xs"}
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            aria-label={t("aria_name")}
            placeholder={t("common:field.name")}
          />
          <input
            className={inputCls + " min-w-40 flex-1 text-xs"}
            value={editCaption}
            onChange={(e) => setEditCaption(e.target.value)}
            aria-label={t("aria_caption")}
            placeholder={t("caption")}
          />
          <IconBtn
            title={t("common:action.save")}
            disabled={!editName.trim() || update.isPending}
            onClick={() => update.mutate(part.id)}
            className="text-emerald-500 hover:bg-emerald-950/50"
          >
            <Check className="h-3.5 w-3.5" />
          </IconBtn>
          <IconBtn title={t("common:action.cancel")} onClick={() => setEditing(false)}>
            <X className="h-3.5 w-3.5" />
          </IconBtn>
        </div>
      ) : (
        <div className="mb-3 flex items-center gap-2">
          <Chip tone={part.kind === "template" ? "purple" : "blue"}>
            {part.kind}
          </Chip>
          <span className="truncate text-xs text-zinc-500">
            {part.filename || "code"} {part.caption ? `— ${part.caption}` : ""}
          </span>
          <IconBtn
            title={t("edit_nc")}
            className="ml-auto"
            onClick={() => {
              setEditing(true);
              setEditName(part.name);
              setEditCaption(part.caption);
            }}
          >
            <Pencil className="h-3.5 w-3.5" />
          </IconBtn>
        </div>
      )}
      {update.isError && (
        <div className="mb-3 text-xs text-red-400">{errMsg(update.error)}</div>
      )}

      <div className="mb-3 flex flex-col gap-1 text-xs text-zinc-500">
        <div>
          {t("added")} <span className="text-zinc-300">{fmtDateTime(part.created_at)}</span>
        </div>
        {part.source_version_id && (
          <div>
            {t("source")}{" "}
            <Link
              to={`/versions/${part.source_version_id}`}
              className="text-brand-400 hover:underline"
            >
              {t("source_version", { id: part.source_version_id.slice(0, 8) })}
            </Link>
          </div>
        )}
        {part.params && Object.keys(part.params).length > 0 && (
          <div className="mt-1 rounded border border-zinc-800 bg-zinc-900/60 p-2 font-mono text-[11px] text-zinc-400">
            {Object.entries(part.params).map(([k, v]) => (
              <div key={k}>
                <span className="text-zinc-300">{k}</span>: {JSON.stringify(v)}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        {part.url && (
          <a href={part.url} className="inline-flex">
            <Btn variant="ghost">
              <Download className="h-3.5 w-3.5" /> {t("common:action.download")}
            </Btn>
          </a>
        )}
        <Btn
          variant="danger"
          className="ml-auto"
          onClick={() => onDelete(part)}
        >
          <Trash2 className="h-3.5 w-3.5" /> {t("common:action.delete")}
        </Btn>
      </div>
    </Modal>
  );
}

/* ---------- standard parts catalog ---------- */

function CatalogBrowser() {
  const { t } = useTranslation("library");
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [detail, setDetail] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const catalog = useQuery({
    queryKey: ["catalog"],
    queryFn: api.getCatalog,
    enabled: open,
    staleTime: Infinity,
  });

  const toggle = (k: string) =>
    setExpanded((s) => {
      const n = new Set(s);
      if (n.has(k)) n.delete(k);
      else n.add(k);
      return n;
    });

  const needle = q.trim().toLowerCase();
  const filtered = useMemo(() => {
    const data = catalog.data ?? {};
    if (!needle) return data;
    const out: typeof data = {};
    for (const [cat, classes] of Object.entries(data)) {
      const hits = Object.fromEntries(
        Object.entries(classes).filter(([cls, e]) => {
          const hay = [
            cls,
            cat,
            e.doc,
            ...Object.keys(e.types ?? {}),
            ...Object.values(e.types ?? {}).flat(),
            ...(e.params ?? []),
          ]
            .join(" ")
            .toLowerCase();
          return hay.includes(needle);
        }),
      );
      if (Object.keys(hits).length) out[cat] = hits;
    }
    return out;
  }, [catalog.data, needle]);

  return (
    <Card className="mt-4 p-4">
      <button
        className="flex w-full items-center gap-2 text-left"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? (
          <ChevronDown className="h-4 w-4 text-zinc-500" />
        ) : (
          <ChevronRight className="h-4 w-4 text-zinc-500" />
        )}
        <Package className="h-4 w-4 text-zinc-500" />
        <span className="text-xs font-semibold uppercase tracking-wider text-faint">
          {t("catalog")}
        </span>
      </button>
      {open && (
        <div className="mt-3">
          <div className="relative mb-2">
            <Search className="pointer-events-none absolute left-2 top-2 h-3.5 w-3.5 text-faint" />
            <input
              className={inputCls + " pl-7 text-xs"}
              placeholder={t("catalog_ph")}
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
          {catalog.isLoading ? (
            <div className="flex justify-center py-4 text-zinc-500">
              <Spinner />
            </div>
          ) : catalog.isError ? (
            <EmptyState title={t("catalog_unavail")} hint={t("catalog_unavail_hint")} />
          ) : (
            Object.entries(filtered).map(([cat, classes]) => (
              <div key={cat} className="mb-1">
                <button
                  className="flex w-full items-center gap-1.5 rounded px-1 py-1 text-sm text-zinc-300 hover:bg-zinc-800"
                  onClick={() => toggle(cat)}
                >
                  {expanded.has(cat) || needle ? (
                    <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
                  ) : (
                    <ChevronRight className="h-3.5 w-3.5 text-zinc-500" />
                  )}
                  {cat}
                  <span className="text-xs text-zinc-500">
                    ({Object.keys(classes).length})
                  </span>
                </button>
                {(expanded.has(cat) || needle) && (
                  <div className="ml-6 flex flex-col gap-0.5 border-l border-zinc-800 pl-2">
                    {Object.entries(classes).map(([cls, entry]) => {
                      const sizes = entry.types
                        ? Object.values(entry.types).flat()
                        : [];
                      const preview =
                        entry.kind === "parametric"
                          ? (entry.params ?? []).slice(0, 6).join(", ")
                          : sizes.slice(0, 8).join(", ") +
                            (sizes.length > 8 ? `, +${sizes.length - 8}` : "");
                      const openDetail = detail === cls;
                      return (
                        <div key={cls} className="py-0.5 text-xs">
                          <button
                            className="text-left font-mono text-zinc-300 hover:text-zinc-100"
                            onClick={() => setDetail(openDetail ? null : cls)}
                          >
                            {cls}
                          </button>
                          <span className="ml-2 text-zinc-500">
                            {entry.kind === "parametric" ? t("params_prefix") : ""}
                            {preview}
                          </span>
                          {openDetail && (
                            <div className="my-1 rounded border border-zinc-800 bg-zinc-900/60 p-2 font-mono text-[11px] text-zinc-400">
                              <div className="mb-1 text-zinc-500">{entry.doc}</div>
                              {entry.types &&
                                Object.entries(entry.types).map(([t, ss]) => (
                                  <div key={t}>
                                    <span className="text-zinc-300">{t}</span>:{" "}
                                    {ss.join(", ")}
                                  </div>
                                ))}
                              {entry.params && (
                                <div>{t("params_list", { list: entry.params.join(", ") })}</div>
                              )}
                              {entry.holes && (
                                <div>{t("holes_list", { list: entry.holes.join(", ") })}</div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </Card>
  );
}
