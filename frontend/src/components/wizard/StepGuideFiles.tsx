import { useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Check, FileText, FolderOpen, Package, Trash2, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { api, errMsg } from "../../lib/api";
import type { Asset } from "../../lib/types";
import { Btn, IconBtn, Spinner } from "../ui";

/** asset kinds usable as guide CAD parts (solids the parts mission can mesh) */
const CAD_KINDS = new Set<Asset["kind"]>(["reference_step", "reference_stl", "cad_glb"]);

/** One upload slot: dropzone + browse, per-section accept list and size cap
 *  matching the server's `_KIND_MAX_BYTES`. */
function UploadSlot({
  projectId,
  accept,
  maxMb,
  onUploaded,
}: {
  projectId: string;
  accept: string;
  maxMb: number;
  onUploaded: (created: Asset[]) => void;
}) {
  const { t } = useTranslation("wizard");
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    const tooBig = Array.from(files).filter((f) => f.size > maxMb * 1024 * 1024);
    if (tooBig.length) {
      setErr(t("gfiles_too_big", { names: tooBig.map((f) => f.name).join(", "), max: maxMb }));
      return;
    }
    setBusy(true);
    setErr("");
    try {
      const created = await api.uploadAssets(projectId, Array.from(files));
      qc.invalidateQueries({ queryKey: ["assets", projectId] });
      onUploaded(created);
    } catch (e) {
      setErr(errMsg(e));
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <div>
      <div
        className={`rounded-lg border-2 border-dashed px-3 py-2.5 text-center text-xs transition-colors ${
          drag ? "border-brand-500 bg-brand-950/20 text-brand-300" : "border-edge2 text-muted"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          upload(e.dataTransfer.files);
        }}
      >
        {busy ? (
          <span className="inline-flex items-center gap-2">
            <Spinner /> {t("common:asset.uploading")}
          </span>
        ) : (
          <>
            <Upload className="mr-1.5 inline h-3.5 w-3.5 align-[-2px]" />
            {t("common:asset.drop_pre")}{" "}
            <button className="text-brand-400 underline" onClick={() => fileRef.current?.click()}>
              {t("common:asset.browse")}
            </button>
            <span className="ml-1 text-faint/70">({accept.replaceAll(",", " ")})</span>
          </>
        )}
        <input
          ref={fileRef}
          type="file"
          multiple
          hidden
          accept={accept}
          onChange={(e) => upload(e.target.files)}
        />
      </div>
      {err && <div className="mt-1 text-[11px] text-red-400">{err}</div>}
    </div>
  );
}

function AssetRow({
  asset,
  selected,
  onPick,
  onDelete,
}: {
  asset: Asset;
  selected: boolean;
  onPick: () => void;
  onDelete: () => void;
}) {
  const { t } = useTranslation();
  const isPdf = asset.kind === "pdf";
  return (
    <div
      className={`group flex items-center gap-2 rounded-md border px-2.5 py-1.5 transition-colors ${
        selected ? "border-brand-500 bg-brand-950/30" : "border-edge hover:border-edge2"
      }`}
    >
      <button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left" onClick={onPick}>
        {isPdf ? (
          <FileText className="h-4 w-4 shrink-0 text-faint" />
        ) : (
          <Package className="h-4 w-4 shrink-0 text-faint" />
        )}
        <span className="min-w-0 flex-1 truncate text-xs text-fg" title={asset.filename}>
          {asset.filename}
        </span>
        <span className="shrink-0 rounded bg-black/60 px-1 text-[9px] uppercase text-muted">
          {isPdf ? "PDF" : asset.kind.replace("reference_", "").replace("cad_", "")}
        </span>
      </button>
      {selected && (
        <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-brand-500">
          <Check className="h-3 w-3 text-white" />
        </span>
      )}
      <IconBtn title={t("asset.delete")} danger className="h-5 w-5 opacity-0 transition-opacity group-hover:opacity-100" onClick={onDelete}>
        <Trash2 className="h-3 w-3" />
      </IconBtn>
    </div>
  );
}

/** Guide files step: required manual PDF (single-select) + optional CAD
 *  parts (multi-select). The project already exists — gotoStep created it. */
export default function StepGuideFiles({
  projectId,
  projectName,
  assets,
  pdfId,
  cadIds,
  onPickPdf,
  onToggleCad,
  onDeleted,
  onNext,
  onBack,
}: {
  projectId: string | null;
  projectName: string;
  assets: Asset[];
  pdfId: string | null;
  cadIds: string[];
  onPickPdf: (id: string) => void;
  onToggleCad: (id: string) => void;
  onDeleted: (id: string) => void;
  onNext: () => void;
  onBack: () => void;
}) {
  const { t } = useTranslation("wizard");
  const qc = useQueryClient();
  const pdfs = assets.filter((a) => a.kind === "pdf");
  const cads = assets.filter((a) => CAD_KINDS.has(a.kind));

  const del = async (id: string) => {
    try {
      await api.deleteAsset(id);
      onDeleted(id);
      if (projectId) qc.invalidateQueries({ queryKey: ["assets", projectId] });
    } catch {
      /* row stays — deletion failed */
    }
  };

  const uploaded = (created: Asset[]) => {
    // auto-select fresh uploads: a new PDF becomes THE manual, CAD files
    // join the parts selection
    const pdf = created.find((a) => a.kind === "pdf");
    if (pdf) onPickPdf(pdf.id);
    created.filter((a) => CAD_KINDS.has(a.kind)).forEach((a) => onToggleCad(a.id));
  };

  return (
    <div>
      <h2 className="text-lg font-semibold">{t("gfiles_title")}</h2>
      <p className="mt-1 text-sm text-faint">{t("gfiles_hint")}</p>

      {projectId ? (
        <div className="mt-4 flex flex-col gap-4">
          <section>
            <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-faint">
              {t("gfiles_pdf_label")} <span className="text-red-400">*</span>
            </div>
            <UploadSlot projectId={projectId} accept=".pdf" maxMb={100} onUploaded={uploaded} />
            {pdfs.length > 0 && (
              <div className="mt-2 flex flex-col gap-1">
                {pdfs.map((a) => (
                  <AssetRow
                    key={a.id}
                    asset={a}
                    selected={a.id === pdfId}
                    onPick={() => onPickPdf(a.id)}
                    onDelete={() => void del(a.id)}
                  />
                ))}
              </div>
            )}
          </section>

          <section>
            <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-faint">
              {t("gfiles_parts_label")}{" "}
              <span className="font-normal normal-case text-faint/70">
                ({t("common:field.optional")})
              </span>
            </div>
            <p className="mb-1.5 text-[11px] text-faint/80">{t("gfiles_parts_hint")}</p>
            <UploadSlot
              projectId={projectId}
              accept=".step,.stp,.stl,.glb,.gltf"
              maxMb={50}
              onUploaded={uploaded}
            />
            {cads.length > 0 && (
              <div className="mt-2 flex flex-col gap-1">
                {cads.map((a) => (
                  <AssetRow
                    key={a.id}
                    asset={a}
                    selected={cadIds.includes(a.id)}
                    onPick={() => onToggleCad(a.id)}
                    onDelete={() => void del(a.id)}
                  />
                ))}
              </div>
            )}
          </section>

          <div className="flex items-center gap-1.5 text-[11px] text-faint/80">
            <FolderOpen className="h-3 w-3" />
            {t("files_in_project")}{" "}
            <Link to={`/projects/${projectId}`} className="text-brand-400 hover:underline">
              {projectName}
            </Link>
          </div>
        </div>
      ) : (
        <div className="mt-4 flex items-center justify-center gap-2 rounded-lg border border-edge py-10 text-sm text-faint">
          <Spinner /> {t("creating_project")}
        </div>
      )}

      <div className="mt-6 flex items-center justify-between">
        <Btn variant="ghost" onClick={onBack}>
          {t("common:action.back")}
        </Btn>
        <Btn onClick={onNext} disabled={!pdfId} title={!pdfId ? t("gfiles_pdf_required") : undefined}>
          {t("continue")} <ArrowRight className="h-4 w-4" />
        </Btn>
      </div>
    </div>
  );
}
