import {
  PencilRuler,
  Plus,
  Shrink,
  Sprout,
  Wand2,
} from "lucide-react";
import { useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { errMsg, api } from "../../lib/api";
import type {
  Asset,
  GenMode,
  Job,
  LibraryPart,
  RunOptions,
  VersionDetail,
  VersionSummary,
} from "../../lib/types";
import type { JobStream } from "../../lib/useJobEvents";
import FeatureEditPanel from "../FeatureEditPanel";
import JobLog from "../JobLog";
import ModeSelector from "../ModeSelector";
import OptionsPanel from "../OptionsPanel";
import ParamsPanel from "../ParamsPanel";
import VersionTree from "../VersionTree";
import { Btn, SectionCollapsible, inputCls } from "../ui";

/** uniform secondary-action row used inside the Actions inspector section */
function ActionRow({
  icon,
  label,
  hint,
  onClick,
}: {
  icon: ReactNode;
  label: string;
  hint: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      title={hint}
      onClick={onClick}
      className="flex w-full items-center gap-2.5 rounded-md border border-edge bg-panel2/40 px-3 py-2 text-left text-sm text-fg transition-all duration-150 hover:border-edge2 hover:bg-panel2"
    >
      <span className="text-muted">{icon}</span>
      <span className="min-w-0 flex-1">{label}</span>
    </button>
  );
}

/** the "edit this model" prompt form — collapsed by default inside the
 *  inspector so the sidebar's primary surface stays the job activity. */
function EditWithPrompt({
  v,
  editPrompt,
  setEditPrompt,
  editAssets,
  setEditAssets,
  editMode,
  setEditMode,
  editOptions,
  setEditOptions,
  editLibraryParts,
  setEditLibraryParts,
  regenMesh,
  setRegenMesh,
  assets,
  library,
  submitting,
  submitError,
  onSubmit,
}: {
  v: VersionDetail;
  editPrompt: string;
  setEditPrompt: (s: string) => void;
  editAssets: Set<string>;
  setEditAssets: (fn: (s: Set<string>) => Set<string>) => void;
  editMode: GenMode;
  setEditMode: (m: GenMode) => void;
  editOptions: RunOptions;
  setEditOptions: (o: RunOptions) => void;
  editLibraryParts: Set<string>;
  setEditLibraryParts: (fn: (s: Set<string>) => Set<string>) => void;
  regenMesh: boolean;
  setRegenMesh: (b: boolean) => void;
  assets: Asset[];
  library: LibraryPart[];
  submitting: boolean;
  submitError: unknown;
  onSubmit: () => void;
}) {
  const { t } = useTranslation("version");
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  const uploadEditAssets = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploading(true);
    try {
      const uploaded = await api.uploadAssets(v.project_id, Array.from(files));
      setEditAssets((s) => new Set([...s, ...uploaded.map((u) => u.id)]));
    } finally {
      setUploading(false);
    }
  };

  const toggleAsset = (id: string) =>
    setEditAssets((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  return (
    <>
      <textarea
        className={`${inputCls} h-20 resize-y`}
        placeholder={t("edit_placeholder")}
        value={editPrompt}
        onChange={(e) => setEditPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && editPrompt.trim()) {
            e.preventDefault();
            onSubmit();
          }
        }}
      />
      <div className="my-2 flex flex-wrap gap-1">
        {assets
          .filter((as) => as.kind === "image" || as.kind === "generated_image")
          .map((as) => (
            <button
              key={as.id}
              type="button"
              aria-label={
                editAssets.has(as.id)
                  ? t("img_remove", { name: as.filename })
                  : t("img_use", { name: as.filename })
              }
              onClick={() => toggleAsset(as.id)}
              className={`h-10 w-10 overflow-hidden rounded border-2 transition-colors ${
                editAssets.has(as.id) ? "border-brand-500" : "border-edge2"
              }`}
              title={as.filename}
            >
              <img src={as.url} className="h-full w-full object-cover" alt="" />
            </button>
          ))}
        <button
          type="button"
          aria-label={t("img_upload")}
          className="flex h-10 w-10 items-center justify-center rounded border border-dashed border-edge2 text-muted transition-colors hover:border-faint hover:text-fg"
          onClick={() => fileRef.current?.click()}
          title={t("img_upload")}
          disabled={uploading}
        >
          <Plus className="h-4 w-4" />
        </button>
        <input
          ref={fileRef}
          type="file"
          multiple
          hidden
          accept="image/*"
          onChange={(e) => uploadEditAssets(e.target.files)}
        />
      </div>
      <div className="my-2">
        <ModeSelector value={editMode} onChange={setEditMode} />
      </div>
      <div className="my-2">
        <OptionsPanel value={editOptions} onChange={setEditOptions} />
      </div>
      {library.length > 0 && (
        <div className="my-2">
          <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-faint">
            {t("library_parts")}
            {editLibraryParts.size
              ? ` — ${t("library_selected", { count: editLibraryParts.size })}`
              : ""}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {library.map((p) => {
              const on = editLibraryParts.has(p.id);
              return (
                <button
                  key={p.id}
                  type="button"
                  title={`${p.kind}: ${p.caption || p.filename || p.name}`}
                  className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
                    on
                      ? "border-brand-600 bg-brand-950/60 text-brand-200"
                      : "border-edge2 bg-panel2 text-muted hover:border-faint/50 hover:text-fg"
                  }`}
                  onClick={() =>
                    setEditLibraryParts((s) => {
                      const n = new Set(s);
                      if (n.has(p.id)) n.delete(p.id);
                      else n.add(p.id);
                      return n;
                    })
                  }
                >
                  {p.name}
                  <span className="ml-1 text-[10px] text-faint">{p.kind}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
      {v.resolved_mode === "hybrid" && (
        <label className="my-2 flex items-center gap-2 text-xs text-muted">
          <input
            type="checkbox"
            checked={regenMesh}
            onChange={(e) => setRegenMesh(e.target.checked)}
            className="accent-brand-500"
          />
          {t("regen_mesh")}
        </label>
      )}
      {submitError != null && (
        <div className="mb-2 text-xs text-red-400">{errMsg(submitError)}</div>
      )}
      <Btn
        variant="success"
        className="w-full"
        onClick={onSubmit}
        busy={submitting}
        disabled={!editPrompt.trim()}
      >
        {t("apply_edit")}
      </Btn>
    </>
  );
}

/** the right-hand inspector column — agent activity on top, then a uniform
 *  action list, then progressively less-used panels behind collapsibles. */
export default function Inspector({
  v,
  jobId,
  stream,
  jobs,
  versions,
  assets,
  library,
  onSelectJob,
  onDone,
  onEditVisual,
  onEditStructural,
  onEditOrganic,
  edit,
}: {
  v: VersionDetail;
  jobId: string | null;
  stream: JobStream;
  jobs: Job[] | undefined;
  versions: VersionSummary[];
  assets: Asset[];
  library: LibraryPart[];
  onSelectJob: (id: string) => void;
  onDone: () => void;
  onEditVisual: () => void;
  onEditStructural: () => void;
  onEditOrganic: () => void;
  edit: Omit<Parameters<typeof EditWithPrompt>[0], "v" | "assets" | "library">;
}) {
  const { t } = useTranslation("version");
  const finished =
    v.status === "succeeded" || v.status === "needs_review";
  const hasGlb = v.artifacts.some((a) => a.kind === "glb");

  return (
    <div className="flex flex-col gap-4">
      <SectionCollapsible
        title={t("insp_activity")}
        persistKey="ver:activity"
        pad="p-3"
      >
        <JobLog
          jobId={jobId}
          stream={stream}
          jobs={jobs}
          onSelectJob={onSelectJob}
          onDone={onDone}
        />
      </SectionCollapsible>

      {finished && (
        <SectionCollapsible title={t("insp_actions")} persistKey="ver:actions" pad="p-3">
          <div className="flex flex-col gap-1.5">
            {hasGlb && (
              <ActionRow
                icon={<PencilRuler className="h-4 w-4" />}
                label={t("action_edit_visual")}
                hint={t("action_edit_visual_hint")}
                onClick={onEditVisual}
              />
            )}
            <ActionRow
              icon={<Shrink className="h-4 w-4" />}
              label={t("action_structural")}
              hint={t("action_structural_hint")}
              onClick={onEditStructural}
            />
            {v.artifacts.some((a) => a.kind === "glb" || a.kind === "stl") && (
              <ActionRow
                icon={<Sprout className="h-4 w-4" />}
                label={t("action_organic")}
                hint={t("action_organic_hint")}
                onClick={onEditOrganic}
              />
            )}
          </div>
        </SectionCollapsible>
      )}

      {finished && (
        <SectionCollapsible
          title={
            <span className="inline-flex items-center gap-1.5">
              <Wand2 className="h-3 w-3" /> {t("insp_edit_prompt")}
            </span>
          }
          right={
            <span className="text-[11px] normal-case tracking-normal text-faint">
              {t("ctrl_enter")}
            </span>
          }
          defaultOpen={false}
          persistKey="ver:edit-prompt"
          pad="p-3"
        >
          <EditWithPrompt v={v} assets={assets} library={library} {...edit} />
        </SectionCollapsible>
      )}

      {!!v.params_schema?.length && (
        <SectionCollapsible
          title={t("insp_params")}
          defaultOpen={false}
          persistKey="ver:params"
          pad="p-3"
        >
          <ParamsPanel
            versionId={v.id}
            schema={v.params_schema}
            params={v.params}
          />
        </SectionCollapsible>
      )}

      {v.resolved_mode !== "mesh" && (
        <SectionCollapsible
          title={t("insp_feature_edit")}
          defaultOpen={false}
          persistKey="ver:feature-edit"
          pad="p-3"
        >
          <FeatureEditPanel versionId={v.id} />
        </SectionCollapsible>
      )}

      <SectionCollapsible
        title={t("insp_versions")}
        count={versions.length}
        defaultOpen={false}
        persistKey="ver:tree"
        pad="p-3"
      >
        <VersionTree versions={versions} activeId={v.id} />
      </SectionCollapsible>
    </div>
  );
}
