import { ArrowRight, FolderOpen } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import type { Asset } from "../../lib/types";
import AssetManager from "../AssetManager";
import { Btn, Spinner } from "../ui";

/** Wizard images step. The project is created by the parent when the step is
 *  entered, so the full AssetManager (upload, AI image generation, sketch
 *  toggle, editor) is available immediately — everything stays in the wizard. */
export default function StepImages({
  projectId,
  projectName,
  assets,
  selected,
  onToggle,
  onAssetReplaced,
  onDeleted,
  onNext,
  onBack,
}: {
  projectId: string | null;
  projectName: string;
  assets: Asset[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  onAssetReplaced: (srcId: string, newId: string) => void;
  /** an asset was deleted server-side — drop it from the wizard selection */
  onDeleted?: (id: string) => void;
  onNext: () => void;
  onBack: () => void;
}) {
  const { t } = useTranslation("wizard");
  return (
    <div>
      <h2 className="text-lg font-semibold">
        {t("images_title")}{" "}
        <span className="text-sm font-normal text-faint">
          ({t("common:field.optional")})
        </span>
      </h2>
      <p className="mt-1 text-sm text-faint">{t("images_hint")}</p>

      {projectId ? (
        <div className="mt-4">
          <AssetManager
            projectId={projectId}
            assets={assets}
            selected={selected}
            onToggle={onToggle}
            onAssetReplaced={onAssetReplaced}
            onDeleted={onDeleted}
          />
          <div className="mt-2 flex items-center gap-1.5 text-[11px] text-faint/80">
            <FolderOpen className="h-3 w-3" />
            {t("files_in_project")}{" "}
            <Link
              to={`/projects/${projectId}`}
              className="text-brand-400 hover:underline"
            >
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
        <div className="flex items-center gap-3">
          <button
            className="text-xs text-faint underline transition-colors hover:text-muted"
            onClick={onNext}
          >
            {t("skip_images")}
          </button>
          <Btn onClick={onNext}>
            {t("continue")} <ArrowRight className="h-4 w-4" />
          </Btn>
        </div>
      </div>
    </div>
  );
}
