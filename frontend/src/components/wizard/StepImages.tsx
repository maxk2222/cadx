import { ArrowRight, FolderOpen } from "lucide-react";
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
  onNext,
  onBack,
}: {
  projectId: string | null;
  projectName: string;
  assets: Asset[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  onAssetReplaced: (srcId: string, newId: string) => void;
  onNext: () => void;
  onBack: () => void;
}) {
  return (
    <div>
      <h2 className="text-lg font-semibold">
        Reference images{" "}
        <span className="text-sm font-normal text-zinc-500">(optional)</span>
      </h2>
      <p className="mt-1 text-sm text-zinc-500">
        Photos, drawings or sketches help the agent understand the shape — or let
        a local image model generate a concept. Selected images feed the run.
      </p>

      {projectId ? (
        <div className="mt-4">
          <AssetManager
            projectId={projectId}
            assets={assets}
            selected={selected}
            onToggle={onToggle}
            onAssetReplaced={onAssetReplaced}
          />
          <div className="mt-2 flex items-center gap-1.5 text-[11px] text-zinc-600">
            <FolderOpen className="h-3 w-3" />
            files are stored in project{" "}
            <Link
              to={`/projects/${projectId}`}
              className="text-brand-400 hover:underline"
            >
              {projectName}
            </Link>
          </div>
        </div>
      ) : (
        <div className="mt-4 flex items-center justify-center gap-2 rounded-lg border border-zinc-800 py-10 text-sm text-zinc-500">
          <Spinner /> creating project…
        </div>
      )}

      <div className="mt-6 flex items-center justify-between">
        <Btn variant="ghost" onClick={onBack}>
          Back
        </Btn>
        <div className="flex items-center gap-3">
          <button
            className="text-xs text-zinc-500 underline hover:text-zinc-300"
            onClick={onNext}
          >
            continue without images
          </button>
          <Btn onClick={onNext}>
            Continue <ArrowRight className="h-4 w-4" />
          </Btn>
        </div>
      </div>
    </div>
  );
}
