import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, FileText, Images, Layers, Rocket } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import StepImages from "../components/wizard/StepImages";
import StepMode from "../components/wizard/StepMode";
import StepPrompt from "../components/wizard/StepPrompt";
import StepReview from "../components/wizard/StepReview";
import WizardProgress from "../components/wizard/WizardProgress";
import { Card } from "../components/ui";
import { api, errMsg } from "../lib/api";
import { deriveProjectName, suggestMode } from "../lib/suggest";
import type { GenMode, MeshHealth, RunOptions } from "../lib/types";
import { useTitle } from "../lib/useTitle";

const DRAFT_KEY = "wizard:draft";

const STEPS = [
  { id: 0, labelKey: "step_describe", icon: FileText },
  { id: 1, labelKey: "step_images", icon: Images },
  { id: 2, labelKey: "step_mode", icon: Layers },
  { id: 3, labelKey: "step_generate", icon: Rocket },
];

interface WizardState {
  step: number;
  /** furthest step ever reached — stepper may navigate back AND forward to it */
  maxStep: number;
  phase: "edit" | "progress";
  prompt: string;
  mode: GenMode;
  /** user picked a mode explicitly — the auto-suggestion must not override it */
  modeTouched: boolean;
  options: RunOptions;
  name: string;
  projectId: string | null;
  selected: string[];
  jobId: string | null;
  versionId: string | null;
}

const INITIAL: WizardState = {
  step: 0,
  maxStep: 0,
  phase: "edit",
  prompt: "",
  mode: "auto",
  modeTouched: false,
  options: {},
  name: "",
  projectId: null,
  selected: [],
  jobId: null,
  versionId: null,
};

function loadDraft(): WizardState {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) return INITIAL;
    const d = { ...INITIAL, ...(JSON.parse(raw) as Partial<WizardState>) };
    // progress without a job/version is a broken draft — back to the steps
    if (d.phase === "progress" && !(d.jobId && d.versionId)) {
      d.phase = "edit";
      d.step = 0;
    }
    return d;
  } catch {
    return INITIAL;
  }
}

export default function StartPage() {
  const { t } = useTranslation("wizard");
  useTitle(t("page_title"));
  const nav = useNavigate();
  const qc = useQueryClient();
  const [s, setS] = useState<WizardState>(loadDraft);
  const [error, setError] = useState("");
  const [creating, setCreating] = useState(false);
  const seenAssets = useRef<Set<string>>(new Set());
  const projectIdRef = useRef<string | null>(s.projectId);
  /** in-flight createProject promise — dedupes concurrent ensureProject calls */
  const creatingRef = useRef<Promise<string> | null>(null);

  // draft persistence — reload resumes the wizard (and a running job)
  useEffect(() => {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(s));
  }, [s]);

  // stale draft: resumed job no longer exists → back to the input steps
  useEffect(() => {
    if (s.phase !== "progress" || !s.jobId) return;
    api.getJob(s.jobId).catch(() =>
      setS((w) =>
        w.jobId === s.jobId
          ? { ...w, phase: "edit", step: 0, jobId: null, versionId: null }
          : w
      )
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const set = (patch: Partial<WizardState>) =>
    setS((w) => ({
      ...w,
      ...patch,
      maxStep:
        patch.step != null ? Math.max(w.maxStep, patch.step) : w.maxStep,
    }));

  const ensureProject = async (): Promise<string> => {
    if (projectIdRef.current) return projectIdRef.current;
    if (creatingRef.current) return creatingRef.current;
    const run = (async () => {
      const name = s.name.trim() || deriveProjectName(s.prompt);
      const p = await api.createProject(name, "");
      projectIdRef.current = p.id;
      setS((w) => ({
        ...w,
        projectId: p.id,
        name: w.name || name,
      }));
      qc.invalidateQueries({ queryKey: ["projects"] });
      return p.id;
    })();
    creatingRef.current = run;
    try {
      return await run;
    } finally {
      creatingRef.current = null;
    }
  };

  /** entering the images step requires the project to exist (uploads,
   *  image generation and asset listing all need an id). */
  const gotoStep = async (next: number) => {
    setError("");
    if (next >= 1 && !projectIdRef.current && !creatingRef.current) {
      setCreating(true);
      try {
        await ensureProject();
      } catch (e) {
        setError(t("create_failed", { msg: errMsg(e) }));
        setCreating(false);
        return;
      }
      setCreating(false);
    }
    set({ step: next });
  };

  /** mode step entered for the first time → preselect the suggestion */
  const gotoModeStep = () => {
    void gotoStep(2).then(() => {
      setS((w) =>
        w.step === 2 && !w.modeTouched ? { ...w, mode: suggested } : w
      );
    });
  };

  // draft restored mid-wizard without a project (legacy draft) — create it
  useEffect(() => {
    if (s.phase === "edit" && s.step >= 1 && !s.projectId && !creatingRef.current) {
      ensureProject().catch((e) => {
        setError(t("create_failed", { msg: errMsg(e) }));
        set({ step: 0 });
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.step, s.projectId, s.phase]);

  const assets = useQuery({
    queryKey: ["assets", s.projectId],
    queryFn: () => api.listAssets(s.projectId!),
    enabled: !!s.projectId,
  });

  const meshHealth = useQuery<MeshHealth>({
    queryKey: ["meshHealth"],
    queryFn: api.meshHealth,
    staleTime: 30_000,
    retry: false,
  });

  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    staleTime: 60_000,
    retry: false,
  });

  // auto-select newly added assets (deselected ones stay deselected)
  useEffect(() => {
    const list = assets.data ?? [];
    const fresh = list.filter((a) => !seenAssets.current.has(a.id));
    if (!fresh.length) return;
    fresh.forEach((a) => seenAssets.current.add(a.id));
    setS((w) => ({ ...w, selected: [...w.selected, ...fresh.map((a) => a.id)] }));
  }, [assets.data]);

  const suggested = suggestMode(
    s.prompt,
    s.selected.length > 0,
    meshHealth.data?.ok ?? false
  );

  const generate = useMutation({
    mutationFn: async () => {
      const pid = await ensureProject();
      const name = s.name.trim();
      if (name) await api.updateProject(pid, { name });
      return api.generate(pid, s.prompt, s.selected, s.mode, s.options);
    },
    onSuccess: (r) => {
      setError("");
      setS((w) => ({
        ...w,
        phase: "progress",
        jobId: r.job_id,
        versionId: r.version_id,
      }));
    },
    onError: (e) => setError(errMsg(e)),
  });

  /** "start over" would orphan the draft project the wizard created — offer
   *  to delete it when it never produced a version. */
  const discardEmptyDraft = async (pid: string) => {
    try {
      const versions = await api.listVersions(pid);
      if (versions.length) return;
      if (window.confirm(t("discard_draft_confirm"))) {
        await api.deleteProject(pid);
        qc.invalidateQueries({ queryKey: ["projects"] });
      }
    } catch {
      /* lookup/delete failed — keep the project */
    }
  };

  const reset = () => {
    localStorage.removeItem(DRAFT_KEY);
    const pid = projectIdRef.current;
    projectIdRef.current = null;
    creatingRef.current = null;
    seenAssets.current = new Set();
    setError("");
    setS(INITIAL);
    if (pid) void discardEmptyDraft(pid);
  };

  return (
    <div className="mx-auto max-w-3xl p-6 lg:p-8">
      <div className="anim-fade-in-up mb-6 text-center">
        <h1 className="text-3xl font-bold tracking-tight">{t("hero_title")}</h1>
        <p className="mt-1 text-sm text-faint">{t("hero_sub")}</p>
      </div>

      {/* stepper */}
      <div className="mb-6 flex items-center gap-1">
        {STEPS.map((st, i) => {
          const Icon = st.icon;
          const active = s.phase === "edit" && s.step === i;
          const reached = i <= s.maxStep || s.phase === "progress";
          return (
            <div key={st.id} className="flex flex-1 items-center gap-1">
              {i > 0 && <span className="h-px flex-1 bg-edge" />}
              <button
                type="button"
                disabled={!reached || s.phase === "progress"}
                onClick={() => set({ step: i })}
                className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-all duration-150 disabled:cursor-default ${
                  active
                    ? "glow-accent bg-brand-600 text-white"
                    : reached
                      ? "bg-panel2 text-zinc-300 hover:bg-edge2"
                      : "bg-panel text-faint/60"
                }`}
              >
                {s.phase === "progress" || (!active && i < s.step) ? (
                  <Check className="h-3 w-3" />
                ) : (
                  <Icon className="h-3.5 w-3.5" />
                )}
                {t(st.labelKey)}
              </button>
              {i < STEPS.length - 1 && <span className="h-px flex-1 bg-edge" />}
            </div>
          );
        })}
      </div>

      <Card className="p-5">
        {s.phase === "progress" && s.jobId && s.versionId ? (
          <WizardProgress
            jobId={s.jobId}
            versionId={s.versionId}
            projectId={s.projectId}
            projectName={s.name}
            onSelectJob={(id) => set({ jobId: id })}
            onDone={() => {
              qc.invalidateQueries({ queryKey: ["version", s.versionId] });
              qc.invalidateQueries({ queryKey: ["versions", s.projectId] });
              qc.invalidateQueries({ queryKey: ["project", s.projectId] });
              qc.invalidateQueries({ queryKey: ["projects"] });
            }}
            onAdjustPrompt={() =>
              set({ phase: "edit", step: 0, jobId: null, versionId: null })
            }
            onNewPart={reset}
          />
        ) : s.step === 0 ? (
          <StepPrompt
            prompt={s.prompt}
            onChange={(v) => set({ prompt: v })}
            onNext={() => void gotoStep(1)}
            busy={creating}
            error={error}
          />
        ) : s.step === 1 ? (
          <StepImages
            projectId={s.projectId}
            projectName={s.name}
            assets={assets.data ?? []}
            selected={new Set(s.selected)}
            onToggle={(id) =>
              set({
                selected: s.selected.includes(id)
                  ? s.selected.filter((x) => x !== id)
                  : [...s.selected, id],
              })
            }
            onAssetReplaced={(srcId, newId) =>
              set({
                selected: s.selected.includes(srcId)
                  ? s.selected.map((x) => (x === srcId ? newId : x))
                  : s.selected,
              })
            }
            onDeleted={(id) =>
              set({ selected: s.selected.filter((x) => x !== id) })
            }
            onNext={() => gotoModeStep()}
            onBack={() => set({ step: 0 })}
          />
        ) : s.step === 2 ? (
          <StepMode
            value={s.mode}
            onChange={(m) => set({ mode: m, modeTouched: true })}
            suggested={suggested}
            hasImages={s.selected.length > 0}
            health={meshHealth.data}
            loading={meshHealth.isPending}
            onNext={() => set({ step: 3 })}
            onBack={() => set({ step: 1 })}
          />
        ) : (
          <StepReview
            name={s.name || deriveProjectName(s.prompt)}
            nameAuto={!s.name || s.name === deriveProjectName(s.prompt)}
            onNameChange={(v) => set({ name: v })}
            prompt={s.prompt}
            assetCount={s.selected.length}
            mode={s.mode}
            suggested={suggested}
            options={s.options}
            onOptionsChange={(o) => set({ options: o })}
            busy={generate.isPending}
            llmConfigured={health.data?.llm_configured}
            error={error}
            onGenerate={() => generate.mutate()}
            onBack={() => set({ step: 2 })}
          />
        )}
      </Card>

      {s.phase === "edit" && (s.jobId || s.projectId) && (
        <div className="mt-3 text-center">
          <button
            className="text-[11px] text-faint/70 underline transition-colors hover:text-muted"
            onClick={reset}
          >
            {t("start_over")}
            {s.projectId ? ` ${t("start_over_note")}` : ""}
          </button>
        </div>
      )}

      <div className="mt-8 text-center text-xs text-faint/70">
        {t("looking_existing")}{" "}
        <button className="text-brand-400 hover:underline" onClick={() => nav("/projects")}>
          {t("browse_projects")}
        </button>
      </div>
    </div>
  );
}
