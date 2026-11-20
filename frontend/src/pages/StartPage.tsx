import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, FileText, Goal, Images, Layers, NotebookPen, Paperclip, Rocket } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate, useSearchParams } from "react-router-dom";
import StepGoal from "../components/wizard/StepGoal";
import StepGuideFiles from "../components/wizard/StepGuideFiles";
import StepGuideNotes from "../components/wizard/StepGuideNotes";
import StepGuideReview from "../components/wizard/StepGuideReview";
import StepImages from "../components/wizard/StepImages";
import StepMode from "../components/wizard/StepMode";
import StepPrompt from "../components/wizard/StepPrompt";
import StepReview from "../components/wizard/StepReview";
import WizardProgress from "../components/wizard/WizardProgress";
import { Card, ConfirmModal } from "../components/ui";
import { api, errMsg } from "../lib/api";
import { deriveProjectName, suggestMode } from "../lib/suggest";
import type { GenMode, GuideOpts, MeshHealth, RunOptions, WizardGoal } from "../lib/types";
import { useTitle } from "../lib/useTitle";

const DRAFT_KEY = "wizard:draft";

const PART_STEPS = [
  { id: 0, labelKey: "step_goal", icon: Goal },
  { id: 1, labelKey: "step_describe", icon: FileText },
  { id: 2, labelKey: "step_images", icon: Images },
  { id: 3, labelKey: "step_mode", icon: Layers },
  { id: 4, labelKey: "step_generate", icon: Rocket },
];

const GUIDE_STEPS = [
  { id: 0, labelKey: "step_goal", icon: Goal },
  { id: 1, labelKey: "step_files", icon: Paperclip },
  { id: 2, labelKey: "step_notes", icon: NotebookPen },
  { id: 3, labelKey: "step_start", icon: Rocket },
];

interface WizardState {
  /** "" = goal not picked yet — step 0 shows the Part/Anleitung cards */
  goal: WizardGoal | "";
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
  /** user typed a name — auto-derivation (prompt / PDF stem) must not override */
  nameTouched: boolean;
  projectId: string | null;
  /** deep-linked into an existing project (?project=) — the wizard must
   *  never create a new one nor offer to delete it on "start over" */
  adoptedProject: boolean;
  selected: string[];
  /** guide flow: the instruction PDF (required, single) + CAD part uploads */
  pdfId: string | null;
  cadIds: string[];
  guideOpts: GuideOpts;
  jobId: string | null;
  versionId: string | null;
}

const INITIAL: WizardState = {
  goal: "",
  step: 0,
  maxStep: 0,
  phase: "edit",
  prompt: "",
  mode: "auto",
  modeTouched: false,
  options: {},
  name: "",
  nameTouched: false,
  projectId: null,
  adoptedProject: false,
  selected: [],
  pdfId: null,
  cadIds: [],
  guideOpts: { allowGeneration: true, attachImages: true, language: "", audience: "" },
  jobId: null,
  versionId: null,
};

const stem = (filename?: string) => (filename ?? "").replace(/\.[^.]+$/, "").trim();

function loadDraft(): WizardState {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) return INITIAL;
    const parsed = JSON.parse(raw) as Partial<WizardState>;
    const d = { ...INITIAL, ...parsed };
    if (parsed.goal === undefined) {
      // pre-goal draft: steps shift +1 (0 goal | 1 describe | 2 images |
      // 3 mode | 4 review) — a stored part run resumes at the same place
      d.goal = "part";
      d.step = Math.min((parsed.step ?? 0) + 1, PART_STEPS.length - 1);
      d.maxStep = Math.min((parsed.maxStep ?? 0) + 1, PART_STEPS.length - 1);
      d.nameTouched = !!(d.name && d.name !== deriveProjectName(d.prompt));
    }
    // progress without a job/version is a broken draft — back to the steps
    if (d.phase === "progress" && !(d.jobId && d.versionId)) {
      d.phase = "edit";
      d.step = 0;
      d.goal = d.goal || "";
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
  const [sp] = useSearchParams();
  const qc = useQueryClient();
  const [s, setS] = useState<WizardState>(() => {
    const pid = sp.get("project");
    const goal = sp.get("goal");
    const validGoal = goal === "guide" || goal === "part" ? goal : "";
    // deep link (?goal=&project= — e.g. "Neue Anleitung" on a project
    // page): fresh wizard bound to the existing project, goal pre-picked
    if (pid)
      return {
        ...INITIAL,
        projectId: pid,
        adoptedProject: true,
        goal: validGoal,
        step: validGoal ? 1 : 0,
        maxStep: validGoal ? 1 : 0,
      };
    const d = loadDraft();
    return validGoal
      ? { ...d, goal: validGoal, step: Math.max(d.step, 1), maxStep: Math.max(d.maxStep, 1) }
      : d;
  });
  const [error, setError] = useState("");
  const [creating, setCreating] = useState(false);
  // "start over" on an empty wizard-created project offers deletion via
  // ConfirmModal — same pattern as the rest of the app, no window.confirm
  const [discardPid, setDiscardPid] = useState<string | null>(null);
  const [discardBusy, setDiscardBusy] = useState(false);
  const seenAssets = useRef<Set<string>>(new Set());
  const projectIdRef = useRef<string | null>(s.projectId);
  /** in-flight createProject promise — dedupes concurrent ensureProject calls */
  const creatingRef = useRef<Promise<string> | null>(null);

  const steps = s.goal === "guide" ? GUIDE_STEPS : s.goal === "part" ? PART_STEPS : [PART_STEPS[0]];

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
          ? { ...w, phase: "edit", step: w.goal ? 1 : 0, jobId: null, versionId: null }
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

  const autoName = () =>
    s.goal === "guide"
      ? stem(pdfAsset?.filename) || deriveProjectName(s.prompt)
      : deriveProjectName(s.prompt);

  const ensureProject = async (): Promise<string> => {
    if (projectIdRef.current) return projectIdRef.current;
    if (creatingRef.current) return creatingRef.current;
    const run = (async () => {
      const name = s.name.trim() || autoName();
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

  /** entering step ≥1 requires the project to exist (uploads, image
   *  generation and asset listing all need an id). */
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
    void gotoStep(3).then(() => {
      setS((w) =>
        w.step === 3 && !w.modeTouched ? { ...w, mode: suggested } : w
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

  const pdfAsset = (assets.data ?? []).find((a) => a.id === s.pdfId);

  // adopted project: seed the untouched name field with the real project
  // name — review then shows the truth and Start can't silently rename it
  const adoptedProj = useQuery({
    queryKey: ["project", s.projectId],
    queryFn: () => api.getProject(s.projectId!),
    enabled: s.adoptedProject && !!s.projectId,
  });
  useEffect(() => {
    const p = adoptedProj.data;
    if (!p) return;
    setS((w) =>
      w.adoptedProject && !w.nameTouched && w.name !== p.name
        ? { ...w, name: p.name }
        : w
    );
  }, [adoptedProj.data]);

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

  // auto-select newly added IMAGE assets (deselected ones stay deselected;
  // pdf/cad kinds are guide inputs and pick themselves in StepGuideFiles)
  useEffect(() => {
    const list = assets.data ?? [];
    const fresh = list.filter(
      (a) =>
        !seenAssets.current.has(a.id) &&
        (a.kind === "image" || a.kind === "generated_image")
    );
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
      // adopted projects keep their name unless the user deliberately
      // retyped it — the wizard never renames someone else's project
      if (name && (!s.adoptedProject || s.nameTouched))
        await api.updateProject(pid, { name });
      if (s.goal === "guide") {
        if (!s.pdfId) throw new Error(t("gfiles_pdf_required"));
        return api.startGuide(pid, {
          pdf_asset_id: s.pdfId,
          cad_asset_ids: s.cadIds,
          prompt: s.prompt,
          allow_generation: s.guideOpts.allowGeneration,
          attach_images: s.guideOpts.attachImages,
          language: s.guideOpts.language,
          audience: s.guideOpts.audience,
        });
      }
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
      setDiscardPid(pid);
    } catch {
      /* lookup failed — keep the project */
    }
  };

  const confirmDiscard = async () => {
    if (!discardPid) return;
    setDiscardBusy(true);
    try {
      await api.deleteProject(discardPid);
      qc.invalidateQueries({ queryKey: ["projects"] });
    } catch {
      /* delete failed — project stays around empty */
    } finally {
      setDiscardBusy(false);
      setDiscardPid(null);
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
    if (pid && !s.adoptedProject) void discardEmptyDraft(pid);
  };

  const onAssetDeleted = (id: string) =>
    set({
      selected: s.selected.filter((x) => x !== id),
      pdfId: s.pdfId === id ? null : s.pdfId,
      cadIds: s.cadIds.filter((x) => x !== id),
    });

  const pickPdf = (id: string) => {
    const a = (assets.data ?? []).find((x) => x.id === id);
    set({
      pdfId: id,
      // an untouched auto name follows the manual's filename — but an
      // adopted project keeps its real name (Start can't rename it anyway)
      ...(s.nameTouched || s.adoptedProject ? {} : { name: stem(a?.filename) || s.name }),
    });
  };

  return (
    <div className="mx-auto max-w-3xl p-6 lg:p-8">
      <div className="anim-fade-in-up mb-6 text-center">
        <h1 className="text-3xl font-bold tracking-tight">{t("hero_title")}</h1>
        <p className="mt-1 text-sm text-faint">{t("hero_sub")}</p>
      </div>

      {/* stepper */}
      <div className="mb-6 flex items-center gap-1">
        {steps.map((st, i) => {
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
                      ? "bg-panel2 text-fg hover:bg-edge2"
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
              {i < steps.length - 1 && <span className="h-px flex-1 bg-edge" />}
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
            variant={s.goal === "guide" ? "guide" : "part"}
            onSelectJob={(id) => set({ jobId: id })}
            onDone={() => {
              qc.invalidateQueries({ queryKey: ["version", s.versionId] });
              qc.invalidateQueries({ queryKey: ["versions", s.projectId] });
              qc.invalidateQueries({ queryKey: ["project", s.projectId] });
              qc.invalidateQueries({ queryKey: ["projects"] });
            }}
            onAdjustPrompt={() =>
              // back to editing after a failed run — for a guide the review
              // step is the useful anchor (files/notes stay in the draft)
              set({
                phase: "edit",
                step: s.goal === "guide" ? 3 : 1,
                jobId: null,
                versionId: null,
              })
            }
            onNewPart={reset}
          />
        ) : s.step === 0 || !s.goal ? (
          <StepGoal
            busy={creating}
            error={error}
            onPick={(g) => {
              // switching branches: the reached steps don't map 1:1 — cap
              // the stepper at the step we're about to enter
              if (g !== s.goal)
                setS((w) => ({ ...w, goal: g, maxStep: Math.min(w.maxStep, 1) }));
              else set({ goal: g });
              void gotoStep(1);
            }}
          />
        ) : s.goal === "part" ? (
          s.step === 1 ? (
            <StepPrompt
              prompt={s.prompt}
              onChange={(v) => set({ prompt: v })}
              onNext={() => void gotoStep(2)}
              busy={creating}
              error={error}
            />
          ) : s.step === 2 ? (
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
              onDeleted={onAssetDeleted}
              onNext={() => gotoModeStep()}
              onBack={() => set({ step: 1 })}
            />
          ) : s.step === 3 ? (
            <StepMode
              value={s.mode}
              onChange={(m) => set({ mode: m, modeTouched: true })}
              suggested={suggested}
              hasImages={s.selected.length > 0}
              health={meshHealth.data}
              loading={meshHealth.isPending}
              onNext={() => set({ step: 4 })}
              onBack={() => set({ step: 2 })}
            />
          ) : (
            <StepReview
              name={s.name || deriveProjectName(s.prompt)}
              nameAuto={!s.nameTouched}
              onNameChange={(v) => set({ name: v, nameTouched: true })}
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
              onBack={() => set({ step: 3 })}
            />
          )
        ) : s.step === 1 ? (
          <StepGuideFiles
            projectId={s.projectId}
            projectName={s.name}
            assets={assets.data ?? []}
            pdfId={s.pdfId}
            cadIds={s.cadIds}
            onPickPdf={pickPdf}
            onToggleCad={(id) =>
              set({
                cadIds: s.cadIds.includes(id)
                  ? s.cadIds.filter((x) => x !== id)
                  : [...s.cadIds, id],
              })
            }
            onDeleted={onAssetDeleted}
            onNext={() => set({ step: 2 })}
            onBack={() => set({ step: 0 })}
          />
        ) : s.step === 2 ? (
          <StepGuideNotes
            prompt={s.prompt}
            onChange={(v) => set({ prompt: v })}
            onNext={() => set({ step: 3 })}
            onBack={() => set({ step: 1 })}
          />
        ) : (
          <StepGuideReview
            name={s.name || autoName()}
            nameAuto={!s.nameTouched}
            onNameChange={(v) => set({ name: v, nameTouched: true })}
            pdfName={pdfAsset?.filename ?? ""}
            cadCount={s.cadIds.length}
            notes={s.prompt}
            opts={s.guideOpts}
            onOptsChange={(o) => set({ guideOpts: o })}
            busy={generate.isPending}
            llmConfigured={health.data?.llm_configured}
            error={error}
            onStart={() => generate.mutate()}
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
            {s.projectId && !s.adoptedProject ? ` ${t("start_over_note")}` : ""}
          </button>
        </div>
      )}

      <div className="mt-8 text-center text-xs text-faint/70">
        {t("looking_existing")}{" "}
        <button className="text-brand-400 hover:underline" onClick={() => nav("/projects")}>
          {t("browse_projects")}
        </button>
      </div>

      <ConfirmModal
        open={!!discardPid}
        onClose={() => setDiscardPid(null)}
        onConfirm={() => void confirmDiscard()}
        busy={discardBusy}
        title={t("common:action.delete")}
        body={t("discard_draft_confirm")}
      />
    </div>
  );
}
