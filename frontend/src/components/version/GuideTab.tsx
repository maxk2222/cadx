import { Check, Pencil, RefreshCw, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, errMsg } from "../../lib/api";
import type { GuideDetail, GuideStepDetail } from "../../lib/types";
import SafeImg from "../SafeImg";
import Viewer3D from "../Viewer3D";
import { Btn, Card, Chip, Modal, Spinner } from "../ui";

const mediaUrl = (vid: string, name: string) =>
  `/api/versions/${vid}/guide/media/${encodeURIComponent(name)}`;

/** Guide tab: interactive assembly view (same 3D editor as the model tab)
 *  plus step text/image editing and one-click LLM rewrites per audience. */
export default function GuideTab({
  versionId,
  glbUrl,
}: {
  versionId: string;
  glbUrl: string | null;
}) {
  const { t } = useTranslation("version");
  const [guide, setGuide] = useState<GuideDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  if (!loaded) {
    api
      .getGuide(versionId)
      .then(setGuide)
      .catch((e) => setErr(errMsg(e)))
      .finally(() => setLoaded(true));
  }

  if (err)
    return <Card className="p-4 text-xs text-muted">{t("guide_load_failed", { err })}</Card>;
  if (!guide)
    return (
      <Card className="flex h-40 items-center justify-center">
        <Spinner />
      </Card>
    );

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <div>
        <div className="h-[420px] overflow-hidden rounded-lg border border-edge bg-bg">
          <Viewer3D url={glbUrl} />
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-muted">
          <Chip tone="zinc">{t("guide_instances", { n: guide.instances })}</Chip>
          <Chip tone="zinc">{t("guide_joints", { n: guide.joints })}</Chip>
          <Chip tone={guide.review.score >= 8 ? "green" : "amber"}>
            {t("guide_score", { n: guide.review.score })}
          </Chip>
          <span className="ml-auto">
            <RewriteBar
              versionId={versionId}
              onDone={(steps) => setGuide({ ...guide, steps })}
            />
          </span>
        </div>
        {guide.master_notes.length > 0 && (
          <Card className="mt-2 p-2">
            <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-faint">
              {t("guide_master_notes")}
            </div>
            <ul className="list-inside list-disc space-y-0.5 text-[11px] text-muted">
              {guide.master_notes.map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ul>
          </Card>
        )}
      </div>
      <div className="flex max-h-[560px] flex-col gap-2 overflow-y-auto pr-1">
        {guide.steps.map((s) => (
          <StepCard
            key={s.number}
            versionId={versionId}
            media={guide.media}
            step={s}
            onSaved={(upd) =>
              setGuide({
                ...guide,
                steps: guide.steps.map((x) => (x.number === upd.number ? upd : x)),
              })
            }
          />
        ))}
        {!guide.steps.length && (
          <div className="p-4 text-center text-xs text-muted">{t("guide_no_steps")}</div>
        )}
      </div>
    </div>
  );
}

function RewriteBar({
  versionId,
  onDone,
}: {
  versionId: string;
  onDone: (steps: GuideStepDetail[]) => void;
}) {
  const { t } = useTranslation("version");
  const [audience, setAudience] = useState<"beginner" | "pro">("beginner");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  return (
    <span className="flex items-center gap-1.5">
      <select
        className="input py-0.5 text-[11px]"
        value={audience}
        onChange={(e) => setAudience(e.target.value as "beginner" | "pro")}
        aria-label={t("guide_audience")}
      >
        <option value="beginner">{t("guide_audience_beginner")}</option>
        <option value="pro">{t("guide_audience_pro")}</option>
      </select>
      <Btn
        size="sm"
        busy={busy}
        title={err ?? t("guide_rewrite_hint")}
        onClick={() => {
          setBusy(true);
          setErr(null);
          api
            .rewriteGuide(versionId, audience)
            .then((r) => onDone(r.steps))
            .catch((e) => setErr(errMsg(e)))
            .finally(() => setBusy(false));
        }}
      >
        <RefreshCw className="h-3 w-3" /> {t("guide_rewrite")}
      </Btn>
    </span>
  );
}

function StepCard({
  versionId,
  step,
  media,
  onSaved,
}: {
  versionId: string;
  step: GuideStepDetail;
  media: string[];
  onSaved: (s: GuideStepDetail) => void;
}) {
  const { t } = useTranslation("version");
  const [edit, setEdit] = useState(false);
  const [busy, setBusy] = useState(false);
  const [picker, setPicker] = useState(false);
  const [draft, setDraft] = useState(step);
  const img = useMemo(
    () => (step.media_image ? mediaUrl(versionId, step.media_image) : null),
    [versionId, step.media_image],
  );

  const save = () => {
    setBusy(true);
    api
      .patchGuideStep(versionId, step.number, {
        title: draft.title,
        text: draft.text,
        display_text: draft.display_text,
        safety_hint: draft.safety_hint,
      })
      .then(onSaved)
      .catch(() => {})
      .finally(() => {
        setBusy(false);
        setEdit(false);
      });
  };

  const pick = (name: string | null) => {
    setPicker(false);
    api
      .patchGuideStep(versionId, step.number, { media_image: name ?? "" })
      .then(onSaved)
      .catch(() => {});
  };

  if (!edit)
    return (
      <Card className="p-2.5">
        <div className="flex items-start gap-2">
          <Chip tone="blue" className="mt-0.5 shrink-0">
            {step.number}
          </Chip>
          <div className="min-w-0 flex-1">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate text-xs font-medium text-fg">
                {step.title || step.display_text}
              </span>
              <span className="flex shrink-0 items-center gap-1">
                {step.action_type && (
                  <Chip tone="zinc" className="text-[9px]">
                    {step.action_type}
                  </Chip>
                )}
                <button
                  className="rounded p-1 text-faint hover:bg-panel2 hover:text-fg"
                  onClick={() => {
                    setDraft(step);
                    setEdit(true);
                  }}
                  aria-label={t("guide_edit_step")}
                >
                  <Pencil className="h-3 w-3" />
                </button>
              </span>
            </div>
            <p className="mt-0.5 text-[11px] leading-snug text-muted">{step.text}</p>
            {step.display_text && (
              <p className="mt-1 rounded bg-panel2 px-1.5 py-0.5 text-[10px] text-brand-300">
                AR: {step.display_text}
              </p>
            )}
            {step.safety_hint && (
              <p className="mt-0.5 text-[10px] text-amber-400">⚠ {step.safety_hint}</p>
            )}
          </div>
          {img && (
            <button
              className="h-14 w-14 shrink-0 overflow-hidden rounded border border-edge2 hover:border-brand-500"
              onClick={() => setPicker(true)}
              aria-label={t("guide_pick_media")}
            >
              <SafeImg
                src={img}
                alt={step.media_image ?? ""}
                className="h-full w-full object-cover"
                fallback={<div className="h-full w-full bg-panel2" />}
              />
            </button>
          )}
        </div>
        {picker && (
          <Modal open title={t("guide_pick_media")} onClose={() => setPicker(false)}>
            <div className="grid max-h-[50vh] grid-cols-4 gap-2 overflow-y-auto p-1">
              <button
                className="flex h-16 items-center justify-center rounded border border-edge2 text-[10px] text-muted hover:border-brand-500"
                onClick={() => pick(null)}
              >
                {t("guide_no_image")}
              </button>
              {media.map((m) => (
                <button
                  key={m}
                  className={
                    "overflow-hidden rounded border " +
                    (m === step.media_image
                      ? "border-brand-500 ring-1 ring-brand-500"
                      : "border-edge2 hover:border-brand-500")
                  }
                  onClick={() => pick(m)}
                >
                  <SafeImg
                    src={mediaUrl(versionId, m)}
                    alt={m}
                    className="h-16 w-full object-cover"
                    fallback={<div className="h-16 bg-panel2" />}
                  />
                </button>
              ))}
            </div>
          </Modal>
        )}
      </Card>
    );

  return (
    <Card className="border-brand-800/70 p-2.5">
      <div className="flex items-center gap-2">
        <Chip tone="blue">{step.number}</Chip>
        <input
          className="input flex-1 py-0.5 text-xs"
          value={draft.title}
          placeholder={t("guide_title_ph")}
          onChange={(e) => setDraft({ ...draft, title: e.target.value })}
        />
        <button
          className="rounded p-1 text-faint hover:text-fg disabled:opacity-40"
          disabled={busy}
          onClick={save}
          aria-label={t("guide_save")}
        >
          {busy ? <Spinner className="h-3 w-3" /> : <Check className="h-3 w-3" />}
        </button>
        <button
          className="rounded p-1 text-faint hover:text-fg"
          onClick={() => setEdit(false)}
          aria-label={t("guide_cancel")}
        >
          <X className="h-3 w-3" />
        </button>
      </div>
      <textarea
        className="input mt-1.5 w-full text-[11px]"
        rows={3}
        value={draft.text}
        placeholder={t("guide_text_ph")}
        onChange={(e) => setDraft({ ...draft, text: e.target.value })}
      />
      <div className="mt-1.5 grid grid-cols-2 gap-1.5">
        <input
          className="input py-0.5 text-[11px]"
          value={draft.display_text}
          placeholder={t("guide_display_ph")}
          onChange={(e) => setDraft({ ...draft, display_text: e.target.value })}
        />
        <input
          className="input py-0.5 text-[11px]"
          value={draft.safety_hint}
          placeholder={t("guide_safety_ph")}
          onChange={(e) => setDraft({ ...draft, safety_hint: e.target.value })}
        />
      </div>
    </Card>
  );
}
