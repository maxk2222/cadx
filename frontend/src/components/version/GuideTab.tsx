import { ChevronLeft, ChevronRight, PencilRuler, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, errMsg } from "../../lib/api";
import type { GuideDetail } from "../../lib/types";
import SafeImg from "../SafeImg";
import GuideEditor from "./GuideEditor";
import GuideScene from "./GuideScene";
import { Btn, Card, Chip, Spinner } from "../ui";

const mediaUrl = (vid: string, name: string) =>
  `/api/versions/${vid}/guide/media/${encodeURIComponent(name)}`;

/** Guide overview: read-only step-scrubbed preview + meta chips; the real
 *  KIARA-style editor opens in a modal (GuideEditor). */
export default function GuideTab({
  versionId,
  name,
}: {
  versionId: string;
  name: string;
}) {
  const { t } = useTranslation("version");
  const [guide, setGuide] = useState<GuideDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [cur, setCur] = useState(0);
  const [open, setOpen] = useState(false);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    api
      .getGuide(versionId)
      .then(setGuide)
      .catch((e) => setErr(errMsg(e)))
      .finally(() => setLoaded(true));
  }, [versionId, reload]);

  const steps = useMemo(
    () => [...(guide?.steps ?? [])].sort((a, b) => a.number - b.number),
    [guide],
  );
  const step = steps[Math.min(cur, Math.max(0, steps.length - 1))];

  if (err)
    return (
      <Card className="p-4 text-xs text-muted">
        {t("guide_load_failed", { err })}
      </Card>
    );
  if (!loaded || !guide)
    return (
      <Card className="flex h-40 items-center justify-center">
        <Spinner />
      </Card>
    );

  const img = step?.media_image ? mediaUrl(versionId, step.media_image) : null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Btn size="sm" onClick={() => setOpen(true)}>
          <PencilRuler className="h-3.5 w-3.5" /> {t("guide_open_editor")}
        </Btn>
        <Chip tone="zinc">{t("guide_instances", { n: guide.instances })}</Chip>
        <Chip tone="zinc">{t("guide_joints", { n: guide.joints })}</Chip>
        <Chip tone={guide.review.score >= 8 ? "green" : "amber"}>
          {t("guide_score", { n: guide.review.score })}/10
        </Chip>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr_300px]">
        <div className="h-[560px] overflow-hidden rounded-lg border border-edge bg-bg">
          <GuideScene
            versionId={versionId}
            parts={guide.parts}
            instances={guide.instance_list ?? []}
            step={step?.number ?? 0}
            sel={null}
            onSelect={() => {}}
            onTransformed={() => {}}
            readonly
            annotations={guide.annotations ?? []}
          />
        </div>
        <Card className="flex h-fit flex-col gap-2 p-3">
          <div className="flex items-center justify-between">
            <Btn
              size="sm"
              disabled={cur === 0}
              onClick={() => setCur((c) => Math.max(0, c - 1))}
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </Btn>
            <span className="text-xs text-fg">
              {t("guide_step_of", { n: step?.number ?? 0, m: steps.length })}
            </span>
            <Btn
              size="sm"
              disabled={cur >= steps.length - 1}
              onClick={() => setCur((c) => Math.min(steps.length - 1, c + 1))}
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </Btn>
          </div>
          {step && (
            <>
              <div className="text-xs font-medium text-fg">{step.title}</div>
              <p className="whitespace-pre-wrap text-[11px] leading-relaxed text-muted">
                {step.display_text || step.text}
              </p>
              {step.required_parts?.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {step.required_parts.map((p) => (
                    <Chip key={p} tone="zinc" className="text-[10px]">
                      {p}
                    </Chip>
                  ))}
                </div>
              )}
              {step.safety_hint && (
                <div className="rounded border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[11px] text-amber-300">
                  {step.safety_hint}
                </div>
              )}
              {img && (
                <SafeImg
                  src={img}
                  alt={step.media_image ?? ""}
                  className="w-full rounded border border-edge2 object-contain"
                  fallback={<div className="h-24 bg-panel2" />}
                />
              )}
            </>
          )}
        </Card>
      </div>

      <VariantPanel
        versionId={versionId}
        variants={guide.variants ?? {}}
        onChanged={() => setReload((r) => r + 1)}
      />

      {guide.review.findings.length > 0 && (
        <Card className="p-3">
          <div className="mb-1.5 flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-faint">
            {t("guide_findings")}
            <span className="text-muted normal-case">
              {t("guide_review_iter", { n: guide.review.iteration })}
            </span>
          </div>
          <ul className="space-y-1 text-[11px]">
            {guide.review.findings.map((f, i) => (
              <li key={i} className="flex gap-2">
                <Chip
                  tone={
                    f.severity === "blocker"
                      ? "red"
                      : f.severity === "warning"
                        ? "amber"
                        : "zinc"
                  }
                  className="shrink-0"
                >
                  {f.severity}
                </Chip>
                <span className="text-muted">{f.text}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {guide.master_notes.length > 0 && (
        <Card className="p-3">
          <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-faint">
            {t("guide_notes")}
          </div>
          <ul className="list-inside list-disc space-y-0.5 text-[11px] text-muted">
            {guide.master_notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </Card>
      )}

      {open && (
        <GuideEditor
          versionId={versionId}
          name={name}
          onClose={() => {
            setOpen(false);
            setReload((r) => r + 1);
          }}
        />
      )}
    </div>
  );
}

const LANGS = ["de", "en", "fr", "es", "it"] as const;

function variantLabel(key: string, t: (k: string, o?: Record<string, unknown>) => string): string {
  if (key === "_original") return t("guide_variant_original");
  const [aud, lang] = key.split(":");
  const audLabel =
    aud === "beginner" || aud === "intermediate" || aud === "pro"
      ? t(`guide_audience_${aud}`)
      : aud;
  const langLabel = LANGS.includes(lang as (typeof LANGS)[number])
    ? t(`guide_lang_${lang}`)
    : lang;
  return `${audLabel} · ${langLabel}`;
}

/** Text variants: which rewrite the guide ships + LLM rewrite controls.
 *  Lives on the overview (not inside the 3D editor) — variants are
 *  document-level, not scene-level. */
function VariantPanel({
  versionId,
  variants,
  onChanged,
}: {
  versionId: string;
  variants: Record<string, unknown>;
  onChanged: () => void;
}) {
  const { t } = useTranslation("version");
  const [audience, setAudience] = useState("beginner");
  const [lang, setLang] = useState("de");
  const [langFree, setLangFree] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const keys = useMemo(() => Object.keys(variants).sort(), [variants]);
  const effLang = lang === "other" ? langFree.trim() : lang;

  return (
    <Card className="flex flex-col gap-2 p-3">
      <div className="text-[10px] font-medium uppercase tracking-wider text-faint">
        {t("guide_variant")}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {keys.length > 0 && (
          <select
            className="input py-1 text-[11px]"
            defaultValue=""
            title={t("guide_variant_hint")}
            onChange={(e) => {
              const k = e.target.value;
              if (!k) return;
              setBusy(true);
              setErr(null);
              api
                .applyGuideVariant(versionId, k)
                .then(onChanged)
                .catch((e2) => setErr(errMsg(e2)))
                .finally(() => setBusy(false));
            }}
          >
            <option value="" disabled>
              {t("guide_variant")}…
            </option>
            {keys.map((k) => (
              <option key={k} value={k}>
                {variantLabel(k, t)}
              </option>
            ))}
          </select>
        )}
        <select
          className="input py-1 text-[11px]"
          value={audience}
          onChange={(e) => setAudience(e.target.value)}
          aria-label={t("guide_audience")}
        >
          <option value="beginner">{t("guide_audience_beginner")}</option>
          <option value="intermediate">{t("guide_audience_intermediate")}</option>
          <option value="pro">{t("guide_audience_pro")}</option>
        </select>
        <select
          className="input py-1 text-[11px]"
          value={lang}
          onChange={(e) => setLang(e.target.value)}
          aria-label={t("guide_language_ph")}
        >
          {LANGS.map((l) => (
            <option key={l} value={l}>
              {t(`guide_lang_${l}`)}
            </option>
          ))}
          <option value="other">{t("guide_lang_other")}</option>
        </select>
        {lang === "other" && (
          <input
            className="input w-24 py-1 text-[11px]"
            value={langFree}
            onChange={(e) => setLangFree(e.target.value)}
            placeholder={t("guide_language_ph")}
          />
        )}
        <Btn
          size="sm"
          busy={busy}
          title={err ?? t("guide_rewrite_hint")}
          onClick={() => {
            setBusy(true);
            setErr(null);
            api
              .rewriteGuide(versionId, audience, effLang)
              .then(onChanged)
              .catch((e2) => setErr(errMsg(e2)))
              .finally(() => setBusy(false));
          }}
        >
          <RefreshCw className="h-3 w-3" /> {t("guide_rewrite")}
        </Btn>
      </div>
      {err && <div className="text-[11px] text-red-400">{err}</div>}
    </Card>
  );
}
