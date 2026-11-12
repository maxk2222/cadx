import { ChevronLeft, ChevronRight, PencilRuler } from "lucide-react";
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
          {t("guide_score", { n: guide.review.score })}
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
