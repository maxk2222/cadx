import { Check, ChevronLeft, ChevronRight, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import * as THREE from "three";
import { api, errMsg } from "../../lib/api";
import type { GuideDetail, GuideStepDetail } from "../../lib/types";
import SafeImg from "../SafeImg";
import GuideScene from "./GuideScene";
import { Btn, Card, Chip, Modal, Spinner } from "../ui";

const mediaUrl = (vid: string, name: string) =>
  `/api/versions/${vid}/guide/media/${encodeURIComponent(name)}`;

/** Kiara-style guide editor: left sidebar with step navigation + inline
 *  step editing, centre 3D scene (cumulative build, new parts highlighted,
 *  gizmo transforms), right inspector with numeric transform fields.
 *  Step switches autosave dirty fields — same flow as the KIARA editor. */
export default function GuideTab({ versionId }: { versionId: string }) {
  const { t } = useTranslation("version");
  const [guide, setGuide] = useState<GuideDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [cur, setCur] = useState(0);
  const [sel, setSel] = useState<string | null>(null);
  const [draft, setDraft] = useState<GuideStepDetail | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [picker, setPicker] = useState(false);

  useEffect(() => {
    api
      .getGuide(versionId)
      .then(setGuide)
      .catch((e) => setErr(errMsg(e)))
      .finally(() => setLoaded(true));
  }, [versionId]);

  const steps = useMemo(
    () => [...(guide?.steps ?? [])].sort((a, b) => a.number - b.number),
    [guide],
  );
  const step = steps[Math.min(cur, Math.max(0, steps.length - 1))];

  // sync draft whenever the current step changes (and is not being edited)
  useEffect(() => {
    if (step && !dirty) setDraft(step);
  }, [step, dirty]);

  const saveStep = async (s: GuideStepDetail) => {
    setSaving(true);
    try {
      const upd = await api.patchGuideStep(versionId, s.number, {
        title: s.title,
        text: s.text,
        display_text: s.display_text,
        safety_hint: s.safety_hint,
      });
      setGuide((g) =>
        g
          ? { ...g, steps: g.steps.map((x) => (x.number === upd.number ? upd : x)) }
          : g,
      );
      setDirty(false);
    } finally {
      setSaving(false);
    }
  };

  const goto = (idx: number) => {
    const next = Math.max(0, Math.min(steps.length - 1, idx));
    if (dirty && draft) void saveStep(draft); // Kiara autosave on step switch
    setCur(next);
    setSel(null);
  };

  if (err)
    return <Card className="p-4 text-xs text-muted">{t("guide_load_failed", { err })}</Card>;
  if (!loaded || !guide)
    return (
      <Card className="flex h-40 items-center justify-center">
        <Spinner />
      </Card>
    );

  const img = draft?.media_image ? mediaUrl(versionId, draft.media_image) : null;

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-[240px_1fr_210px]">
      {/* ——— left sidebar: steps + text editing ——— */}
      <div className="flex flex-col gap-2">
        <Card className="p-2.5">
          <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-faint">
            {t("guide_steps")}
          </div>
          <div className="flex items-center justify-between gap-1">
            <Btn size="sm" disabled={cur === 0} onClick={() => goto(cur - 1)}>
              <ChevronLeft className="h-3.5 w-3.5" />
            </Btn>
            <span className="text-xs text-fg">
              {t("guide_step_of", { n: step?.number ?? 0, m: steps.length })}
            </span>
            <Btn size="sm" disabled={cur >= steps.length - 1} onClick={() => goto(cur + 1)}>
              <ChevronRight className="h-3.5 w-3.5" />
            </Btn>
          </div>
          {draft && step && (
            <div className="mt-2 flex flex-col gap-1.5">
              <input
                className="input py-1 text-xs"
                value={draft.title}
                placeholder={t("guide_title_ph")}
                onChange={(e) => {
                  setDraft({ ...draft, title: e.target.value });
                  setDirty(true);
                }}
              />
              <textarea
                className="input w-full text-[11px]"
                rows={7}
                value={draft.text}
                placeholder={t("guide_text_ph")}
                onChange={(e) => {
                  setDraft({ ...draft, text: e.target.value });
                  setDirty(true);
                }}
              />
              <input
                className="input py-1 text-[11px]"
                value={draft.display_text}
                placeholder={t("guide_display_ph")}
                onChange={(e) => {
                  setDraft({ ...draft, display_text: e.target.value });
                  setDirty(true);
                }}
              />
              <input
                className="input py-1 text-[11px]"
                value={draft.safety_hint}
                placeholder={t("guide_safety_ph")}
                onChange={(e) => {
                  setDraft({ ...draft, safety_hint: e.target.value });
                  setDirty(true);
                }}
              />
              <div className="flex items-center gap-2">
                <Btn size="sm" busy={saving} disabled={!dirty} onClick={() => saveStep(draft)}>
                  <Check className="h-3 w-3" /> {t("guide_save")}
                </Btn>
                {img && (
                  <button
                    className="h-11 w-11 overflow-hidden rounded border border-edge2 hover:border-brand-500"
                    onClick={() => setPicker(true)}
                    aria-label={t("guide_pick_media")}
                  >
                    <SafeImg
                      src={img}
                      alt={draft.media_image ?? ""}
                      className="h-full w-full object-cover"
                      fallback={<div className="h-full w-full bg-panel2" />}
                    />
                  </button>
                )}
                {!img && (
                  <Btn size="sm" onClick={() => setPicker(true)}>
                    {t("guide_pick_media")}
                  </Btn>
                )}
              </div>
            </div>
          )}
        </Card>

        <Card className="p-2.5">
          <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-faint">
            {t("guide_rewrite")}
          </div>
          <RewriteBar
            versionId={versionId}
            onDone={(steps) =>
              setGuide({ ...guide, steps: steps.map((s) => ({ ...s })) })
            }
          />
        </Card>

        <Card className="flex flex-wrap gap-1.5 p-2.5 text-[11px] text-muted">
          <Chip tone="zinc">{t("guide_instances", { n: guide.instances })}</Chip>
          <Chip tone="zinc">{t("guide_joints", { n: guide.joints })}</Chip>
          <Chip tone={guide.review.score >= 8 ? "green" : "amber"}>
            {t("guide_score", { n: guide.review.score })}
          </Chip>
        </Card>
      </div>

      {/* ——— centre: 3D scene ——— */}
      <div className="h-[560px] overflow-hidden rounded-lg border border-edge bg-bg">
        <GuideScene
          versionId={versionId}
          parts={guide.parts}
          instances={guide.instance_list ?? []}
          step={step?.number ?? 0}
          sel={sel}
          onSelect={setSel}
          onTransformed={(iid, m) => {
            api
              .patchGuideInstance(versionId, iid, m)
              .then((r) =>
                setGuide({
                  ...guide,
                  instance_list: guide.instance_list.map((i) =>
                    i.id === iid ? { ...i, transform: r.transform } : i,
                  ),
                }),
              )
              .catch((e) => setErr(errMsg(e)));
          }}
        />
      </div>

      {/* ——— right: inspector ——— */}
      <Inspector
        guide={guide}
        sel={sel}
        onApply={(iid, m) => {
          api
            .patchGuideInstance(versionId, iid, m)
            .then((r) =>
              setGuide({
                ...guide,
                instance_list: guide.instance_list.map((i) =>
                  i.id === iid ? { ...i, transform: r.transform } : i,
                ),
              }),
            )
            .catch((e) => setErr(errMsg(e)));
        }}
      />

      {picker && step && (
        <Modal open title={t("guide_pick_media")} onClose={() => setPicker(false)}>
          <div className="grid max-h-[50vh] grid-cols-4 gap-2 overflow-y-auto p-1">
            <button
              className="flex h-16 items-center justify-center rounded border border-edge2 text-[10px] text-muted hover:border-brand-500"
              onClick={() => {
                setPicker(false);
                void api
                  .patchGuideStep(versionId, step.number, { media_image: "" })
                  .then((upd) =>
                    setGuide({
                      ...guide,
                      steps: guide.steps.map((x) => (x.number === upd.number ? upd : x)),
                    }),
                  );
              }}
            >
              {t("guide_no_image")}
            </button>
            {guide.media.map((m) => (
              <button
                key={m}
                className={
                  "overflow-hidden rounded border " +
                  (m === step.media_image
                    ? "border-brand-500 ring-1 ring-brand-500"
                    : "border-edge2 hover:border-brand-500")
                }
                onClick={() => {
                  setPicker(false);
                  void api
                    .patchGuideStep(versionId, step.number, { media_image: m })
                    .then((upd) =>
                      setGuide({
                        ...guide,
                        steps: guide.steps.map((x) => (x.number === upd.number ? upd : x)),
                      }),
                    );
                }}
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
  const [audience, setAudience] = useState("beginner");
  const [lang, setLang] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  return (
    <span className="flex flex-wrap items-center gap-1.5">
      <select
        className="input py-0.5 text-[11px]"
        value={audience}
        onChange={(e) => setAudience(e.target.value)}
        aria-label={t("guide_audience")}
      >
        <option value="beginner">{t("guide_audience_beginner")}</option>
        <option value="intermediate">{t("guide_audience_intermediate")}</option>
        <option value="pro">{t("guide_audience_pro")}</option>
      </select>
      <input
        className="input w-24 py-0.5 text-[11px]"
        value={lang}
        onChange={(e) => setLang(e.target.value)}
        placeholder={t("guide_language_ph")}
        aria-label={t("guide_language_ph")}
      />
      <Btn
        size="sm"
        busy={busy}
        title={err ?? t("guide_rewrite_hint")}
        onClick={() => {
          setBusy(true);
          setErr(null);
          api
            .rewriteGuide(versionId, audience, lang)
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

/** Right-hand inspector: identity of the selected instance plus numeric
 *  position/rotation (Euler deg) fields that PATCH the transform. */
function Inspector({
  guide,
  sel,
  onApply,
}: {
  guide: GuideDetail;
  sel: string | null;
  onApply: (iid: string, m: number[][]) => void;
}) {
  const { t } = useTranslation("version");
  const inst = guide.instance_list?.find((i) => i.id === sel);
  const [f, setF] = useState({ px: 0, py: 0, pz: 0, rx: 0, ry: 0, rz: 0 });
  const selRef = useRef<string | null>(null);

  useEffect(() => {
    if (!inst || selRef.current === inst.id) return;
    selRef.current = inst.id;
    const m = new THREE.Matrix4();
    if (inst.transform?.length === 4)
      m.set(...(inst.transform.flat() as Parameters<THREE.Matrix4["set"]>));
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    m.decompose(p, q, s);
    const e = new THREE.Euler().setFromQuaternion(q);
    setF({
      px: +p.x.toFixed(2),
      py: +p.y.toFixed(2),
      pz: +p.z.toFixed(2),
      rx: +THREE.MathUtils.radToDeg(e.x).toFixed(1),
      ry: +THREE.MathUtils.radToDeg(e.y).toFixed(1),
      rz: +THREE.MathUtils.radToDeg(e.z).toFixed(1),
    });
  }, [inst]);

  const num = (k: keyof typeof f, label: string) => (
    <label className="flex items-center gap-1 text-[10px] text-faint" key={k}>
      <span className="w-3">{label}</span>
      <input
        type="number"
        step={k.startsWith("r") ? 5 : 1}
        className="input w-full py-0.5 text-[11px]"
        value={f[k]}
        onChange={(e) => setF({ ...f, [k]: Number(e.target.value) })}
      />
    </label>
  );

  return (
    <Card className="flex h-fit flex-col gap-2 p-2.5">
      <div className="text-[10px] font-medium uppercase tracking-wider text-faint">
        {t("guide_inspector")}
      </div>
      {!inst ? (
        <p className="text-[11px] text-muted">{t("guide_pick_instance")}</p>
      ) : (
        <>
          <div className="text-[11px]">
            <div className="font-medium text-fg">{inst.id}</div>
            <div className="text-muted">
              {guide.parts.find((p) => p.id === inst.part_id)?.name ?? inst.part_id}
            </div>
            <div className="text-faint">{t("guide_step_n", { n: inst.placed_in_step })}</div>
          </div>
          <div className="grid grid-cols-1 gap-1">
            <div className="text-[10px] uppercase text-faint">{t("guide_pos")}</div>
            {num("px", "X")}
            {num("py", "Y")}
            {num("pz", "Z")}
            <div className="mt-1 text-[10px] uppercase text-faint">{t("guide_rot")}</div>
            {num("rx", "X")}
            {num("ry", "Y")}
            {num("rz", "Z")}
          </div>
          <Btn
            size="sm"
            onClick={() => {
              const m = new THREE.Matrix4().compose(
                new THREE.Vector3(f.px, f.py, f.pz),
                new THREE.Quaternion().setFromEuler(
                  new THREE.Euler(
                    THREE.MathUtils.degToRad(f.rx),
                    THREE.MathUtils.degToRad(f.ry),
                    THREE.MathUtils.degToRad(f.rz),
                  ),
                ),
                new THREE.Vector3(1, 1, 1),
              );
              const e = m.elements;
              onApply(inst.id, [
                [e[0], e[4], e[8], e[12]],
                [e[1], e[5], e[9], e[13]],
                [e[2], e[6], e[10], e[14]],
                [e[3], e[7], e[11], e[15]],
              ]);
            }}
          >
            {t("guide_apply")}
          </Btn>
        </>
      )}
    </Card>
  );
}
