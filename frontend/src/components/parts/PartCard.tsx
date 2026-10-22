import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, CircleDot, Pencil, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, errMsg } from "../../lib/api";
import type { PartGroup } from "../../lib/parts";
import VersionTree from "../VersionTree";
import {
  Card,
  Chip,
  ConfirmModal,
  OverflowMenu,
  inputCls,
  useToast,
} from "../ui";

/** One part card on the project page: collapsible header (P{n} chip +
 *  inline-renamable name + count) over the part's own version tree. Custom
 *  header instead of SectionCollapsible because the name is primary content
 *  (not an uppercase section label) and the OverflowMenu must not nest
 *  inside the collapse toggle. Open state persists via the same
 *  `collapse:` localStorage convention as Collapsible.
 *  Deleting a part cascades its whole subtree — ConfirmModal spells out the
 *  version count; the backend still 409s on running jobs. */
export default function PartCard({
  group,
  projectId,
  activeId,
  forceOpen,
  startEditing,
}: {
  group: PartGroup;
  projectId: string;
  /** currently viewed version — highlighted in the tree */
  activeId?: string;
  /** ?part=<id> deep-link — scrolls into view + expands once. */
  forceOpen?: boolean;
  /** open the name editor immediately (freshly created part). */
  startEditing?: boolean;
}) {
  const { t } = useTranslation("projects");
  const qc = useQueryClient();
  const toast = useToast();
  const { part, versions, active } = group;
  const persistKey = `collapse:part:${part.id}`;
  const [open, setOpen] = useState(() => {
    try {
      const s = localStorage.getItem(persistKey);
      if (s != null) return s === "1";
    } catch {
      /* private mode */
    }
    return true;
  });
  const [nameEdit, setNameEdit] = useState(!!startEditing);
  const [nameDraft, setNameDraft] = useState(part.name);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (forceOpen) {
      setOpen(true);
      ref.current?.scrollIntoView({ block: "nearest" });
    }
  }, [forceOpen]);

  const toggle = () => {
    setOpen((o) => {
      const n = !o;
      try {
        localStorage.setItem(persistKey, n ? "1" : "0");
      } catch {
        /* private mode */
      }
      return n;
    });
  };

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["parts", projectId] });
    qc.invalidateQueries({ queryKey: ["versions", projectId] });
    qc.invalidateQueries({ queryKey: ["project", projectId] });
    qc.invalidateQueries({ queryKey: ["projects"] });
  };

  const rename = useMutation({
    mutationFn: (name: string) => api.updatePart(part.id, name),
    onSuccess: invalidate,
    onError: (e) => toast(errMsg(e), "error"),
  });

  const del = useMutation({
    mutationFn: () => api.deletePart(part.id),
    onSuccess: (r) => {
      invalidate();
      toast(t("part_deleted", { count: r.versions_deleted }), "success");
    },
    onError: (e) => toast(errMsg(e), "error"),
  });

  const commitName = () => {
    setNameEdit(false);
    const name = nameDraft.trim();
    if (name !== part.name) rename.mutate(name);
  };

  // synthetic grouping for versions whose part row is unknown (fetch skew)
  const synthetic = part.number === 0;

  return (
    <div ref={ref}>
      <Card className="p-3">
        <div className="mb-1 flex items-center gap-2">
          <button
            type="button"
            onClick={toggle}
            aria-expanded={open}
            className="group -m-1 rounded p-1 text-left"
          >
            <ChevronDown
              className={`h-3.5 w-3.5 text-faint transition-transform duration-200 group-hover:text-muted ${
                open ? "" : "-rotate-90"
              }`}
            />
          </button>
          <Chip tone={synthetic ? "zinc" : "blue"} className="shrink-0">
            {synthetic ? "P?" : `P${part.number}`}
          </Chip>
          {nameEdit ? (
            <input
              autoFocus
              className={`${inputCls} w-44 py-0.5 text-sm`}
              placeholder={t("part_name_ph")}
              maxLength={200}
              onChange={(e) => setNameDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") e.currentTarget.blur();
                if (e.key === "Escape") setNameEdit(false);
              }}
              onBlur={commitName}
            />
          ) : (
            <button
              type="button"
              onClick={toggle}
              className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
              title={part.name || t("part_untitled")}
            >
              <span className="truncate text-sm font-medium text-fg">
                {part.name || (
                  <span className="italic text-faint">{t("part_untitled")}</span>
                )}
              </span>
            </button>
          )}
          <span className="shrink-0 text-[11px] text-faint/70">
            {synthetic ? versions.length : part.version_count}
          </span>
          <span className="flex-1" />
          {part.thumbnail_url && (
            <img
              src={part.thumbnail_url}
              alt=""
              className="h-7 w-7 shrink-0 rounded border border-edge object-contain"
            />
          )}
          {active && (
            <CircleDot className="h-3.5 w-3.5 shrink-0 animate-pulse text-brand-400" />
          )}
          {!synthetic && (
            <>
              <button
                type="button"
                title={t("part_rename_title")}
                onClick={() => {
                  setNameDraft(part.name);
                  setNameEdit(true);
                }}
                className="rounded p-1 text-faint/60 transition-colors hover:bg-panel2 hover:text-muted"
              >
                <Pencil className="h-3 w-3" />
              </button>
              <OverflowMenu
                title={t("part_actions")}
                items={[
                  {
                    icon: <Pencil className="h-3.5 w-3.5" />,
                    label: t("common:action.rename"),
                    onClick: () => {
                      setNameDraft(part.name);
                      setNameEdit(true);
                    },
                  },
                  {
                    icon: <Trash2 className="h-3.5 w-3.5" />,
                    label: t("common:action.delete"),
                    danger: true,
                    onClick: () => setDeleteOpen(true),
                  },
                ]}
              />
            </>
          )}
        </div>
        <div
          className={`grid transition-[grid-template-rows] duration-200 ease-out ${
            open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
          }`}
        >
          <div className="min-h-0 overflow-hidden">
            {versions.length === 0 ? (
              <p className="px-2 py-3 text-xs italic text-faint">
                {t("part_empty")}
              </p>
            ) : (
              <VersionTree versions={versions} activeId={activeId} />
            )}
          </div>
        </div>
      </Card>

      <ConfirmModal
        open={deleteOpen}
        onClose={() => setDeleteOpen(false)}
        onConfirm={() => del.mutate()}
        busy={del.isPending}
        title={t("part_delete_title")}
        body={t("part_delete_body", {
          name: part.name || `P${part.number}`,
          // server total — the filtered tree may show fewer
          count: part.version_count,
        })}
      />
    </div>
  );
}
