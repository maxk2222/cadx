import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../lib/api";
import type { Project } from "../lib/types";
import { Btn, Field, Modal, inputCls } from "./ui";

export default function ProjectFormModal({
  onClose,
  project,
  onCreated,
}: {
  onClose: () => void;
  project?: Project;
  onCreated?: (p: Project) => void;
}) {
  const { t } = useTranslation("projects");
  const qc = useQueryClient();
  const [name, setName] = useState(project?.name ?? "");
  const [desc, setDesc] = useState(project?.description ?? "");

  const save = useMutation({
    mutationFn: () =>
      project
        ? api.updateProject(project.id, { name: name.trim(), description: desc.trim() })
        : api.createProject(name.trim(), desc.trim()),
    onSuccess: (p) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      qc.invalidateQueries({ queryKey: ["project", p.id] });
      onClose();
      onCreated?.(p);
    },
  });

  return (
    <Modal open onClose={onClose} title={project ? t("modal.edit_title") : t("modal.new_title")}>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (name.trim()) save.mutate();
        }}
      >
        <Field label={t("common:field.name")}>
          <input
            className={inputCls}
            value={name}
            autoFocus
            placeholder={t("modal.name_ph")}
            onChange={(e) => setName(e.target.value)}
          />
        </Field>
        <Field label={t("common:field.description")} hint={t("common:field.optional")}>
          <textarea
            className={`${inputCls} h-20 resize-none`}
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
          />
        </Field>
        <div className="flex justify-end gap-2">
          <Btn variant="ghost" type="button" onClick={onClose}>
            {t("common:action.cancel")}
          </Btn>
          <Btn type="submit" busy={save.isPending} disabled={!name.trim()}>
            {project ? t("common:action.save") : t("common:action.create")}
          </Btn>
        </div>
        {save.isError && <div className="mt-2 text-sm text-red-400">{String(save.error)}</div>}
      </form>
    </Modal>
  );
}
