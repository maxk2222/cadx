import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Boxes,
  MessageSquarePlus,
  Square,
  Trash2,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import Composer from "../components/chat/Composer";
import MessageBubble from "../components/chat/MessageBubble";
import LivePreview from "../components/LivePreview";
import Viewer3D from "../components/Viewer3D";
import {
  Btn,
  Card,
  ConfirmModal,
  Crumbs,
  EmptyState,
  IconBtn,
  SectionTitle,
  Spinner,
} from "../components/ui";
import { api } from "../lib/api";
import type { Asset } from "../lib/types";
import { useChatEvents } from "../lib/useChatEvents";
import { useTitle } from "../lib/useTitle";

/* ---------------------------------------------------------------------- */
/* index: project picker + chat list                                      */
/* ---------------------------------------------------------------------- */

function ChatIndex() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const [projectId, setProjectId] = useState<string | null>(null);
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const chats = useQuery({
    queryKey: ["chats", projectId],
    queryFn: () => api.listChats(projectId!),
    enabled: !!projectId,
  });
  const create = useMutation({
    mutationFn: (pid: string) => api.createChat(pid),
    onSuccess: (c) => nav(`/chat/${c.id}`),
  });
  const del = useMutation({
    mutationFn: (id: string) => api.deleteChat(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["chats", projectId] }),
  });

  // auto-select the single / first project
  useEffect(() => {
    if (!projectId && projects.data?.length) setProjectId(projects.data[0].id);
  }, [projects.data, projectId]);

  return (
    <div className="mx-auto max-w-4xl p-6">
      <Crumbs
        items={[<span key="c" className="font-medium text-zinc-200">agent chat</span>]}
      />
      <div className="mb-4 mt-4 flex items-center justify-between">
        <div>
          <SectionTitle>Chats</SectionTitle>
          <div className="text-xs text-zinc-500">
            Talk to the agent — it can generate, edit, analyse and optimise parts.
          </div>
        </div>
        <Btn
          onClick={() => projectId && create.mutate(projectId)}
          disabled={!projectId}
          busy={create.isPending}
        >
          <MessageSquarePlus className="h-4 w-4" /> new chat
        </Btn>
      </div>
      <div className="mb-4 flex flex-wrap gap-2">
        {(projects.data ?? []).map((p) => (
          <button
            key={p.id}
            onClick={() => setProjectId(p.id)}
            className={`rounded-full border px-3 py-1 text-sm transition-colors ${
              p.id === projectId
                ? "border-sky-600 bg-sky-950 text-sky-200"
                : "border-zinc-800 text-zinc-400 hover:border-zinc-600"
            }`}
          >
            {p.name}
          </button>
        ))}
      </div>
      {projects.data?.length === 0 && (
        <EmptyState
          title="No projects yet"
          hint="Create a project first — chats live inside projects."
        />
      )}
      {projectId && (
        <div className="flex flex-col gap-2">
          {(chats.data ?? []).map((c) => (
            <Card
              key={c.id}
              className="flex items-center gap-3 !p-3 transition-colors hover:border-zinc-700"
            >
              <Link to={`/chat/${c.id}`} className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium text-zinc-200">
                  {c.title}
                </div>
                <div className="text-xs text-zinc-500">
                  {new Date(c.updated_at).toLocaleString()}
                  {c.turn_running && " · running…"}
                </div>
              </Link>
              <IconBtn title="delete chat" danger onClick={() => del.mutate(c.id)}>
                <Trash2 className="h-4 w-4" />
              </IconBtn>
            </Card>
          ))}
          {chats.data?.length === 0 && (
            <EmptyState title="No chats yet" hint="Start a new chat with the agent." />
          )}
        </div>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------------- */
/* conversation                                                           */
/* ---------------------------------------------------------------------- */

function ChatView({ chatId }: { chatId: string }) {
  const qc = useQueryClient();
  const [show3d, setShow3d] = useState(true);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const nav = useNavigate();
  const bottomRef = useRef<HTMLDivElement>(null);

  const state = useChatEvents(chatId, (type) => {
    if (type === "turn_end" || type === "tool_result" || type === "message") {
      qc.invalidateQueries({ queryKey: ["chat", chatId] });
      if (state.chat?.project_id)
        qc.invalidateQueries({ queryKey: ["assets", state.chat.project_id] });
    }
  });
  const { chat, messages, toolEvents, turnRunning, connected } = state;
  useTitle(chat?.title ?? "Chat");

  const assets = useQuery({
    queryKey: ["assets", chat?.project_id],
    queryFn: () => api.listAssets(chat!.project_id),
    enabled: !!chat?.project_id,
  });
  const assetsById = useMemo(
    () => new Map<string, Asset>((assets.data ?? []).map((a) => [a.id, a])),
    [assets.data],
  );

  // newest spawned version → right-side 3D panel
  const panelVid = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      for (const s of messages[i].meta?.spawned ?? []) {
        if (s.version_id) return s.version_id;
      }
    }
    return null;
  }, [messages]);
  const panelVersion = useQuery({
    queryKey: ["version", panelVid],
    queryFn: () => api.getVersion(panelVid!),
    enabled: !!panelVid && show3d,
    refetchInterval: (q) => {
      const v = q.state.data;
      return v && !v.artifacts.some((a) => a.kind === "glb") ? 4000 : false;
    },
  });
  const glb = panelVersion.data?.artifacts.find((a) => a.kind === "glb");

  const send = useMutation({
    mutationFn: (body: { content: string; asset_ids: string[] }) =>
      api.postChatMessage(chatId, body),
    onError: (e) => toastErr(e),
  });
  const [sendErr, setSendErr] = useState("");
  const toastErr = (e: unknown) => setSendErr(String(e));

  const cancel = useMutation({ mutationFn: () => api.cancelChat(chatId) });
  const del = useMutation({
    mutationFn: () => api.deleteChat(chatId),
    onSuccess: () => nav("/chat"),
  });

  // autoscroll on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, toolEvents]);

  const streamingId = messages.find((m) => m.meta?.streaming)?.id;

  return (
    <div className="flex h-screen min-h-0">
      {/* conversation column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-2.5">
          <Crumbs
            items={[
              <Link key="c" to="/chat" className="hover:text-zinc-300">
                agent chat
              </Link>,
              <span key="t" className="font-medium text-zinc-200">
                {chat?.title ?? "…"}
              </span>,
            ]}
          />
          <div className="ml-auto flex items-center gap-1.5">
            {!connected && (
              <span className="text-xs text-amber-500">reconnecting…</span>
            )}
            {turnRunning && (
              <Btn
                variant="ghost"
                className="!px-2 !py-1 text-xs"
                onClick={() => cancel.mutate()}
                busy={cancel.isPending}
              >
                <Square className="h-3 w-3" /> stop
              </Btn>
            )}
            <IconBtn
              title={show3d ? "hide model panel" : "show model panel"}
              onClick={() => setShow3d((v) => !v)}
            >
              <Boxes className="h-4 w-4" />
            </IconBtn>
            <IconBtn title="delete chat" danger onClick={() => setDeleteOpen(true)}>
              <Trash2 className="h-4 w-4" />
            </IconBtn>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          <div className="mx-auto flex max-w-3xl flex-col gap-3">
            {messages.length === 0 && !turnRunning && (
              <EmptyState
                title="Start the conversation"
                hint="Describe a part, attach reference images, or ask the agent to modify an existing version."
              />
            )}
            {messages.map((m) => (
              <MessageBubble
                key={m.id}
                msg={m}
                assetsById={assetsById}
                liveTools={m.id === streamingId ? toolEvents : undefined}
              />
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="border-t border-zinc-800 p-3">
          <div className="mx-auto max-w-3xl">
            {sendErr && (
              <div className="mb-2 text-xs text-red-400">{sendErr}</div>
            )}
            <Composer
              projectId={chat?.project_id ?? ""}
              disabled={!chat}
              onSend={(content, asset_ids) =>
                send.mutate({ content, asset_ids })
              }
            />
          </div>
        </div>
      </div>

      {/* right: live model panel */}
      {show3d && panelVid && (
        <div className="hidden w-[420px] shrink-0 flex-col border-l border-zinc-800 lg:flex">
          <div className="border-b border-zinc-800 px-3 py-2 text-xs font-medium text-zinc-400">
            latest result
          </div>
          <div className="relative min-h-0 flex-1">
            {glb ? (
              <Viewer3D url={glb.url} />
            ) : (
              <LivePreview versionId={panelVid} />
            )}
          </div>
        </div>
      )}

      <ConfirmModal
        open={deleteOpen}
        title="Delete chat?"
        body="The conversation history is removed; spawned versions and jobs stay."
        confirmLabel="delete"
        onConfirm={() => {
          setDeleteOpen(false);
          del.mutate();
        }}
        onClose={() => setDeleteOpen(false)}
      />
    </div>
  );
}

/* ---------------------------------------------------------------------- */

export default function ChatPage() {
  const { chatId } = useParams<{ chatId: string }>();
  if (!chatId) return <ChatIndex />;
  return <ChatView key={chatId} chatId={chatId} />;
}
