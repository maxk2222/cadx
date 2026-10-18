import { ImagePlus, SendHorizonal, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, errMsg } from "../../lib/api";
import type { Asset } from "../../lib/types";
import { Spinner } from "../ui";

/** Chat composer — text + image attachments via picker, paste or drag&drop.
 *  Files upload immediately (project asset endpoint); the staged assets are
 *  attached to the message on send. */
export default function Composer({
  projectId,
  ensureProject,
  disabled,
  onSend,
}: {
  projectId: string | null;
  /** called on first upload when the chat has no project yet — lazy-creates
   *  one and binds the chat to it (like the New-part wizard does) */
  ensureProject?: () => Promise<string | null>;
  disabled?: boolean;
  /** resolves on success — the composer only clears the draft then, so a
   *  failed send never loses the user's text/attachments */
  onSend: (content: string, assetIds: string[]) => Promise<unknown> | unknown;
}) {
  const [text, setText] = useState("");
  const [staged, setStaged] = useState<Asset[]>([]);
  const [uploading, setUploading] = useState(0);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  // autosize
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "0px";
    ta.style.height = Math.min(200, ta.scrollHeight) + "px";
  }, [text]);

  const upload = async (files: File[]) => {
    const imgs = files.filter((f) => f.type.startsWith("image/"));
    if (!imgs.length) {
      setError("only image files are supported in chat");
      return;
    }
    setError("");
    setUploading((n) => n + imgs.length);
    try {
      const pid = projectId ?? (await ensureProject?.()) ?? null;
      if (!pid) {
        setError("couldn't create a project for the upload");
        return;
      }
      const assets = await api.uploadAssets(pid, imgs);
      setStaged((s) => [...s, ...assets]);
    } catch (e) {
      setError(String(e));
    } finally {
      setUploading((n) => n - imgs.length);
    }
  };

  const send = async () => {
    const content = text.trim();
    if ((!content && !staged.length) || disabled || uploading) return;
    try {
      await onSend(
        content,
        staged.map((a) => a.id),
      );
      setText("");
      setStaged([]);
      setError("");
    } catch (e) {
      setError(errMsg(e));
    }
  };

  return (
    <div
      className={`rounded-lg border bg-zinc-900 transition-colors ${
        drag ? "border-brand-500" : "border-zinc-800"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        void upload([...e.dataTransfer.files]);
      }}
    >
      {staged.length > 0 && (
        <div className="flex flex-wrap gap-2 px-3 pt-3">
          {staged.map((a) => (
            <div key={a.id} className="group relative">
              <img
                src={a.url}
                alt={a.filename}
                className="h-14 w-14 rounded border border-zinc-700 object-cover"
              />
              <button
                className="absolute -right-1.5 -top-1.5 hidden h-4 w-4 items-center justify-center rounded-full bg-zinc-700 text-zinc-200 group-hover:flex"
                onClick={() => setStaged((s) => s.filter((x) => x.id !== a.id))}
                title="remove attachment"
                aria-label="remove attachment"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="flex items-end gap-2 p-2">
        <button
          className="flex h-8 w-8 items-center justify-center rounded-md text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
          title="attach images"
          aria-label="attach images"
          onClick={() => fileRef.current?.click()}
          disabled={disabled}
        >
          <ImagePlus className="h-4 w-4" />
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => {
            void upload([...(e.target.files ?? [])]);
            e.target.value = "";
          }}
        />
        <textarea
          ref={taRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onPaste={(e) => {
            const files = [...e.clipboardData.files];
            if (files.length) {
              e.preventDefault();
              void upload(files);
            }
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          placeholder={
            drag ? "drop images here" : "Describe a part, ask for a change… (Enter to send)"
          }
          rows={1}
          className="max-h-48 flex-1 resize-none bg-transparent px-1 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none"
        />
        {uploading > 0 && <Spinner className="h-4 w-4 self-center" />}
        <button
          className="flex h-8 w-8 items-center justify-center rounded-md bg-brand-600 text-white transition-colors hover:bg-brand-500 disabled:opacity-40"
          title="send"
          aria-label="send message"
          onClick={() => void send()}
          disabled={disabled || uploading > 0 || (!text.trim() && !staged.length)}
        >
          <SendHorizonal className="h-4 w-4" />
        </button>
      </div>
      {error && <div className="px-3 pb-2 text-xs text-red-400">{error}</div>}
    </div>
  );
}
