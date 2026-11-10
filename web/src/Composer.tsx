import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Api, Conversation, FileTypes, Ref } from "./api";
import { mentionAt, useMentionOptions } from "./conversations/mention";

const MAX_FILES = 8;

/** A file picked for the next message; `image` ones go to the model as pixels. */
export type Pending = { file: File; image: boolean };

const extOf = (name: string) => name.slice(name.lastIndexOf(".") + 1).toLowerCase();

type Props = {
  api: Api;
  conversations: Conversation[];
  /** Conversation being written in, which cannot cite itself. */
  current: string | null;
  busy: boolean;
  hello: string | false;
  /** Model picker, reasoning control: shown in the tools row. */
  tools: ReactNode;
  /** The chosen model sees images; otherwise they are refused here with the reason. */
  vision: boolean;
  fileTypes: FileTypes;
  onSend: (text: string, refs: Ref[], files: Pending[]) => void;
  onStop: () => void;
};

export function Composer({ api, conversations, current, busy, hello, tools, vision, fileTypes: types, onSend, onStop }: Props) {
  const [draft, setDraft] = useState("");
  const [refs, setRefs] = useState<Ref[]>([]);
  const [files, setFiles] = useState<Pending[]>([]);
  const [warn, setWarn] = useState<string | null>(null);
  const [mention, setMention] = useState<{ start: number; query: string } | null>(null);
  const [hi, setHi] = useState(0);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const exclude = [...refs.map((r) => r.id), ...(current ? [current] : [])];
  const options = useMentionOptions(api, conversations, mention?.query ?? null, exclude);

  const previews = useMemo(() => new Map(files.filter((f) => f.image).map((f) => [f, URL.createObjectURL(f.file)])), [files]);
  useEffect(() => () => previews.forEach((u) => URL.revokeObjectURL(u)), [previews]);
  useEffect(() => {
    if (vision || !files.some((f) => f.image)) return;
    setFiles((fs) => fs.filter((f) => !f.image));
    setWarn("Esse modelo não enxerga imagens; elas foram tiradas da mensagem.");
  }, [vision, files]);

  function track(text: string, caret: number) {
    setDraft(text);
    const m = mentionAt(text, caret);
    if (m?.query !== mention?.query) setHi(0);
    setMention(m);
  }

  function pick(r: Ref) {
    if (!mention) return;
    const end = mention.start + 1 + mention.query.length;
    setDraft(draft.slice(0, mention.start) + draft.slice(end).replace(/^ /, ""));
    setRefs([...refs, r]);
    setMention(null);
    input.current?.focus();
  }

  /** From the picker, a paste or a drop. The caller copies FileLists first: they die when the input resets. */
  function addFiles(picked: File[]) {
    if (!picked.length) return;
    setWarn(null);
    const image = (f: File) => types.images.includes(extOf(f.name));
    const accepted = (f: File) => (image(f) ? vision : types.documents.includes(extOf(f.name)));
    const good = picked.filter(accepted);
    const bad = picked.filter((f) => !good.includes(f));
    const take = good.slice(0, Math.max(0, MAX_FILES - files.length));
    if (take.length) setFiles([...files, ...take.map((file) => ({ file, image: image(file) }))]);
    if (bad.some(image)) setWarn("Esse modelo não enxerga imagens. Escolha um modelo de visão (com o mmproj ao lado do .gguf).");
    else if (bad.length) setWarn("Aceito documentos de texto, código, .pdf e imagens.");
    else if (good.length > take.length) setWarn(`Até ${MAX_FILES} anexos por mensagem.`);
  }

  const remove = (f: Pending) => setFiles(files.filter((x) => x !== f));
  const kB = (n: number) => (n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${Math.round(n / 1024)} kB` : `${(n / (1024 * 1024)).toFixed(1)} MB`);

  function submit() {
    const text = draft.trim();
    if ((!text && files.length === 0) || busy) return;
    onSend(text, refs, files);
    setDraft("");
    setRefs([]);
    setFiles([]);
    setMention(null);
  }

  const picking = mention !== null && options.length > 0;
  const canSend = !busy && (draft.trim() !== "" || files.length > 0);

  return (
    <form
      className={`field composer${dragging ? " is-dropping" : ""}`}
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      onDragOver={(e) => {
        if (busy || !e.dataTransfer.types.includes("Files")) return;
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false);
      }}
      onDrop={(e) => {
        if (!e.dataTransfer.files.length) return;
        e.preventDefault();
        setDragging(false);
        addFiles(Array.from(e.dataTransfer.files));
      }}
      onPaste={(e) => {
        const pasted = Array.from(e.clipboardData.files);
        if (!pasted.length) return;
        e.preventDefault();
        addFiles(pasted);
      }}
    >
      {dragging && (
        <div className="drop-hint" aria-hidden>
          Solte para anexar
        </div>
      )}
      {hello && <p className="hello">{hello}</p>}
      {warn && (
        <p className="hint" role="status">
          {warn}
        </p>
      )}
      {picking && (
        <ul className="mention" role="listbox" aria-label="Citar conversa">
          {options.map((o, i) => (
            <li
              key={o.id}
              role="option"
              aria-selected={i === hi}
              className={i === hi ? "is-hi" : undefined}
              onPointerEnter={() => setHi(i)}
              onPointerDown={(e) => (e.preventDefault(), pick(o))}
            >
              {o.title}
            </li>
          ))}
        </ul>
      )}
      {(refs.length > 0 || files.length > 0) && (
        <span className="ref-chips">
          {refs.map((r) => (
            <span key={r.id} className="ref-chip">
              <span>{r.title}</span>
              <button type="button" aria-label={`Remover ${r.title}`} onClick={() => setRefs(refs.filter((x) => x.id !== r.id))}>
                ×
              </button>
            </span>
          ))}
          {files.map((f, i) => (
            <span key={`${i}-${f.file.name}`} className={`ref-chip is-file${previews.has(f) ? " is-image" : ""}`}>
              {previews.has(f) ? (
                <img src={previews.get(f)} alt="" />
              ) : (
                <svg viewBox="0 0 24 24" aria-hidden>
                  <path d="M7 5h7l5 5v9a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zM13 5v6h6" />
                </svg>
              )}
              <span>
                {f.file.name} <em>{kB(f.file.size)}</em>
              </span>
              <button type="button" aria-label={`Remover ${f.file.name}`} onClick={() => remove(f)}>
                ×
              </button>
            </span>
          ))}
        </span>
      )}
      <textarea
        ref={input}
        rows={1}
        autoFocus
        value={draft}
        placeholder="Escreva uma mensagem. Use @ para citar outra conversa."
        aria-label="Mensagem"
        aria-autocomplete="list"
        aria-expanded={picking}
        onChange={(e) => track(e.target.value, e.target.selectionStart)}
        onSelect={(e) => mention && track(e.currentTarget.value, e.currentTarget.selectionStart)}
        onKeyDown={(e) => {
          if (picking) {
            const moves: Record<string, () => void> = {
              ArrowDown: () => setHi((h) => (h + 1) % options.length),
              ArrowUp: () => setHi((h) => (h - 1 + options.length) % options.length),
              Enter: () => pick(options[hi]),
              Tab: () => pick(options[hi]),
              Escape: () => setMention(null),
            };
            if (moves[e.key]) {
              e.preventDefault();
              return moves[e.key]();
            }
          }
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <div className="tools">
        {tools}
        <input
          ref={fileInput}
          type="file"
          hidden
          multiple
          accept={[...types.documents, ...(vision ? types.images : [])].map((e) => `.${e}`).join(",")}
          aria-hidden
          tabIndex={-1}
          onChange={(e) => {
            const picked = Array.from(e.target.files ?? []);
            e.target.value = "";
            addFiles(picked);
          }}
        />
        <button
          type="button"
          className="attach"
          aria-label="Anexar arquivo"
          title={vision ? "Anexar documento ou imagem (ou cole / arraste aqui)" : "Anexar documento: texto, código, .pdf (ou arraste aqui)"}
          onClick={() => fileInput.current?.click()}
        >
          <svg viewBox="0 0 24 24" aria-hidden>
            <path d="M8 12.5V8a4 4 0 0 1 8 0v8a2.5 2.5 0 0 1-5 0V9.5a1 1 0 0 1 2 0V16" />
          </svg>
        </button>
        {busy ? (
          <button type="button" className="send" onClick={onStop} aria-label="Parar resposta">
            <svg viewBox="0 0 24 24" aria-hidden>
              <rect x="7" y="7" width="10" height="10" rx="2" />
            </svg>
          </button>
        ) : (
          <button className="send" disabled={!canSend} aria-label="Enviar">
            <svg viewBox="0 0 24 24" aria-hidden>
              <path d="M12 19V5M6 11l6-6 6 6" />
            </svg>
          </button>
        )}
      </div>
    </form>
  );
}
