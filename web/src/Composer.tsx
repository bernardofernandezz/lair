import { useRef, useState, type ReactNode } from "react";
import type { Api, Conversation, Ref } from "./api";
import { mentionAt, useMentionOptions } from "./conversations/mention";

type Props = {
  api: Api;
  conversations: Conversation[];
  /** Conversation being written in, which cannot cite itself. */
  current: string | null;
  busy: boolean;
  hello: string | false;
  /** Model picker, reasoning control: shown in the tools row. */
  tools: ReactNode;
  onSend: (text: string, refs: Ref[]) => void;
  onStop: () => void;
};

export function Composer({ api, conversations, current, busy, hello, tools, onSend, onStop }: Props) {
  const [draft, setDraft] = useState("");
  const [refs, setRefs] = useState<Ref[]>([]);
  const [mention, setMention] = useState<{ start: number; query: string } | null>(null);
  const [hi, setHi] = useState(0);
  const input = useRef<HTMLTextAreaElement>(null);
  const exclude = [...refs.map((r) => r.id), ...(current ? [current] : [])];
  const options = useMentionOptions(api, conversations, mention?.query ?? null, exclude);

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

  function submit() {
    const text = draft.trim();
    if (!text || busy) return;
    onSend(text, refs);
    setDraft("");
    setRefs([]);
    setMention(null);
  }

  const picking = mention !== null && options.length > 0;

  return (
    <form
      className="field composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      {hello && <p className="hello">{hello}</p>}
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
      {refs.length > 0 && (
        <span className="ref-chips">
          {refs.map((r) => (
            <span key={r.id} className="ref-chip">
              <span>{r.title}</span>
              <button type="button" aria-label={`Remover ${r.title}`} onClick={() => setRefs(refs.filter((x) => x.id !== r.id))}>
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
        {busy ? (
          <button type="button" className="send" onClick={onStop} aria-label="Parar resposta">
            <svg viewBox="0 0 24 24" aria-hidden>
              <rect x="7" y="7" width="10" height="10" rx="2" />
            </svg>
          </button>
        ) : (
          <button className="send" disabled={!draft.trim()} aria-label="Enviar">
            <svg viewBox="0 0 24 24" aria-hidden>
              <path d="M12 19V5M6 11l6-6 6 6" />
            </svg>
          </button>
        )}
      </div>
    </form>
  );
}
