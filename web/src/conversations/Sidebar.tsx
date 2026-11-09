import { useEffect, useState } from "react";
import type { Api, Conversation, Hit } from "../api";
import { useSearch } from "./useSearch";

type Props = {
  api: Api;
  open: boolean;
  onToggle: () => void;
  conversations: Conversation[];
  current: string | null;
  onOpen: (id: string | null) => void;
  onChanged: () => void;
  onSkills: () => void;
};

export function Sidebar({ api, open, onToggle, conversations, current, onOpen, onChanged, onSkills }: Props) {
  const [q, setQ] = useState("");
  const hits = useSearch(api, q);
  const [editing, setEditing] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  useEffect(() => setConfirming(null), [conversations]);

  const items: Hit[] = hits ?? conversations.map((c) => ({ id: c.id, title: c.title, snippet: "" }));

  async function rename(id: string, title: string) {
    setEditing(null);
    if (title.trim()) await api.update(id, { title: title.trim() });
    onChanged();
  }

  async function remove(id: string) {
    if (confirming !== id) return setConfirming(id);
    await api.remove(id);
    if (id === current) onOpen(null);
    onChanged();
  }

  return (
    <>
      <button className="sidebar-toggle" aria-label={open ? "Fechar conversas" : "Abrir conversas"} aria-expanded={open} onClick={onToggle}>
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M4 6h16M4 12h10M4 18h16" />
        </svg>
      </button>
      <aside className="sidebar" aria-label="Conversas" inert={!open}>
        <button className="ghost sidebar-new" onClick={() => onOpen(null)}>
          Nova conversa
        </button>
        <input className="sidebar-search" type="search" placeholder="Buscar conversas" aria-label="Buscar conversas" value={q} onChange={(e) => setQ(e.target.value)} />
        <ul className="conv-list">
          {items.map((c) => (
            <li key={c.id} className={c.id === current ? "is-current" : undefined}>
              {editing === c.id ? (
                <form onSubmit={(e) => (e.preventDefault(), rename(c.id, new FormData(e.currentTarget).get("title") as string))}>
                  <input
                    name="title"
                    className="conv-rename"
                    aria-label="Novo título"
                    defaultValue={c.title}
                    autoFocus
                    onBlur={(e) => rename(c.id, e.currentTarget.value)}
                    onKeyDown={(e) => e.key === "Escape" && setEditing(null)}
                  />
                </form>
              ) : (
                <button className="conv" onClick={() => onOpen(c.id)} aria-current={c.id === current ? "page" : undefined}>
                  <span className="conv-title">{c.title || "Sem título"}</span>
                  {c.snippet && <span className="conv-snippet">{c.snippet}</span>}
                </button>
              )}
              <span className="conv-actions">
                <button aria-label={`Renomear ${c.title}`} onClick={() => setEditing(c.id)}>
                  <svg viewBox="0 0 24 24" aria-hidden>
                    <path d="M4 20h4L19 9l-4-4L4 16v4zM14 6l4 4" />
                  </svg>
                </button>
                <button
                  className={confirming === c.id ? "is-confirming" : undefined}
                  aria-label={confirming === c.id ? `Confirmar exclusão de ${c.title}` : `Excluir ${c.title}`}
                  onClick={() => remove(c.id)}
                >
                  {confirming === c.id ? (
                    "Excluir?"
                  ) : (
                    <svg viewBox="0 0 24 24" aria-hidden>
                      <path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12" />
                    </svg>
                  )}
                </button>
              </span>
            </li>
          ))}
        </ul>
        {items.length === 0 && <p className="sidebar-empty">{q ? `Nada encontrado para “${q}”.` : "Suas conversas aparecem aqui."}</p>}
        <footer className="sidebar-foot">
          <button className="ghost" onClick={onSkills}>
            Skills
          </button>
        </footer>
      </aside>
    </>
  );
}
