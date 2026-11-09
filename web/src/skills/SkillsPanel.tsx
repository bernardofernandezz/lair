import { useCallback, useEffect, useRef, useState } from "react";
import type { Api, IndexEntry, Skill } from "../api";
import { SkillEditor } from "./SkillEditor";

const ORIGIN: Record<Skill["origin"], string> = {
  builtin: "Embutida",
  extra: "Pasta extra",
  created: "Criada aqui",
  installed: "Instalada",
};

type Props = { api: Api; open: boolean; onClose: () => void };

export function SkillsPanel({ api, open, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [tab, setTab] = useState<"mine" | "explore">("mine");
  const [skills, setSkills] = useState<Skill[]>([]);
  const [editing, setEditing] = useState<Skill | "new" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.skills().then(setSkills, (e) => setError((e as Error).message));
  }, [api]);

  useEffect(() => {
    const d = dialog.current;
    if (!d) return;
    if (open && !d.open) {
      d.showModal();
      refresh();
    } else if (!open && d.open) d.close();
  }, [open, refresh]);

  async function act(f: () => Promise<unknown>) {
    setError(null);
    try {
      await f();
      refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <dialog ref={dialog} className="panel" onClose={onClose} aria-labelledby="skills-title">
      <header className="panel-head">
        <h2 id="skills-title">Skills</h2>
        <button className="ghost" onClick={onClose}>
          Fechar
        </button>
      </header>
      <p className="panel-intro">
        Skills são instruções extras que entram na conversa quando sua mensagem tem uma das palavras-gatilho.
      </p>

      {editing ? (
        <SkillEditor
          api={api}
          skill={editing === "new" ? null : editing}
          onDone={() => {
            setEditing(null);
            refresh();
          }}
        />
      ) : (
        <>
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === "mine"} onClick={() => setTab("mine")}>
              Minhas skills
            </button>
            <button role="tab" aria-selected={tab === "explore"} onClick={() => setTab("explore")}>
              Explorar e instalar
            </button>
          </div>
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          {tab === "mine" ? (
            <>
              <ul className="skill-list">
                {skills.map((s) => (
                  <li key={s.id} className={s.enabled ? undefined : "is-off"}>
                    <div className="skill-main">
                      <span className="skill-name">{s.name}</span>
                      <span className="badge">{ORIGIN[s.origin]}</span>
                      {s.description && <span className="skill-desc">{s.description}</span>}
                      <span className="skill-when">Gatilhos: {s.when.join(", ")}</span>
                    </div>
                    <div className="skill-actions">
                      <button
                        className="switch"
                        role="switch"
                        aria-checked={s.enabled}
                        aria-label={`${s.enabled ? "Desligar" : "Ligar"} ${s.name}`}
                        onClick={() => act(() => api.toggleSkill(s.id, !s.enabled))}
                      />
                      <button className="ghost" onClick={() => setEditing(s)}>
                        Editar
                      </button>
                      {(s.origin === "created" || s.origin === "installed") && (
                        <button className="ghost" onClick={() => act(() => api.removeSkill(s.id))}>
                          Excluir
                        </button>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
              <button className="solid" onClick={() => setEditing("new")}>
                Nova skill
              </button>
            </>
          ) : (
            <Explore api={api} onInstalled={refresh} />
          )}
        </>
      )}
    </dialog>
  );
}

function Explore({ api, onInstalled }: { api: Api; onInstalled: () => void }) {
  const [index, setIndex] = useState<{ configured: boolean; skills: { entry: IndexEntry; installed: boolean }[] } | null>(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  const load = useCallback(() => {
    api.skillsIndex().then(setIndex, (e) => setMessage({ ok: false, text: `Não foi possível ler o índice: ${(e as Error).message}` }));
  }, [api]);
  useEffect(load, [load]);

  async function install(from: string) {
    setBusy(from);
    setMessage(null);
    try {
      const s = await api.installSkill(from);
      setMessage({ ok: true, text: `“${s.name}” instalada.` });
      setUrl("");
      onInstalled();
      load();
    } catch (e) {
      setMessage({ ok: false, text: (e as Error).message });
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="explore">
      <form
        className="install-url"
        onSubmit={(e) => {
          e.preventDefault();
          if (url.trim()) install(url.trim());
        }}
      >
        <label htmlFor="skill-url">Instalar de um endereço</label>
        <div className="field">
          <input
            id="skill-url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://github.com/você/skills/tree/main/minha-skill"
          />
          <button className="solid" disabled={!url.trim() || busy !== null}>
            {busy === url.trim() ? "Instalando…" : "Instalar"}
          </button>
        </div>
        <p className="hint">Aceita um .zip, uma pasta do GitHub ou o endereço de uma pasta com skill.json e prompt.md.</p>
      </form>
      {message && (
        <p className={message.ok ? "notice-ok" : "error"} role="status">
          {message.text}
        </p>
      )}
      {index && !index.configured && (
        <p className="hint">
          Para navegar por uma lista de skills, inicie o lair com <code>--skills-index</code> (ou <code>LAIR_SKILLS_INDEX</code>)
          apontando para um índice JSON.
        </p>
      )}
      {index?.configured && (
        <ul className="skill-list">
          {index.skills.map(({ entry, installed }) => (
            <li key={entry.url}>
              <div className="skill-main">
                <span className="skill-name">{entry.name}</span>
                {entry.description && <span className="skill-desc">{entry.description}</span>}
              </div>
              <div className="skill-actions">
                <button className="ghost" disabled={busy !== null} onClick={() => install(entry.url)}>
                  {busy === entry.url ? "Instalando…" : installed ? "Atualizar" : "Instalar"}
                </button>
              </div>
            </li>
          ))}
          {index.skills.length === 0 && <p className="hint">O índice não lista nenhuma skill.</p>}
        </ul>
      )}
    </div>
  );
}
