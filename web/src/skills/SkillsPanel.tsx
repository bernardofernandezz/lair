import { useCallback, useEffect, useRef, useState } from "react";
import type { Api, IndexEntry, Skill } from "../api";
import { SkillEditor } from "./SkillEditor";

const ORIGIN: Record<Skill["origin"], string> = {
  builtin: "Embutida",
  extra: "Pasta extra",
  created: "Criada aqui",
  installed: "Instalada",
};

type Props = { api: Api };

/** The skills surface, embeddable: the settings panel hosts it as its Skills tab. */
export function SkillsTab({ api }: Props) {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [editing, setEditing] = useState<Skill | "new" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.skills().then(setSkills, (e) => setError((e as Error).message));
  }, [api]);

  useEffect(refresh, [refresh]);

  async function act(f: () => Promise<unknown>) {
    setError(null);
    try {
      await f();
      refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (editing) {
    return (
      <SkillEditor
        api={api}
        skill={editing === "new" ? null : editing}
        onDone={() => {
          setEditing(null);
          refresh();
        }}
      />
    );
  }

  return (
    <>
      <div className="explore">
        <UploadZip api={api} onInstalled={refresh} />
      </div>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
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
      {skills.length === 0 && <p className="hint">Nenhuma skill ainda. Importe um .zip, instale de um endereço ou crie aqui.</p>}
      <button className="solid" onClick={() => setEditing("new")}>
        Nova skill
      </button>
      <IndexSkills api={api} onInstalled={refresh} />
    </>
  );
}

/** Installs a skill straight from a `.zip` on the user's disk. */
function UploadZip({ api, onInstalled }: { api: Api; onInstalled: () => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  async function upload(f: File | null) {
    if (input.current) input.current.value = "";
    if (!f) return;
    setBusy(true);
    setMessage(null);
    try {
      const s = await api.uploadSkill(f);
      onInstalled();
      setMessage({ ok: true, text: `“${s.name}” importada do zip.` });
    } catch (e) {
      setMessage({ ok: false, text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="install-url">
      <label htmlFor="skill-zip">Importar um .zip do computador</label>
      <div className="field">
        <input ref={input} id="skill-zip" type="file" accept=".zip,application/zip" hidden onChange={(e) => upload(e.target.files?.item(0) ?? null)} />
        <input
          type="text"
          inputMode="none"
          readOnly
          value={busy ? "Enviando…" : "Escolher arquivo…"}
          aria-hidden
          tabIndex={-1}
          onClick={() => input.current?.click()}
          onFocus={(e) => e.target.blur()}
        />
        <button type="button" className="solid" disabled={busy} onClick={() => input.current?.click()}>
          {busy ? "Enviando…" : "Importar"}
        </button>
      </div>
      <p className="hint">Um arquivo .zip contendo skill.json e prompt.md.</p>
      {message && (
        <p className={message.ok ? "notice-ok" : "error"} role="status">
          {message.text}
        </p>
      )}
    </div>
  );
}

/** Skills listed in the configured index, installable; explains itself when no index is set. */
function IndexSkills({ api, onInstalled }: { api: Api; onInstalled: () => void }) {
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
          Para navegar por uma lista de skills, configure o índice aqui acima (ou inicie o lair com <code>--skills-index</code>).
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
