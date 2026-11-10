import { useCallback, useEffect, useRef, useState } from "react";
import type { Api, IndexEntry, Skill } from "../api";
import { SkillEditor } from "./SkillEditor";
import { pick } from "../i18n";

const T = pick({
  en: {
    origin: { builtin: "Built in", extra: "Extra folder", created: "Made here", installed: "Installed" },
    turnOff: "Turn off", turnOn: "Turn on", edit: "Edit", delete: "Delete",
    none: "No skills yet. Import a .zip, install one from a link or make one here.",
    newSkill: "New skill",
    zip: "Import a .zip from this computer", sending: "Uploading…", choose: "Choose file…", import: "Import",
    zipHint: "A .zip file with skill.json and prompt.md.",
    imported: (n: string) => `“${n}” imported from the zip.`,
    indexFailed: "Couldn't read the index: ",
    installed: (n: string) => `“${n}” installed.`,
    fromUrl: "Install from a link", urlPlaceholder: "https://github.com/you/skills/tree/main/my-skill",
    installing: "Installing…", install: "Install", update: "Update",
    urlHint: "Takes a .zip, a GitHub folder or the link to a folder with skill.json and prompt.md.",
    noIndex1: "To browse a list of skills, set the index above (or start lair with ", noIndex2: ").",
    emptyIndex: "The index lists no skills.",
  },
  pt: {
    origin: { builtin: "Embutida", extra: "Pasta extra", created: "Criada aqui", installed: "Instalada" },
    turnOff: "Desligar", turnOn: "Ligar", edit: "Editar", delete: "Excluir",
    none: "Nenhuma skill ainda. Importe um .zip, instale de um endereço ou crie aqui.",
    newSkill: "Nova skill",
    zip: "Importar um .zip do computador", sending: "Enviando…", choose: "Escolher arquivo…", import: "Importar",
    zipHint: "Um arquivo .zip contendo skill.json e prompt.md.",
    imported: (n: string) => `“${n}” importada do zip.`,
    indexFailed: "Não foi possível ler o índice: ",
    installed: (n: string) => `“${n}” instalada.`,
    fromUrl: "Instalar de um endereço", urlPlaceholder: "https://github.com/você/skills/tree/main/minha-skill",
    installing: "Instalando…", install: "Instalar", update: "Atualizar",
    urlHint: "Aceita um .zip, uma pasta do GitHub ou o endereço de uma pasta com skill.json e prompt.md.",
    noIndex1: "Para navegar por uma lista de skills, configure o índice aqui acima (ou inicie o lair com ", noIndex2: ").",
    emptyIndex: "O índice não lista nenhuma skill.",
  },
});

const ORIGIN: Record<Skill["origin"], string> = T.origin;

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
                aria-label={`${s.enabled ? T.turnOff : T.turnOn} ${s.name}`}
                onClick={() => act(() => api.toggleSkill(s.id, !s.enabled))}
              />
              <button className="ghost" onClick={() => setEditing(s)}>
                {T.edit}
              </button>
              {(s.origin === "created" || s.origin === "installed") && (
                <button className="ghost" onClick={() => act(() => api.removeSkill(s.id))}>
                  {T.delete}
                </button>
              )}
            </div>
          </li>
        ))}
      </ul>
      {skills.length === 0 && <p className="hint">{T.none}</p>}
      <button className="solid" onClick={() => setEditing("new")}>
        {T.newSkill}
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
      setMessage({ ok: true, text: T.imported(s.name) });
    } catch (e) {
      setMessage({ ok: false, text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="install-url">
      <label htmlFor="skill-zip">{T.zip}</label>
      <div className="field">
        <input ref={input} id="skill-zip" type="file" accept=".zip,application/zip" hidden onChange={(e) => upload(e.target.files?.item(0) ?? null)} />
        <input
          type="text"
          inputMode="none"
          readOnly
          value={busy ? T.sending : T.choose}
          aria-hidden
          tabIndex={-1}
          onClick={() => input.current?.click()}
          onFocus={(e) => e.target.blur()}
        />
        <button type="button" className="solid" disabled={busy} onClick={() => input.current?.click()}>
          {busy ? T.sending : T.import}
        </button>
      </div>
      <p className="hint">{T.zipHint}</p>
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
    api.skillsIndex().then(setIndex, (e) => setMessage({ ok: false, text: T.indexFailed + (e as Error).message }));
  }, [api]);
  useEffect(load, [load]);

  async function install(from: string) {
    setBusy(from);
    setMessage(null);
    try {
      const s = await api.installSkill(from);
      setMessage({ ok: true, text: T.installed(s.name) });
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
        <label htmlFor="skill-url">{T.fromUrl}</label>
        <div className="field">
          <input
            id="skill-url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder={T.urlPlaceholder}
          />
          <button className="solid" disabled={!url.trim() || busy !== null}>
            {busy === url.trim() ? T.installing : T.install}
          </button>
        </div>
        <p className="hint">{T.urlHint}</p>
      </form>
      {message && (
        <p className={message.ok ? "notice-ok" : "error"} role="status">
          {message.text}
        </p>
      )}
      {index && !index.configured && (
        <p className="hint">
          {T.noIndex1}
          <code>--skills-index</code>
          {T.noIndex2}
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
                  {busy === entry.url ? T.installing : installed ? T.update : T.install}
                </button>
              </div>
            </li>
          ))}
          {index.skills.length === 0 && <p className="hint">{T.emptyIndex}</p>}
        </ul>
      )}
    </div>
  );
}
