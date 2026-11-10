import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import type { Api, Config, Settings } from "../api";
import { modelName } from "../api";
import { isEffort, MODES } from "../Controls";
import { SkillsTab } from "../skills/SkillsPanel";

type Props = { api: Api; open: boolean; onClose: () => void };
type Tab = "general" | "tailnet" | "skills" | "system";

const TABS: { id: Tab; label: string }[] = [
  { id: "general", label: "Geral" },
  { id: "tailnet", label: "Tailnet" },
  { id: "skills", label: "Skills" },
  { id: "system", label: "Sistema" },
];

/** Saves `change` on top of everything else the UI saved, then says `what` was saved. */
type Save = (change: Config, what: string) => Promise<void>;

export function SettingsPanel({ api, open, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [tab, setTab] = useState<Tab>("general");
  const [st, setSt] = useState<Settings | null>(null);
  const [notice, setNotice] = useState<{ ok: boolean; text: string } | null>(null);

  const refresh = useCallback(() => {
    api.settings().then(setSt, (e) => setNotice({ ok: false, text: (e as Error).message }));
  }, [api]);

  useEffect(() => {
    const d = dialog.current;
    if (!d) return;
    if (open && !d.open) {
      d.showModal();
      setNotice(null);
      refresh();
    } else if (!open && d.open) d.close();
  }, [open, refresh]);

  const save: Save = async (change, what) => {
    try {
      await api.saveSettings(change);
      setNotice({ ok: true, text: what });
      refresh();
    } catch (e) {
      setNotice({ ok: false, text: (e as Error).message });
    }
  };

  return (
    <dialog ref={dialog} className="panel" onClose={onClose} aria-labelledby="settings-title">
      <header className="panel-head">
        <h2 id="settings-title">
          <span className="pearl" aria-hidden />
          Configurações
        </h2>
        <button className="ghost" onClick={onClose}>
          Fechar
        </button>
      </header>
      <p className="panel-intro">
        Mudanças valem na hora e ficam salvas para os próximos inícios. O que veio da linha de comando só muda reiniciando o
        servidor sem a flag.
      </p>

      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {notice && (
        <p className={notice.ok ? "notice-ok" : "error"} role="status">
          {notice.text}
        </p>
      )}

      {st && tab === "general" && <General st={st} save={save} />}
      {st && tab === "tailnet" && <Tailnet st={st} save={save} />}
      {tab === "skills" && (
        <>
          {st && <SkillsIndex st={st} save={save} />}
          <SkillsTab api={api} />
        </>
      )}
      {st && tab === "system" && <System st={st} />}
    </dialog>
  );
}

/** A setting given on the command line: shown, but not editable here. */
const locked = (st: Settings, key: keyof Config) => st.flags[key] !== undefined && st.flags[key] !== null;

/** One labelled setting with its hint, and the reason when the command line holds it. */
function Field({ id, label, hint, flag, children }: { id: string; label: string; hint: string; flag: boolean; children: ReactNode }) {
  return (
    <>
      <label htmlFor={id}>{label}</label>
      {children}
      {flag && <p className="flag-note">Veio da linha de comando e vale até o servidor reiniciar sem a flag.</p>}
      <p className="hint">{hint}</p>
    </>
  );
}

/** The value a text setting shows: the flag's when there is one, else what was saved. */
const shown = (st: Settings, key: "searxng" | "skillsIndex" | "name") => st.flags[key] ?? st.saved[key] ?? "";
/** A field's text; `undefined` for a disabled field, which the form leaves out, so a flag's setting is never cleared. */
const text = (form: HTMLFormElement, name: string) => {
  const v = new FormData(form).get(name);
  return v === null ? undefined : String(v);
};

function SkillsIndex({ st, save }: { st: Settings; save: Save }) {
  return (
    <form
      className="set-grid"
      onSubmit={(e) => {
        e.preventDefault();
        save({ skillsIndex: text(e.currentTarget, "skillsIndex") }, "Índice de skills salvo.");
      }}
    >
      <Field id="set-skills-index" label="Índice de skills" hint="Endereço de um JSON que lista skills para navegar e instalar; vazio deixa sem índice." flag={locked(st, "skillsIndex")}>
        <input id="set-skills-index" name="skillsIndex" type="url" defaultValue={shown(st, "skillsIndex")} disabled={locked(st, "skillsIndex")} placeholder="https://…/skills.json" />
      </Field>
      <div className="editor-actions">
        <button className="solid">Salvar</button>
      </div>
    </form>
  );
}

function General({ st, save }: { st: Settings; save: Save }) {
  return (
    <form
      className="set-grid"
      onSubmit={(e) => {
        e.preventDefault();
        const reasoning = text(e.currentTarget, "reasoning");
        save({ searxng: text(e.currentTarget, "searxng"), reasoning: reasoning && isEffort(reasoning) ? reasoning : undefined }, "Configurações gerais salvas.");
      }}
    >
      <Field id="set-searxng" label="Busca na web (SearXNG)" hint="Endereço de uma instância SearXNG configurada para responder em JSON. Sem ela, o agente não busca na web." flag={locked(st, "searxng")}>
        <input id="set-searxng" name="searxng" type="url" defaultValue={shown(st, "searxng")} disabled={locked(st, "searxng")} placeholder="http://localhost:8888" />
      </Field>
      <Field id="set-reasoning" label="Raciocínio padrão" hint="Usado quando a conversa não escolhe um nível de raciocínio." flag={locked(st, "reasoning")}>
        <select id="set-reasoning" name="reasoning" defaultValue={st.system.reasoning} disabled={locked(st, "reasoning")}>
          {MODES.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}: {m.hint.toLowerCase()}
            </option>
          ))}
        </select>
      </Field>
      <div className="editor-actions">
        <button className="solid">Salvar</button>
      </div>
    </form>
  );
}

/** Is tailscaled here, is it up, and how other machines reach this lair. */
function TailStatusCard({ st }: { st: Settings }) {
  const [copied, setCopied] = useState(false);
  const s = st.tailscale.status;
  const localOnly = /^(127\.|localhost|\[::1\])/.test(st.system.addr);
  const tone = s?.running ? "is-ok" : s ? "is-warn" : "is-off";
  const title = s?.running ? "Conectado à tailnet" : s ? `Tailscale ${s.state === "NeedsLogin" ? "precisa de login" : "parado"}` : "Tailscale não encontrado";
  return (
    <section className={`tail-card ${tone}`} aria-live="polite">
      <header>
        <span className="tail-dot" aria-hidden />
        <strong>{title}</strong>
        {s?.host && <span className="badge">{s.host}</span>}
      </header>
      {s?.running && (
        <>
          {s.ips.length > 0 && <p className="mono tail-ips">{s.ips.join(" · ")}</p>}
          {s.url && (
            <div className="tail-url">
              <code>{s.url}</code>
              <button
                type="button"
                className="ghost"
                onClick={() => navigator.clipboard?.writeText(s.url!).then(() => (setCopied(true), setTimeout(() => setCopied(false), 1500)))}
              >
                {copied ? "Copiado" : "Copiar"}
              </button>
            </div>
          )}
          {localOnly ? (
            <p className="hint tail-warn">
              Este lair só escuta em <code>{st.system.addr}</code>, então outros aparelhos ainda não alcançam esse endereço. Reinicie com{" "}
              <code>lair serve --host 0.0.0.0</code> (ou com o IP da tailnet) para abrir.
            </p>
          ) : (
            <p className="hint">Abra esse endereço em qualquer aparelho da sua tailnet para usar este lair.</p>
          )}
        </>
      )}
      {s && !s.running && (
        <p className="hint">
          Rode <code>sudo tailscale up</code> nesta máquina e reabra esta aba.
        </p>
      )}
      {!s && (
        <p className="hint">
          {st.tailscale.socket ? "O socket do tailscaled existe, mas não respondeu. Ele está rodando?" : "Instale o Tailscale (tailscale.com/download) e rode "}
          {!st.tailscale.socket && <code>sudo tailscale up</code>}
          {!st.tailscale.socket && " para juntar suas máquinas e compartilhar modelos entre elas."}
        </p>
      )}
    </section>
  );
}

function Tailnet({ st, save }: { st: Settings; save: Save }) {
  const flagPeers = st.flags.peers ?? [];
  return (
    <>
      <TailStatusCard st={st} />
      <label className="set-toggle">
        <input
          type="checkbox"
          checked={st.system.trustTailnet}
          onChange={(e) => save({ trustTailnet: e.currentTarget.checked }, "Configurações da tailnet salvas.")}
        />
        <span>
          <strong>Confiar nas máquinas da tailnet</strong>
          <span className="hint">
            {st.system.token
              ? "Quem chega pela sua tailnet entra sem o token. Desligue para pedir o token a todos."
              : "Só faz diferença quando o servidor usa token (--token); hoje ele está aberto."}
          </span>
        </span>
      </label>
      <form
        className="set-grid"
        onSubmit={(e) => {
          e.preventDefault();
          const peers = (text(e.currentTarget, "peers") ?? "").split("\n").map((l) => l.trim()).filter(Boolean);
          save({ name: text(e.currentTarget, "name"), peers }, "Configurações da tailnet salvas.");
        }}
      >
        <Field id="set-name" label="Nome deste nó" hint="Como os outros nós o veem na descoberta e a interface nomeia os modelos dele." flag={locked(st, "name")}>
          <input id="set-name" name="name" defaultValue={shown(st, "name")} disabled={locked(st, "name")} placeholder={st.system.autoName} maxLength={64} />
        </Field>
        <Field id="set-peers" label="Peers manuais" hint="Um endereço por linha, na forma host:porta; desligados no Tailscale também podem ficar aqui." flag={false}>
          <textarea id="set-peers" name="peers" rows={3} defaultValue={(st.saved.peers ?? []).join("\n")} placeholder="10.1.2.3:8080" />
        </Field>
        {flagPeers.length > 0 && <p className="flag-note">Também da linha de comando: {flagPeers.join(", ")}.</p>}
        <div className="editor-actions">
          <button className="solid">Salvar</button>
        </div>
      </form>

      <div className="discovered">
        <h3>Nós descobertos</h3>
        {st.discovered.length === 0 && <p className="hint">Nenhum outro nó do tailnet respondeu com lair.</p>}
        <ul className="skill-list">
          {st.discovered.map((n) => (
            <li key={n.name}>
              <div className="skill-main">
                <span className="skill-name">{n.name}</span>
                <span className="badge">v{n.version}</span>
                <span className="skill-desc">
                  {n.models.length === 0 ? "sem modelos" : `${n.models.length} modelo(s): ${n.models.map(modelName).join(", ")}`}
                </span>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

function System({ st }: { st: Settings }) {
  return (
    <div className="set-grid">
      <h3>Este servidor</h3>
      <dl className="system-facts">
        <div>
          <dt>Versão</dt>
          <dd>{st.system.version}</dd>
        </div>
        <div>
          <dt>Token de acesso</dt>
          <dd>{st.system.token ? "ligado" : "desligado (rede local)"}</dd>
        </div>
        <div>
          <dt>Workspace do agente</dt>
          <dd className="mono">{st.system.workspace}</dd>
        </div>
      </dl>
      <h3>Modelos</h3>
      <ul className="skill-list">
        {st.system.models.map((m) => (
          <li key={m.id}>
            <div className="skill-main">
              <span className="skill-name">{m.id}</span>
              {m.thinks && <span className="badge">raciocina</span>}
              {m.loaded && <span className="badge is-live">em memória</span>}
            </div>
          </li>
        ))}
      </ul>
      {st.system.models.length === 0 && <p className="hint">Nenhum modelo .gguf encontrado nesta máquina.</p>}
      <p className="hint">
        Porta, contexto e GPU são escolhidos pelo comando que inicia o lair; veja <code>lair serve --help</code> para o restante.
      </p>
    </div>
  );
}
