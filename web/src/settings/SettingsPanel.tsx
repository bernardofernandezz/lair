import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import type { Api, Config, Settings } from "../api";
import { modelName } from "../api";
import { isEffort, MODES } from "../Controls";
import { SkillsTab } from "../skills/SkillsPanel";
import { ModelsTab } from "./ModelsTab";
import { pick } from "../i18n";

const T = pick({
  en: {
    tabs: { general: "General", models: "Models", tailnet: "Tailnet", skills: "Skills", system: "System" },
    settings: "Settings", close: "Close",
    intro: "Changes apply right away and are kept for the next starts. Anything set on the command line only changes by restarting the server without the flag.",
    flagNote: "Set on the command line; it stays until the server restarts without the flag.",
    indexSaved: "Skills index saved.", index: "Skills index", indexHint: "Link to a JSON listing skills to browse and install; empty means no index.", save: "Save",
    generalSaved: "General settings saved.", searxng: "Web search (SearXNG)", searxngHint: "Link to a SearXNG instance set to answer in JSON. Without one, the agent can't search the web.",
    reasoning: "Default reasoning", reasoningHint: "Used when a chat doesn't pick a reasoning level.",
    connected: "Connected to the tailnet", needsLogin: "needs login", stopped: "stopped", noTailscale: "Tailscale not found", copied: "Copied", copy: "Copy",
    localOnly1: "This lair only listens on ", localOnly2: ", so other devices can't reach that address yet. Restart with ", localOnly3: " (or the tailnet IP) to open it.",
    openAnywhere: "Open that address on any device in your tailnet to use this lair.",
    run1: "Run ", run2: " on this machine and reopen this tab.",
    noSocket: "The tailscaled socket exists but didn't answer. Is it running?", install1: "Install Tailscale (tailscale.com/download) and run ", install2: " to join your machines and share models between them.",
    tailSaved: "Tailnet settings saved.", trust: "Trust tailnet machines",
    trustOn: "Anyone coming through your tailnet gets in without the token. Turn off to ask everyone for the token.",
    trustOff: "Only matters when the server uses a token (--token); right now it's open.",
    nodeName: "This node's name", nodeNameHint: "How other nodes see it in discovery and how the UI names its models.",
    peers: "Manual peers", peersHint: "One address per line, as host:port; machines outside Tailscale can go here too.",
    flagPeers: "Also from the command line: ",
    discovered: "Discovered nodes", noneDiscovered: "No other tailnet node answered as lair.", noModels: "no models", nModels: (n: number, list: string) => `${n} model(s): ${list}`,
    server: "This server", version: "Version", token: "Access token", on: "on", off: "off (local network)", workspace: "Agent workspace", models: "Models",
    thinks: "reasons", vision: "sees images", loaded: "in memory", noGguf: "No .gguf model found on this machine.",
    visionHint1: "To attach images, use a vision model (Qwen2.5-VL, Gemma 3, SmolVLM…) and keep its ", visionHint2: " file in the same folder as the model.",
    flags1: "Port, context and GPU are chosen by the command that starts lair; see ", flags2: " for the rest.",
  },
  pt: {
    tabs: { general: "Geral", models: "Modelos", tailnet: "Tailnet", skills: "Skills", system: "Sistema" },
    settings: "Configurações", close: "Fechar",
    intro: "Mudanças valem na hora e ficam salvas para os próximos inícios. O que veio da linha de comando só muda reiniciando o servidor sem a flag.",
    flagNote: "Veio da linha de comando e vale até o servidor reiniciar sem a flag.",
    indexSaved: "Índice de skills salvo.", index: "Índice de skills", indexHint: "Endereço de um JSON que lista skills para navegar e instalar; vazio deixa sem índice.", save: "Salvar",
    generalSaved: "Configurações gerais salvas.", searxng: "Busca na web (SearXNG)", searxngHint: "Endereço de uma instância SearXNG configurada para responder em JSON. Sem ela, o agente não busca na web.",
    reasoning: "Raciocínio padrão", reasoningHint: "Usado quando a conversa não escolhe um nível de raciocínio.",
    connected: "Conectado à tailnet", needsLogin: "precisa de login", stopped: "parado", noTailscale: "Tailscale não encontrado", copied: "Copiado", copy: "Copiar",
    localOnly1: "Este lair só escuta em ", localOnly2: ", então outros aparelhos ainda não alcançam esse endereço. Reinicie com ", localOnly3: " (ou com o IP da tailnet) para abrir.",
    openAnywhere: "Abra esse endereço em qualquer aparelho da sua tailnet para usar este lair.",
    run1: "Rode ", run2: " nesta máquina e reabra esta aba.",
    noSocket: "O socket do tailscaled existe, mas não respondeu. Ele está rodando?", install1: "Instale o Tailscale (tailscale.com/download) e rode ", install2: " para juntar suas máquinas e compartilhar modelos entre elas.",
    tailSaved: "Configurações da tailnet salvas.", trust: "Confiar nas máquinas da tailnet",
    trustOn: "Quem chega pela sua tailnet entra sem o token. Desligue para pedir o token a todos.",
    trustOff: "Só faz diferença quando o servidor usa token (--token); hoje ele está aberto.",
    nodeName: "Nome deste nó", nodeNameHint: "Como os outros nós o veem na descoberta e a interface nomeia os modelos dele.",
    peers: "Peers manuais", peersHint: "Um endereço por linha, na forma host:porta; desligados no Tailscale também podem ficar aqui.",
    flagPeers: "Também da linha de comando: ",
    discovered: "Nós descobertos", noneDiscovered: "Nenhum outro nó do tailnet respondeu com lair.", noModels: "sem modelos", nModels: (n: number, list: string) => `${n} modelo(s): ${list}`,
    server: "Este servidor", version: "Versão", token: "Token de acesso", on: "ligado", off: "desligado (rede local)", workspace: "Workspace do agente", models: "Modelos",
    thinks: "raciocina", vision: "vê imagens", loaded: "em memória", noGguf: "Nenhum modelo .gguf encontrado nesta máquina.",
    visionHint1: "Para anexar imagens, use um modelo de visão (Qwen2.5-VL, Gemma 3, SmolVLM…) e deixe o arquivo ", visionHint2: " dele na mesma pasta do modelo.",
    flags1: "Porta, contexto e GPU são escolhidos pelo comando que inicia o lair; veja ", flags2: " para o restante.",
  },
});

export type Tab = "general" | "models" | "tailnet" | "skills" | "system";
type Props = {
  api: Api;
  open: boolean;
  onClose: () => void;
  /** The tab shown when the panel opens; the last one used otherwise. */
  tab?: Tab;
  /** A model finished downloading. */
  onModelsChanged: () => void;
};

const TABS = (["general", "models", "tailnet", "skills", "system"] as Tab[]).map((id) => ({ id, label: T.tabs[id] }));

/** Saves `change` on top of everything else the UI saved, then says `what` was saved. */
type Save = (change: Config, what: string) => Promise<void>;

export function SettingsPanel({ api, open, onClose, tab: openOn, onModelsChanged }: Props) {
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
      if (openOn) setTab(openOn);
      setNotice(null);
      refresh();
    } else if (!open && d.open) d.close();
  }, [open, refresh, openOn]);

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
          {T.settings}
        </h2>
        <button className="ghost" onClick={onClose}>
          {T.close}
        </button>
      </header>
      <p className="panel-intro">
        {T.intro}
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
      {tab === "models" && (
        <ModelsTab
          api={api}
          onInstalled={() => {
            refresh();
            onModelsChanged();
          }}
        />
      )}
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
      {flag && <p className="flag-note">{T.flagNote}</p>}
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
        save({ skillsIndex: text(e.currentTarget, "skillsIndex") }, T.indexSaved);
      }}
    >
      <Field id="set-skills-index" label={T.index} hint={T.indexHint} flag={locked(st, "skillsIndex")}>
        <input id="set-skills-index" name="skillsIndex" type="url" defaultValue={shown(st, "skillsIndex")} disabled={locked(st, "skillsIndex")} placeholder="https://…/skills.json" />
      </Field>
      <div className="editor-actions">
        <button className="solid">{T.save}</button>
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
        save({ searxng: text(e.currentTarget, "searxng"), reasoning: reasoning && isEffort(reasoning) ? reasoning : undefined }, T.generalSaved);
      }}
    >
      <Field id="set-searxng" label={T.searxng} hint={T.searxngHint} flag={locked(st, "searxng")}>
        <input id="set-searxng" name="searxng" type="url" defaultValue={shown(st, "searxng")} disabled={locked(st, "searxng")} placeholder="http://localhost:8888" />
      </Field>
      <Field id="set-reasoning" label={T.reasoning} hint={T.reasoningHint} flag={locked(st, "reasoning")}>
        <select id="set-reasoning" name="reasoning" defaultValue={st.system.reasoning} disabled={locked(st, "reasoning")}>
          {MODES.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}: {m.hint.toLowerCase()}
            </option>
          ))}
        </select>
      </Field>
      <div className="editor-actions">
        <button className="solid">{T.save}</button>
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
  const title = s?.running ? T.connected : s ? `Tailscale ${s.state === "NeedsLogin" ? T.needsLogin : T.stopped}` : T.noTailscale;
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
                {copied ? T.copied : T.copy}
              </button>
            </div>
          )}
          {localOnly ? (
            <p className="hint tail-warn">
              {T.localOnly1}
              <code>{st.system.addr}</code>
              {T.localOnly2}
              <code>lair serve --host 0.0.0.0</code>
              {T.localOnly3}
            </p>
          ) : (
            <p className="hint">{T.openAnywhere}</p>
          )}
        </>
      )}
      {s && !s.running && (
        <p className="hint">
          {T.run1}
          <code>sudo tailscale up</code>
          {T.run2}
        </p>
      )}
      {!s && (
        <p className="hint">
          {st.tailscale.socket ? T.noSocket : T.install1}
          {!st.tailscale.socket && <code>sudo tailscale up</code>}
          {!st.tailscale.socket && T.install2}
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
          onChange={(e) => save({ trustTailnet: e.currentTarget.checked }, T.tailSaved)}
        />
        <span>
          <strong>{T.trust}</strong>
          <span className="hint">
            {st.system.token
              ? T.trustOn
              : T.trustOff}
          </span>
        </span>
      </label>
      <form
        className="set-grid"
        onSubmit={(e) => {
          e.preventDefault();
          const peers = (text(e.currentTarget, "peers") ?? "").split("\n").map((l) => l.trim()).filter(Boolean);
          save({ name: text(e.currentTarget, "name"), peers }, T.tailSaved);
        }}
      >
        <Field id="set-name" label={T.nodeName} hint={T.nodeNameHint} flag={locked(st, "name")}>
          <input id="set-name" name="name" defaultValue={shown(st, "name")} disabled={locked(st, "name")} placeholder={st.system.autoName} maxLength={64} />
        </Field>
        <Field id="set-peers" label={T.peers} hint={T.peersHint} flag={false}>
          <textarea id="set-peers" name="peers" rows={3} defaultValue={(st.saved.peers ?? []).join("\n")} placeholder="10.1.2.3:8080" />
        </Field>
        {flagPeers.length > 0 && <p className="flag-note">
            {T.flagPeers}
            {flagPeers.join(", ")}.
          </p>}
        <div className="editor-actions">
          <button className="solid">{T.save}</button>
        </div>
      </form>

      <div className="discovered">
        <h3>{T.discovered}</h3>
        {st.discovered.length === 0 && <p className="hint">{T.noneDiscovered}</p>}
        <ul className="skill-list">
          {st.discovered.map((n) => (
            <li key={n.name}>
              <div className="skill-main">
                <span className="skill-name">{n.name}</span>
                <span className="badge">v{n.version}</span>
                <span className="skill-desc">
                  {n.models.length === 0 ? T.noModels : T.nModels(n.models.length, n.models.map(modelName).join(", "))}
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
      <h3>{T.server}</h3>
      <dl className="system-facts">
        <div>
          <dt>{T.version}</dt>
          <dd>{st.system.version}</dd>
        </div>
        <div>
          <dt>{T.token}</dt>
          <dd>{st.system.token ? T.on : T.off}</dd>
        </div>
        <div>
          <dt>{T.workspace}</dt>
          <dd className="mono">{st.system.workspace}</dd>
        </div>
      </dl>
      <h3>{T.models}</h3>
      <ul className="skill-list">
        {st.system.models.map((m) => (
          <li key={m.id}>
            <div className="skill-main">
              <span className="skill-name">{m.id}</span>
              {m.thinks && <span className="badge">{T.thinks}</span>}
              {m.vision && <span className="badge">{T.vision}</span>}
              {m.loaded && <span className="badge is-live">{T.loaded}</span>}
            </div>
          </li>
        ))}
      </ul>
      {st.system.models.length === 0 && <p className="hint">{T.noGguf}</p>}
      {!st.system.models.some((m) => m.vision) && (
        <p className="hint">
          {T.visionHint1}
          <code>mmproj-*.gguf</code>
          {T.visionHint2}
        </p>
      )}
      <p className="hint">
        {T.flags1}
        <code>lair serve --help</code>
        {T.flags2}
      </p>
    </div>
  );
}
