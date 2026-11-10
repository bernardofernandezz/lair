import { useCallback, useEffect, useRef, useState } from "react";
import { AuthError, client, convPath, type Api, type Conversation, type Effort, type FileTypes, type ModelInfo } from "./api";
import { Chat } from "./Chat";
import { Sidebar } from "./conversations/Sidebar";
import { SettingsPanel, type Tab } from "./settings/SettingsPanel";
import { Gate } from "./Gate";
import { newSignal, Orb, type OrbSignal } from "./Orb";
import { watchFavicon } from "./favicon";
import { load, save } from "./storage";
import { pick } from "./i18n";

const T = pick({
  en: { offline: "The lair server didn't answer. Check that it's running.", retry: "Try again" },
  pt: { offline: "O servidor lair não respondeu. Confira se ele está rodando.", retry: "Tentar de novo" },
});

type Phase =
  | { kind: "connecting" }
  | { kind: "gate"; error?: string; attempt: number }
  | { kind: "offline" }
  | { kind: "ready"; api: Api; models: ModelInfo[]; fileTypes: FileTypes; defaultEffort: Effort };

const idFromPath = () => /^\/c\/([^/]+)/.exec(location.pathname)?.[1] ?? null;

export function App() {
  const signal = useRef<OrbSignal>(newSignal());
  const [phase, setPhase] = useState<Phase>({ kind: "connecting" });
  const [active, setActive] = useState(false);

  useEffect(() => watchFavicon(() => signal.current), []);

  const connect = useCallback(async (token: string) => {
    const api = client(token);
    try {
      const [models, fileTypes, settings] = await Promise.all([api.models(), api.fileTypes(), api.settings()]);
      save("token", token);
      await api.session().catch(() => {});
      setPhase({ kind: "ready", api, models, fileTypes, defaultEffort: settings.system.reasoning });
    } catch (e) {
      if (!(e instanceof AuthError)) return setPhase({ kind: "offline" });
      save("token", "");
      setPhase((p) => ({
        kind: "gate",
        error: token ? e.message : undefined,
        attempt: p.kind === "gate" ? p.attempt + 1 : 0,
      }));
    }
  }, []);

  useEffect(() => {
    connect(load("token"));
  }, [connect]);

  const authLost = useCallback(() => {
    save("token", "");
    setActive(false);
    setPhase({ kind: "gate", error: "O token deixou de valer. Entre de novo.", attempt: 0 });
  }, []);

  return (
    <>
      <Orb signal={signal} docked={phase.kind === "ready" && active} />
      <span className="mark">
        <span className="pearl" aria-hidden />
        lair
      </span>
      {phase.kind === "gate" && <Gate error={phase.error} attempt={phase.attempt} onSubmit={connect} />}
      {phase.kind === "offline" && (
        <div className="center notice" role="alert">
          <p>{T.offline}</p>
          <button className="ghost" onClick={() => connect(load("token"))}>
            {T.retry}
          </button>
        </div>
      )}
      {phase.kind === "ready" && (
        <Workspace api={phase.api} models={phase.models} fileTypes={phase.fileTypes} defaultEffort={phase.defaultEffort} signal={signal} onActive={setActive} onAuthLost={authLost} />
      )}
    </>
  );
}

type WorkspaceProps = {
  api: Api;
  models: ModelInfo[];
  fileTypes: FileTypes;
  defaultEffort: Effort;
  signal: React.RefObject<OrbSignal>;
  onActive: (active: boolean) => void;
  onAuthLost: () => void;
};

/** The signed-in app: conversation list on the side, the open conversation in the middle. */
function Workspace({ api, models, fileTypes, defaultEffort, signal, onActive, onAuthLost }: WorkspaceProps) {
  const [current, setCurrent] = useState(idFromPath);
  // Remounts the chat on navigation only; creating a conversation mid-turn must not cut the stream.
  const [visit, setVisit] = useState(0);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [open, setOpen] = useState(() => (load("sidebar") || (innerWidth > 900 ? "open" : "closed")) === "open");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<Tab | undefined>();
  // Bumped when a model download finishes, so the chat lists it.
  const [modelsKey, setModelsKey] = useState(0);
  const openSettings = (tab?: Tab) => {
    setSettingsTab(tab);
    setSettingsOpen(true);
  };

  const refresh = useCallback(() => {
    api.conversations().then(setConversations, (e) => e instanceof AuthError && onAuthLost());
  }, [api, onAuthLost]);

  useEffect(refresh, [refresh]);
  useEffect(() => save("sidebar", open ? "open" : "closed"), [open]);
  useEffect(() => {
    if (location.hash === "#config") setSettingsOpen(true);
  }, []);
  useEffect(() => {
    const onPop = () => {
      setCurrent(idFromPath());
      setVisit((v) => v + 1);
    };
    addEventListener("popstate", onPop);
    return () => removeEventListener("popstate", onPop);
  }, []);

  const go = useCallback(
    (id: string | null) => {
      history.pushState(null, "", id ? convPath(id) : "/");
      setCurrent(id);
      setVisit((v) => v + 1);
      if (innerWidth <= 900) setOpen(false);
    },
    [],
  );

  return (
    <div className={`workspace${open ? " has-sidebar" : ""}`}>
      <Sidebar
        api={api}
        open={open}
        onToggle={() => setOpen((o) => !o)}
        conversations={conversations}
        current={current}
        onOpen={go}
        onChanged={refresh}
        onSettings={() => openSettings()}
      />
      <SettingsPanel
        api={api}
        open={settingsOpen}
        tab={settingsTab}
        onClose={() => setSettingsOpen(false)}
        onModelsChanged={() => setModelsKey((k) => k + 1)}
      />
      <Chat
        key={visit}
        api={api}
        models={models}
        fileTypes={fileTypes}
        defaultEffort={defaultEffort}
        onOpen={go}
        conversationId={current}
        conversations={conversations}
        signal={signal}
        onActive={onActive}
        onCreated={(id) => {
          history.replaceState(null, "", convPath(id));
          setCurrent(id);
          refresh();
        }}
        onTurnEnd={refresh}
        onNew={() => go(null)}
        onAuthLost={onAuthLost}
        modelsKey={modelsKey}
        onMoreModels={() => openSettings("models")}
      />
    </div>
  );
}
