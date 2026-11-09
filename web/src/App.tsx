import { useCallback, useEffect, useRef, useState } from "react";
import { AuthError, client, type Api, type Conversation, type ModelInfo } from "./api";
import { Chat } from "./Chat";
import { Sidebar } from "./conversations/Sidebar";
import { Gate } from "./Gate";
import { Orb, type OrbSignal } from "./Orb";
import { load, save } from "./storage";

type Phase =
  | { kind: "connecting" }
  | { kind: "gate"; error?: string; attempt: number }
  | { kind: "offline" }
  | { kind: "ready"; api: Api; models: ModelInfo[] };

const idFromPath = () => /^\/c\/([^/]+)/.exec(location.pathname)?.[1] ?? null;

export function App() {
  const signal = useRef<OrbSignal>({ pulse: 0, thinking: false });
  const [phase, setPhase] = useState<Phase>({ kind: "connecting" });
  const [active, setActive] = useState(false);

  const connect = useCallback(async (token: string) => {
    const api = client(token);
    try {
      const models = await api.models();
      save("token", token);
      await api.session().catch(() => {});
      setPhase({ kind: "ready", api, models });
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
      <span className="mark">lair</span>
      {phase.kind === "gate" && <Gate error={phase.error} attempt={phase.attempt} onSubmit={connect} />}
      {phase.kind === "offline" && (
        <div className="center notice" role="alert">
          <p>O servidor lair não respondeu. Confira se ele está rodando.</p>
          <button className="ghost" onClick={() => connect(load("token"))}>
            Tentar de novo
          </button>
        </div>
      )}
      {phase.kind === "ready" && (
        <Workspace api={phase.api} models={phase.models} signal={signal} onActive={setActive} onAuthLost={authLost} />
      )}
    </>
  );
}

type WorkspaceProps = {
  api: Api;
  models: ModelInfo[];
  signal: React.RefObject<OrbSignal>;
  onActive: (active: boolean) => void;
  onAuthLost: () => void;
};

/** The signed-in app: conversation list on the side, the open conversation in the middle. */
function Workspace({ api, models, signal, onActive, onAuthLost }: WorkspaceProps) {
  const [current, setCurrent] = useState(idFromPath);
  // Remounts the chat on navigation only; creating a conversation mid-turn must not cut the stream.
  const [visit, setVisit] = useState(0);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [open, setOpen] = useState(() => (load("sidebar") || (innerWidth > 900 ? "open" : "closed")) === "open");

  const refresh = useCallback(() => {
    api.conversations().then(setConversations, (e) => e instanceof AuthError && onAuthLost());
  }, [api, onAuthLost]);

  useEffect(refresh, [refresh]);
  useEffect(() => save("sidebar", open ? "open" : "closed"), [open]);
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
      history.pushState(null, "", id ? `/c/${id}` : "/");
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
      />
      <Chat
        key={visit}
        api={api}
        models={models}
        conversationId={current}
        conversations={conversations}
        signal={signal}
        onActive={onActive}
        onCreated={(id) => {
          history.replaceState(null, "", `/c/${id}`);
          setCurrent(id);
        }}
        onTurnEnd={refresh}
        onNew={() => go(null)}
        onAuthLost={onAuthLost}
      />
    </div>
  );
}
