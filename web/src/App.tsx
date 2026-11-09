import { useCallback, useEffect, useRef, useState } from "react";
import { AuthError, createSession, listModels, type ModelInfo } from "./api";
import { Chat } from "./Chat";
import { Gate } from "./Gate";
import { Orb, type OrbSignal } from "./Orb";
import { load, save } from "./storage";

type Phase =
  | { kind: "connecting" }
  | { kind: "gate"; error?: string; attempt: number }
  | { kind: "offline" }
  | { kind: "ready"; token: string; models: ModelInfo[] };

export function App() {
  const signal = useRef<OrbSignal>({ pulse: 0, thinking: false });
  const [phase, setPhase] = useState<Phase>({ kind: "connecting" });
  const [active, setActive] = useState(false);

  const connect = useCallback(async (token: string) => {
    try {
      const models = await listModels(token);
      save("token", token);
      await createSession(token).catch(() => {});
      setPhase({ kind: "ready", token, models });
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
        <Chat
          token={phase.token}
          models={phase.models}
          signal={signal}
          onActive={setActive}
          onAuthLost={() => {
            save("token", "");
            setActive(false);
            setPhase({ kind: "gate", error: "O token deixou de valer. Entre de novo.", attempt: 0 });
          }}
        />
      )}
    </>
  );
}
