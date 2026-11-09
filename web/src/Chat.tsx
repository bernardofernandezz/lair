import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import { AuthError, type Api, type Conversation, type Effort, type ModelInfo, type Ref, type StoredMessage } from "./api";
import { Composer } from "./Composer";
import { isEffort, ModelPicker, ReasoningControl } from "./Controls";
import type { OrbSignal } from "./Orb";
import { load, save } from "./storage";
import { Turns, type Turn } from "./Turns";

type Props = {
  api: Api;
  models: ModelInfo[];
  /** `null` for a new conversation, created on the first message. */
  conversationId: string | null;
  conversations: Conversation[];
  signal: RefObject<OrbSignal>;
  onActive: (active: boolean) => void;
  onCreated: (id: string) => void;
  onTurnEnd: () => void;
  onNew: () => void;
  onAuthLost: () => void;
};

const toTurn = (m: StoredMessage): Turn => ({
  role: m.role === "user" ? "user" : "assistant",
  content: m.content,
  reasoning: m.reasoning ?? undefined,
  skill: m.skill,
  refs: m.refs,
});

export function Chat({ api, models: initialModels, conversationId, conversations, signal, onActive, onCreated, onTurnEnd, onNew, onAuthLost }: Props) {
  const [models, setModels] = useState(initialModels);
  const [model, setModel] = useState(() => {
    const saved = load("model");
    return (initialModels.find((m) => m.id === saved) ?? initialModels.find((m) => m.loaded) ?? initialModels[0]).id;
  });
  const [effort, setEffort] = useState<Effort>(() => {
    const saved = load("effort");
    return isEffort(saved) ? saved : "high";
  });
  const [turns, setTurns] = useState<Turn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const conv = useRef(conversationId);
  const abort = useRef<AbortController | null>(null);
  const log = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  const current = models.find((m) => m.id === model) ?? models[0];
  const active = turns.length > 0;
  useEffect(() => onActive(active), [active, onActive]);
  useEffect(() => () => abort.current?.abort(), []);
  useEffect(() => save("model", model), [model]);
  useEffect(() => save("effort", effort), [effort]);

  useEffect(() => {
    if (!conversationId) return;
    api.read(conversationId).then(
      ({ conversation: c, messages }) => {
        setTurns(messages.filter((m) => m.role !== "tool").map(toTurn));
        if (c.model && initialModels.some((m) => m.id === c.model)) setModel(c.model);
        if (c.effort) setEffort(c.effort);
      },
      (e) => (e instanceof AuthError ? onAuthLost() : onNew()),
    );
    // Loads once per mount: App remounts this component when the user opens another conversation.
  }, []);

  useLayoutEffect(() => {
    const el = log.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [turns]);

  async function send(text: string, refs: Ref[]) {
    if (busy) return;
    const ctl = new AbortController();
    abort.current = ctl;
    pinned.current = true;
    setError(null);
    setBusy(true);
    const before = [...turns, { role: "user", content: text, refs } as Turn];
    const reply: Turn = { role: "assistant", content: "", reasoning: "" };
    const show = () => setTurns([...before, { ...reply }]);
    show();
    let thinkStart = 0;
    try {
      if (!conv.current) {
        conv.current = (await api.create({ model, effort })).id;
        onCreated(conv.current);
      }
      await api.turn(conv.current, { content: text, refs: refs.map((r) => r.id), model, effort }, ctl.signal, {
        skill: (s) => {
          reply.skill = s;
          show();
        },
        thinking: (t) => {
          thinkStart ||= performance.now();
          reply.reasoning += t;
          signal.current.thinking = true;
          signal.current.pulse = Math.min(1.6, signal.current.pulse + 0.3);
          show();
        },
        answer: (t) => {
          if (thinkStart && reply.thinkMs === undefined) reply.thinkMs = performance.now() - thinkStart;
          reply.content += t;
          signal.current.thinking = false;
          signal.current.pulse = Math.min(1.6, signal.current.pulse + 0.3);
          show();
        },
        done: () => {},
      });
    } catch (e) {
      if (e instanceof AuthError) return onAuthLost();
      // The server already saved the question, so it stays on screen; only an empty reply is dropped.
      if (!reply.content && !reply.reasoning) setTurns(before);
      if (!ctl.signal.aborted) setError(`Não foi possível responder: ${(e as Error).message}`);
    } finally {
      signal.current.thinking = false;
      setBusy(false);
      abort.current = null;
      onTurnEnd();
      api.models().then(setModels, () => {});
    }
  }

  return (
    <main className={`chat${active ? "" : " is-empty"}`}>
      {active && (
        <button className="ghost actions" onClick={onNew}>
          Nova conversa
        </button>
      )}

      <div
        className="log"
        ref={log}
        onScroll={(e) => {
          const el = e.currentTarget;
          pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
        }}
      >
        <Turns turns={turns} live={busy} loading={!current.loaded && current.id} />
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </div>

      <Composer
        api={api}
        conversations={conversations}
        current={conv.current}
        busy={busy}
        hello={!active && "Escolha um modelo e pergunte qualquer coisa. Tudo roda nesta máquina."}
        onSend={send}
        onStop={() => abort.current?.abort()}
        tools={
          <>
            <ModelPicker models={models} value={model} onChange={setModel} disabled={busy} />
            <ReasoningControl value={effort} onChange={setEffort} available={current.thinks} />
          </>
        }
      />
    </main>
  );
}
