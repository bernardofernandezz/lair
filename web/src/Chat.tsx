import { useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AuthError, listModels, streamChat, type Effort, type ModelInfo, type Msg, type SkillInfo } from "./api";
import { isEffort, ModelPicker, ReasoningControl } from "./Controls";
import type { OrbSignal } from "./Orb";
import { MAX_CANVASES, SceneCard } from "./scene/SceneCard";
import { splitMessage, type Segment } from "./scene/split";
import { load, save } from "./storage";

type Turn = Msg & { reasoning?: string; thinkMs?: number; skill?: SkillInfo };

type Props = {
  token: string;
  models: ModelInfo[];
  signal: RefObject<OrbSignal>;
  onActive: (active: boolean) => void;
  onAuthLost: () => void;
};

function initialModel(models: ModelInfo[]) {
  const saved = load("model");
  return (models.find((m) => m.id === saved) ?? models.find((m) => m.loaded) ?? models[0]).id;
}

export function Chat({ token, models: initialModels, signal, onActive, onAuthLost }: Props) {
  const [models, setModels] = useState(initialModels);
  const [model, setModel] = useState(() => initialModel(initialModels));
  const [effort, setEffort] = useState<Effort>(() => {
    const saved = load("effort");
    return isEffort(saved) ? saved : "high";
  });
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const run = useRef(0);
  const abort = useRef<AbortController | null>(null);
  const log = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const pinned = useRef(true);

  const current = models.find((m) => m.id === model)!;
  const active = turns.length > 0;
  const segments = useMemo(() => turns.map((t) => (t.role === "assistant" ? splitMessage(t.content) : [])), [turns]);
  const sceneStarts = useMemo(() => {
    let seen = 0;
    return segments.map((segs) => {
      const start = seen;
      seen += segs.filter((s) => s.kind === "scene").length;
      return start;
    });
  }, [segments]);
  useEffect(() => onActive(active), [active, onActive]);
  useEffect(() => () => abort.current?.abort(), []);
  useEffect(() => save("model", model), [model]);
  useEffect(() => save("effort", effort), [effort]);

  useLayoutEffect(() => {
    const el = log.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [turns]);

  async function send() {
    const text = draft.trim();
    if (!text || busy) return;
    const before = turns;
    const history: Msg[] = [...turns.map(({ role, content }) => ({ role, content })), { role: "user", content: text }];
    const id = ++run.current;
    const ctl = new AbortController();
    abort.current = ctl;
    pinned.current = true;
    setTurns([...before, { role: "user", content: text }, { role: "assistant", content: "" }]);
    setDraft("");
    setError(null);
    setBusy(true);

    const reply: Turn = { role: "assistant", content: "", reasoning: "" };
    let thinkStart = 0;
    try {
      await streamChat(
        { model, effort, messages: history },
        token,
        ctl.signal,
        (d) => {
          if (run.current !== id) return;
          if (d.reasoning) {
            thinkStart ||= performance.now();
            reply.reasoning += d.reasoning;
          }
          if (d.content) {
            if (thinkStart && reply.thinkMs === undefined) reply.thinkMs = performance.now() - thinkStart;
            reply.content += d.content;
          }
          signal.current.thinking = !reply.content;
          signal.current.pulse = Math.min(1.6, signal.current.pulse + 0.3);
          setTurns([...before, { role: "user", content: text }, { ...reply }]);
        },
        (skill) => {
          if (run.current !== id) return;
          reply.skill = skill;
          setTurns([...before, { role: "user", content: text }, { ...reply }]);
        },
      );
    } catch (e) {
      if (run.current !== id) return;
      if (e instanceof AuthError) return onAuthLost();
      if (!ctl.signal.aborted || !(reply.content || reply.reasoning)) {
        // Nothing usable came back: undo the turn and give the text back, like the terminal chat does.
        setTurns(before);
        setDraft(text);
        if (!ctl.signal.aborted) setError(`Não foi possível responder: ${(e as Error).message}`);
      }
    } finally {
      signal.current.thinking = false;
      if (run.current === id) {
        setBusy(false);
        abort.current = null;
        input.current?.focus();
        listModels(token).then(setModels, () => {});
      }
    }
  }

  function reset() {
    run.current++;
    abort.current?.abort();
    abort.current = null;
    signal.current.thinking = false;
    setTurns([]);
    setError(null);
    setBusy(false);
    input.current?.focus();
  }

  return (
    <main className={`chat${active ? "" : " is-empty"}`}>
      {active && (
        <button className="ghost actions" onClick={reset}>
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
        <ol className="turns">
          {turns.map((t, i) =>
            t.role === "user" ? (
              <li key={i} className="turn user">
                {t.content}
              </li>
            ) : (
              <Assistant
                key={i}
                turn={t}
                segments={segments[i]}
                sceneStart={sceneStarts[i]}
                live={busy && i === turns.length - 1}
                loading={!current.loaded && current.id}
              />
            ),
          )}
        </ol>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </div>

      <form
        className="field composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        {!active && <p className="hello">Escolha um modelo e pergunte qualquer coisa. Tudo roda nesta máquina.</p>}
        <textarea
          ref={input}
          rows={1}
          autoFocus
          value={draft}
          placeholder="Escreva uma mensagem"
          aria-label="Mensagem"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            }
          }}
        />
        <div className="tools">
          <ModelPicker models={models} value={model} onChange={setModel} disabled={busy} />
          <ReasoningControl value={effort} onChange={setEffort} available={current.thinks} />
          {busy ? (
            <button type="button" className="send" onClick={() => abort.current?.abort()} aria-label="Parar resposta">
              <svg viewBox="0 0 24 24" aria-hidden>
                <rect x="7" y="7" width="10" height="10" rx="2" />
              </svg>
            </button>
          ) : (
            <button className="send" disabled={!draft.trim()} aria-label="Enviar">
              <svg viewBox="0 0 24 24" aria-hidden>
                <path d="M12 19V5M6 11l6-6 6 6" />
              </svg>
            </button>
          )}
        </div>
      </form>
    </main>
  );
}

function Assistant({
  turn,
  segments,
  sceneStart,
  live,
  loading,
}: {
  turn: Turn;
  segments: Segment[];
  sceneStart: number;
  live: boolean;
  loading: string | false;
}) {
  const thinking = live && !turn.content;
  let scene = sceneStart;
  return (
    <li className="turn assistant" aria-busy={live}>
      {live && !turn.reasoning && !turn.content && (
        <p className="status">{loading ? `Carregando ${loading}` : "Preparando a resposta"}</p>
      )}
      {turn.reasoning && <Reasoning text={turn.reasoning} thinking={thinking} ms={turn.thinkMs} />}
      {turn.skill && (
        <p className="skills" aria-label="Skill ativada">
          <span className="skill">
            <span className="skill-dot" aria-hidden />
            {turn.skill.name}
          </span>
        </p>
      )}
      {turn.content && (
        <div className="answer" aria-live={live ? "polite" : undefined}>
          {segments.map((segment, i) =>
            segment.kind === "md" ? (
              <Markdown key={i} remarkPlugins={[remarkGfm]}>
                {segment.text}
              </Markdown>
            ) : (
              <SceneCard
                key={i}
                source={segment.source}
                closed={segment.closed}
                render={scene++ < MAX_CANVASES}
              />
            ),
          )}
        </div>
      )}
    </li>
  );
}

/** The model's reasoning, kept visually apart from its answer: open while it thinks, folded once it answers. */
function Reasoning({ text, thinking, ms }: { text: string; thinking: boolean; ms?: number }) {
  const [open, setOpen] = useState(thinking);
  const body = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!thinking) setOpen(false);
  }, [thinking]);
  useLayoutEffect(() => {
    if (thinking && body.current) body.current.scrollTop = body.current.scrollHeight;
  }, [text, thinking]);

  const label = thinking ? "Pensando" : ms ? `Pensou por ${Math.max(1, Math.round(ms / 1000))} s` : "Raciocínio";
  return (
    <details className={`reasoning${thinking ? " is-live" : ""}`} open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>
        <span className={thinking ? "thinking" : undefined}>{label}</span>
      </summary>
      <div className="reasoning-body" ref={body}>
        {text.trim()}
      </div>
    </details>
  );
}
