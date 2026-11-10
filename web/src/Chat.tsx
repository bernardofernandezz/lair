import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import { AuthError, ConnectionLost, type Api, type Conversation, type Effort, type FileTypes, type ModelInfo, type Ref, type UploadedFile } from "./api";
import { Composer, type Pending } from "./Composer";
import { isEffort, ModelPicker, ReasoningControl } from "./Controls";
import { pulse, settle, typed, type OrbSignal } from "./Orb";
import { load, save } from "./storage";
import { Turns } from "./Turns";
import { addCall, addText, emptyReply, fromStored, setResult, snapshot, type Turn } from "./parts";

type Props = {
  api: Api;
  models: ModelInfo[];
  fileTypes: FileTypes;
  /** The server's reasoning level, used until the user picks one for this conversation. */
  defaultEffort: Effort;
  /** Opens another conversation in place. */
  onOpen: (id: string) => void;
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

export function Chat({ api, models: initialModels, fileTypes, defaultEffort, onOpen, conversationId, conversations, signal, onActive, onCreated, onTurnEnd, onNew, onAuthLost }: Props) {
  const [models, setModels] = useState(initialModels);
  const [model, setModel] = useState(() => {
    const saved = load("model");
    return (initialModels.find((m) => m.id === saved) ?? initialModels.find((m) => m.loaded) ?? initialModels[0]).id;
  });
  // `null` until the user picks a level: the server then applies its configured default.
  const [effort, setEffort] = useState<Effort | null>(() => {
    const saved = load("effort");
    return isEffort(saved) ? saved : null;
  });
  const [agent, setAgent] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const conv = useRef(conversationId);
  const abort = useRef<AbortController | null>(null);
  const log = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const alive = useRef(true);
  useEffect(() => () => void (alive.current = false), []);

  const current = models.find((m) => m.id === model) ?? models[0];
  const active = turns.length > 0;
  useEffect(() => onActive(active), [active, onActive]);
  useEffect(() => () => abort.current?.abort(), []);
  useEffect(() => save("model", model), [model]);
  useEffect(() => save("effort", effort ?? ""), [effort]);

  useEffect(() => {
    if (!conversationId) return;
    api.read(conversationId).then(
      ({ conversation: c, messages }) => {
        setTurns(fromStored(messages));
        if (c.model && initialModels.some((m) => m.id === c.model)) setModel(c.model);
        if (c.effort) setEffort(c.effort);
        setAgent(c.agent);
      },
      (e) => (e instanceof AuthError ? onAuthLost() : onNew()),
    );
    // Loads once per mount: App remounts this component when the user opens another conversation.
  }, []);

  useLayoutEffect(() => {
    const el = log.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [turns]);

  async function send(text: string, refs: Ref[], files: Pending[]) {
    if (busy) return;
    const ctl = new AbortController();
    abort.current = ctl;
    pinned.current = true;
    setError(null);
    signal.current.error = false;
    setBusy(true);
    const before: Turn[] = [...turns, { role: "user", content: text, refs, attachments: files.map(({ file, image }) => ({ name: file.name, image, src: image ? URL.createObjectURL(file) : undefined })) }];
    const reply = emptyReply();
    const show = () => setTurns([...before, snapshot(reply)]);
    show();
    let thinkStart = 0;
    try {
      if (!conv.current) {
        conv.current = (await api.create({ model, effort, agent })).id;
        onCreated(conv.current);
      }
      const attachments: UploadedFile[] = [];
      for (const { file } of files) attachments.push(await api.uploadFile(file));
      const sig = signal.current;
      await api.turn(conv.current, { content: text, refs: refs.map((r) => r.id), model, effort: effort ?? undefined, agent, attachments }, ctl.signal, (e) => {
        switch (e.event) {
          case "skill":
            reply.skill = e.data;
            break;
          case "thinking":
            thinkStart ||= performance.now();
            reply.reasoning += e.data.text;
            sig.thinking = true;
            pulse(sig);
            break;
          case "answer":
            if (thinkStart && reply.thinkMs === undefined) reply.thinkMs = performance.now() - thinkStart;
            addText(reply, e.data.text);
            sig.thinking = false;
            pulse(sig);
            break;
          case "tool_call":
            addCall(reply, e.data, e.data.needs_approval);
            sig.tool = true;
            sig.waiting = e.data.needs_approval;
            break;
          case "tool_result":
            setResult(reply, e.data);
            sig.tool = false;
            sig.waiting = false;
            pulse(sig, true);
            break;
          case "done":
            return;
        }
        show();
      });
    } catch (e) {
      if (e instanceof AuthError) return onAuthLost();
      signal.current.error = true;
      // The server already saved the question, so it stays on screen; only an empty reply is dropped.
      if (reply.parts.length === 0 && !reply.reasoning) setTurns(before);
      if (e instanceof ConnectionLost) recover();
      else if (!ctl.signal.aborted) setError(`Não foi possível responder: ${(e as Error).message}`);
    } finally {
      settle(signal.current);
      setBusy(false);
      abort.current = null;
      onTurnEnd();
      api.models().then(setModels, () => {});
    }
  }

  /** Waits for the server to answer again, then shows the conversation as it was saved. */
  async function recover() {
    setError("A conexão com o lair caiu. O que já foi gerado está salvo; esperando o servidor voltar…");
    for (let tries = 0; alive.current && tries < 90; tries++) {
      try {
        setModels(await api.models());
        if (conv.current) setTurns(fromStored((await api.read(conv.current)).messages));
        setError(null);
        signal.current.error = false;
        onTurnEnd();
        return;
      } catch {
        signal.current.error = true;
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
    if (alive.current) setError("O lair não voltou a responder. Confira se o servidor está rodando e recarregue a página.");
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
        <Turns
          turns={turns}
          live={busy}
          loading={!current.loaded && current.id}
          fileUrl={api.fileUrl}
          onDecide={(id, allow, always) => api.approve(id, { allow, always }).catch((e) => setError((e as Error).message))}
          onOpen={onOpen}
        />
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
        vision={!!current.vision}
        fileTypes={fileTypes}
        onType={() => typed(signal.current)}
        onStop={() => abort.current?.abort()}
        tools={
          <>
            <ModelPicker models={models} value={model} onChange={setModel} disabled={busy} />
            <ReasoningControl value={effort ?? defaultEffort} onChange={setEffort} available={current.thinks} />
            <button
              type="button"
              className={`chip agent-toggle${agent ? " is-on" : ""}`}
              aria-pressed={agent}
              title="Deixa o modelo usar ferramentas: arquivos do workspace, shell, web e suas conversas. Ações que mudam algo pedem sua aprovação."
              onClick={() => setAgent((a) => !a)}
            >
              Agente
            </button>
          </>
        }
      />
    </main>
  );
}
