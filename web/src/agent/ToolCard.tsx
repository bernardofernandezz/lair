import { useState } from "react";
import type { ToolStep } from "../parts";

const LABELS: Record<string, string> = {
  list_dir: "Listar pasta",
  read_file: "Ler arquivo",
  write_file: "Escrever arquivo",
  edit_file: "Editar arquivo",
  delete_file: "Apagar arquivo",
  run_command: "Rodar comando",
  web_search: "Buscar na web",
  fetch_url: "Abrir página",
  search_conversations: "Buscar conversas",
  read_conversation: "Ler conversa",
};

const STATUS: Record<ToolStep["status"], string> = {
  pending: "Aguardando você",
  running: "Executando",
  ok: "Feito",
  error: "Falhou",
  denied: "Negado",
};

/** The one argument that says what the step is about: a path, a command, a query… */
function subject(args: Record<string, unknown>): string {
  for (const key of ["command", "path", "query", "url", "id"]) {
    if (typeof args[key] === "string") return args[key] as string;
  }
  return "";
}

type Props = {
  step: ToolStep;
  onDecide: (allow: boolean, always: boolean) => void;
};

export function ToolCard({ step, onDecide }: Props) {
  const [deciding, setDeciding] = useState(false);
  const pending = step.status === "pending";
  const detail = pending ? JSON.stringify(step.arguments, null, 2) : step.output;

  function decide(allow: boolean, always = false) {
    setDeciding(true);
    onDecide(allow, always);
  }

  return (
    <div className={`tool is-${step.status}`} role={pending ? "group" : undefined} aria-label={pending ? "Ação aguardando aprovação" : undefined}>
      <details open={pending}>
        <summary>
          <span className="tool-dot" aria-hidden />
          <span className="tool-name">{LABELS[step.name] ?? step.name}</span>
          <code className="tool-subject">{subject(step.arguments)}</code>
          <span className="tool-status">{STATUS[step.status]}</span>
        </summary>
        {detail && <pre className="tool-detail">{detail}</pre>}
      </details>
      {pending && (
        <div className="tool-actions">
          <button className="solid" disabled={deciding} onClick={() => decide(true)}>
            Permitir
          </button>
          <button className="ghost" disabled={deciding} onClick={() => decide(true, true)}>
            Sempre nesta conversa
          </button>
          <button className="ghost" disabled={deciding} onClick={() => decide(false)}>
            Negar
          </button>
        </div>
      )}
    </div>
  );
}
