import { useState } from "react";
import type { ToolStep } from "../parts";

const STATUS: Record<ToolStep["status"], string> = {
  pending: "Aguardando você",
  running: "Executando",
  ok: "Feito",
  error: "Falhou",
  denied: "Negado",
};

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
          <span className="tool-name">{step.label}</span>
          <code className="tool-subject">{step.subject}</code>
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
