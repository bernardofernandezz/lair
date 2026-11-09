import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { parseScene } from "./schema";
import { SceneView } from "./Scene";

/** At most this many WebGL canvases are mounted in one conversation. */
export const MAX_CANVASES = 6;

type Props = {
  source: string;
  closed: boolean;
  /** False once the conversation already used all canvas slots. */
  render: boolean;
};

/**
 * One inline 3D scene. Renders while the fence is still open when the JSON is
 * already valid, so the scene appears as the model finishes writing it.
 */
export function SceneCard({ source, closed, render }: Props) {
  const parsed = useMemo(() => parseScene(source), [source]);
  const [manual, setManual] = useState<boolean | null>(null);
  const [resetKey, setResetKey] = useState(0);
  const [full, setFull] = useState(false);
  const [assetError, setAssetError] = useState<string | null>(null);
  const frame = useRef<HTMLDivElement>(null);

  const spin = manual ?? (!parsed.ok || parsed.scene.autoRotate);

  useEffect(() => {
    const update = () => setFull(document.fullscreenElement === frame.current);
    document.addEventListener("fullscreenchange", update);
    return () => document.removeEventListener("fullscreenchange", update);
  }, []);

  const onAssetError = useCallback((id: string, message: string) => {
    setAssetError(`${id}: ${message}`);
  }, []);

  const title = (parsed.ok && parsed.scene.title) || "Cena 3D";
  const pending = !closed && !parsed.ok;

  function toggleFullscreen() {
    if (full) void document.exitFullscreen();
    else frame.current?.requestFullscreen?.().catch(() => {});
  }

  return (
    <figure className={`scene${full ? " is-full" : ""}`} ref={frame}>
      <header className="scene-head">
        <span className="scene-title" title={title}>
          {title}
        </span>
        <span className="scene-badge">3D</span>
        <span className="scene-actions">
          {render && parsed.ok && (
            <button type="button" className="ghost" onClick={() => setManual(!spin)} aria-pressed={spin}>
              {spin ? "Pausar" : "Girar"}
            </button>
          )}
          {render && parsed.ok && (
            <button
              type="button"
              className="ghost"
              onClick={() => {
                setResetKey((k) => k + 1);
                setManual(null);
              }}
            >
              Reiniciar
            </button>
          )}
          {render && parsed.ok && (
            <button type="button" className="ghost" onClick={toggleFullscreen}>
              {full ? "Sair" : "Tela cheia"}
            </button>
          )}
        </span>
      </header>

      {pending && (
        <div className="scene-pending" role="status">
          <span className="scene-spinner" aria-hidden />
          Montando a cena…
        </div>
      )}

      {!pending && !parsed.ok && (
        <p className="scene-error" role="alert">
          Não foi possível montar a cena: {parsed.error}
        </p>
      )}

      {parsed.ok && !render && (
        <p className="scene-note">
          Limite de {MAX_CANVASES} cenas interativas nesta conversa. Esta ficou só como JSON; peça uma nova cena se
          quiser vê-la em 3D.
        </p>
      )}

      {parsed.ok && render && (
        <div className="scene-stage">
          <SceneView
            scene={parsed.scene}
            autoRotate={spin}
            resetKey={resetKey}
            onAssetError={onAssetError}
          />
        </div>
      )}

      {assetError && (
        <p className="scene-error" role="alert">
          Modelo indisponível — {assetError}
        </p>
      )}

      {parsed.ok && (
        <details className="scene-json">
          <summary>Ver JSON</summary>
          <pre>{JSON.stringify(parsed.scene, null, 2)}</pre>
        </details>
      )}
    </figure>
  );
}
