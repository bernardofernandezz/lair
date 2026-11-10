import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type * as THREE from "three";
import { parseScene } from "./schema";
import { HELPER, SceneView } from "./Scene";
import { pick } from "../i18n";

const T = pick({
  en: { scene: "3D scene", pause: "Pause", spin: "Spin", reset: "Reset", copied: "JSON copied.", copyFailed: "Couldn't copy.", copy: "Copy JSON", exportFailed: "Couldn't export: ", glb: "Download .glb", exit: "Exit", full: "Full screen", building: "Building the scene…", failed: "Couldn't build the scene: ", limit: (n: number) => `Limit of ${n} interactive scenes in this chat. This one stays as JSON; ask for a new scene to see it in 3D.`, unavailable: "Model unavailable — ", json: "View JSON" },
  pt: { scene: "Cena 3D", pause: "Pausar", spin: "Girar", reset: "Reiniciar", copied: "JSON copiado.", copyFailed: "Não foi possível copiar.", copy: "Copiar JSON", exportFailed: "Não foi possível exportar: ", glb: "Baixar .glb", exit: "Sair", full: "Tela cheia", building: "Montando a cena…", failed: "Não foi possível montar a cena: ", limit: (n: number) => `Limite de ${n} cenas interativas nesta conversa. Esta ficou só como JSON; peça uma nova cena se quiser vê-la em 3D.`, unavailable: "Modelo indisponível — ", json: "Ver JSON" },
});

/** Saves the scene as a binary glTF, without the grid and shadow catcher. */
async function downloadGlb(scene: THREE.Scene, title: string) {
  const { GLTFExporter } = await import("three/examples/jsm/exporters/GLTFExporter.js");
  const helpers = scene.children.filter((o) => o.name === HELPER && o.visible);
  helpers.forEach((o) => (o.visible = false));
  try {
    const glb = (await new GLTFExporter().parseAsync(scene, { binary: true, onlyVisible: true })) as ArrayBuffer;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([glb], { type: "model/gltf-binary" }));
    a.download = `${title.replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-|-$/g, "") || "cena"}.glb`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
  } finally {
    helpers.forEach((o) => (o.visible = true));
  }
}

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
  const three = useRef<THREE.Scene | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const spin = manual ?? (!parsed.ok || parsed.scene.autoRotate);

  useEffect(() => {
    const update = () => setFull(document.fullscreenElement === frame.current);
    document.addEventListener("fullscreenchange", update);
    return () => document.removeEventListener("fullscreenchange", update);
  }, []);

  const onAssetError = useCallback((id: string, message: string) => {
    setAssetError(`${id}: ${message}`);
  }, []);

  const title = (parsed.ok && parsed.scene.title) || T.scene;
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
              {spin ? T.pause : T.spin}
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
              {T.reset}
            </button>
          )}
          {parsed.ok && (
            <button
              type="button"
              className="ghost"
              onClick={() =>
                navigator.clipboard.writeText(source.trim()).then(
                  () => setNote(T.copied),
                  () => setNote(T.copyFailed),
                )
              }
            >
              {T.copy}
            </button>
          )}
          {render && parsed.ok && (
            <button
              type="button"
              className="ghost"
              onClick={() => three.current && downloadGlb(three.current, title).catch((e) => setNote(T.exportFailed + (e as Error).message))}
            >
              {T.glb}
            </button>
          )}
          {render && parsed.ok && (
            <button type="button" className="ghost" onClick={toggleFullscreen}>
              {full ? T.exit : T.full}
            </button>
          )}
        </span>
      </header>

      {pending && (
        <div className="scene-pending" role="status">
          <span className="scene-spinner" aria-hidden />
          {T.building}
        </div>
      )}

      {!pending && !parsed.ok && (
        <p className="scene-error" role="alert">
          {T.failed}
          {parsed.error}
        </p>
      )}

      {parsed.ok && !render && (
        <p className="scene-note">
          {T.limit(MAX_CANVASES)}
        </p>
      )}

      {parsed.ok && render && (
        <div className="scene-stage">
          <SceneView
            scene={parsed.scene}
            autoRotate={spin}
            resetKey={resetKey}
            onAssetError={onAssetError}
            onReady={(s) => (three.current = s)}
          />
        </div>
      )}

      {note && (
        <p className="scene-note" role="status">
          {note}
        </p>
      )}

      {assetError && (
        <p className="scene-error" role="alert">
          {T.unavailable}
          {assetError}
        </p>
      )}

      {parsed.ok && (
        <details className="scene-json">
          <summary>{T.json}</summary>
          <pre>{JSON.stringify(parsed.scene, null, 2)}</pre>
        </details>
      )}
    </figure>
  );
}
