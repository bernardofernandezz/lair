import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const cache = new Map<string, Promise<THREE.Group>>();

/** Loads a local GLB/GLTF from the server. `id` was validated by `parseScene`; the server checks it again. */
export function loadAsset(id: string): Promise<THREE.Group> {
  const cached = cache.get(id);
  if (cached) return cached;
  const url = `/v1/assets/${id.split("/").map(encodeURIComponent).join("/")}`;

  const manager = new THREE.LoadingManager();
  manager.setURLModifier((raw) => {
    if (raw.startsWith("data:") || raw.startsWith("blob:")) return raw;
    try {
      const u = new URL(raw, window.location.href);
      if (u.origin === window.location.origin && u.pathname.startsWith("/v1/assets/")) return u.href;
    } catch {}
    throw new Error(`asset externo bloqueado: ${raw}`);
  });

  const loader = new GLTFLoader(manager);
  const promise = new Promise<THREE.Group>((resolve, reject) => {
    loader.load(
      url,
      (gltf) => resolve(gltf.scene),
      undefined,
      (e) => {
        cache.delete(id);
        reject(new Error(friendly(e)));
      },
    );
  });
  cache.set(id, promise);
  return promise;
}

function friendly(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e);
  if (/draco|meshopt|ktx2/i.test(msg)) return "This model uses compression (Draco/meshopt/KTX2), which is not supported.";
  return msg || "failed to load the model";
}
