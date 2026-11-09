import JSON5 from "json5";

export type Vec3 = [number, number, number];

export type Geometry =
  | { kind: "box"; size: Vec3 }
  | { kind: "sphere"; radius: number; segments: number }
  | { kind: "cylinder"; radiusTop: number; radiusBottom: number; height: number; segments: number }
  | { kind: "cone"; radius: number; height: number; segments: number }
  | { kind: "torus"; radius: number; tube: number; segments: [number, number] }
  | { kind: "plane"; size: [number, number] }
  | { kind: "capsule"; radius: number; length: number }
  | { kind: "icosahedron"; radius: number; detail: number }
  | { kind: "polyhedron"; vertices: Vec3[]; faces: [number, number, number][]; flat: boolean }
  | { kind: "points"; positions: Vec3[]; colors?: string[]; size: number }
  | { kind: "lines"; positions: Vec3[]; closed: boolean; color: string };

export type Material = {
  color: string;
  metalness: number;
  roughness: number;
  opacity: number;
  emissive: string;
  emissiveIntensity: number;
  wireframe: boolean;
  flatShading: boolean;
};

export type Orbit = { radius: number; speed: number; axis: "x" | "y" | "z"; phase: number };

export type Node = {
  geometry?: Geometry;
  asset?: string;
  material: Material;
  position: Vec3;
  rotation: Vec3;
  scale: Vec3;
  spin?: Vec3;
  orbit?: Orbit;
  bob?: { amplitude: number; speed: number };
  children: Node[];
};

export type Light = {
  kind: "ambient" | "directional" | "point" | "spot";
  color: string;
  intensity: number;
  position?: Vec3;
  distance?: number;
  decay?: number;
  angle?: number;
  penumbra?: number;
};

export type Camera = { position: Vec3; target: Vec3 };

export type Scene = {
  title?: string;
  background?: string;
  camera?: Camera;
  /** Starts spinning unless the scene says `camera.autoRotate: false`. */
  autoRotate: boolean;
  lights: Light[];
  grid: boolean;
  nodes: Node[];
};

export type ParseResult = { ok: true; scene: Scene } | { ok: false; error: string };

// Limits are also told to the model in skills/3d/prompt.md; keep both in step.
const MAX_SOURCE = 256 * 1024;
const MAX_NODES = 400;
const MAX_DEPTH = 8;
const MAX_VERTICES = 2_000;
const MAX_FACES = 4_000;
const MAX_POINTS = 20_000;
const MAX_LINES = 5_000;
const MAX_ASSETS = 20;

class SceneError extends Error {}

function fail(path: string, message: string): never {
  throw new SceneError(path ? `${path}: ${message}` : message);
}

/** Parses one scene block. Unknown keys are ignored; malformed values never throw. */
export function parseScene(source: string): ParseResult {
  if (source.length > MAX_SOURCE) {
    return { ok: false, error: `Cena grande demais (${Math.round(source.length / 1024)} KB; limite 256 KB).` };
  }
  let raw: unknown;
  try {
    raw = JSON5.parse(source);
  } catch (e) {
    return { ok: false, error: `JSON inválido: ${(e as Error).message}` };
  }
  try {
    return { ok: true, scene: buildScene(raw) };
  } catch (e) {
    return { ok: false, error: e instanceof SceneError ? e.message : `Cena inválida: ${(e as Error).message}` };
  }
}

function buildScene(raw: unknown): Scene {
  const s = obj(raw, "cena");
  const nodes = arr(s.nodes, "nodes", MAX_NODES);
  if (nodes.length === 0) fail("nodes", "a cena precisa de pelo menos um node");
  const count = { n: 0 };
  const assets = new Set<string>();
  const title = s.title == null ? undefined : text(s.title, "title", 200);
  return {
    title,
    background: s.background == null ? undefined : color(s.background, "background"),
    camera: s.camera == null ? undefined : parseCamera(s.camera, "camera"),
    autoRotate: s.camera == null || bool(obj(s.camera, "camera").autoRotate, "camera.autoRotate", true),
    lights: s.lights == null ? [] : arr(s.lights, "lights", 16).map((l, i) => parseLight(l, `lights[${i}]`)),
    grid: bool(s.grid, "grid", false),
    nodes: nodes.map((n, i) => parseNode(n, `nodes[${i}]`, 1, count, assets)),
  };
}

function parseCamera(v: unknown, path: string): Camera {
  const c = obj(v, path);
  return {
    position: vec3(c.position, `${path}.position`, [4, 3, 6], -1e5, 1e5),
    target: vec3(c.target, `${path}.target`, [0, 0, 0], -1e5, 1e5),
  };
}

function parseLight(v: unknown, path: string): Light {
  const l = obj(v, path);
  const kind = text(l.kind, `${path}.kind`, 16);
  if (!["ambient", "directional", "point", "spot"].includes(kind)) {
    fail(`${path}.kind`, `luz desconhecida "${kind}"`);
  }
  const light: Light = {
    kind: kind as Light["kind"],
    color: color(l.color, `${path}.color`, "#ffffff"),
    intensity: num(l.intensity, `${path}.intensity`, 1, 0, 100),
  };
  if (kind !== "ambient") {
    light.position = vec3(l.position, `${path}.position`, [4, 8, 4], -1e5, 1e5);
  }
  if (kind === "point" || kind === "spot") {
    light.distance = num(l.distance, `${path}.distance`, 0, 0, 1e5);
    light.decay = num(l.decay, `${path}.decay`, 2, 0, 10);
  }
  if (kind === "spot") {
    light.angle = num(l.angle, `${path}.angle`, Math.PI / 3, 0, Math.PI / 2);
    light.penumbra = num(l.penumbra, `${path}.penumbra`, 0, 0, 1);
  }
  return light;
}

function parseNode(
  v: unknown,
  path: string,
  depth: number,
  count: { n: number },
  assets: Set<string>,
): Node {
  if (depth > MAX_DEPTH) fail(path, `profundidade máxima de ${MAX_DEPTH}`);
  count.n += 1;
  if (count.n > MAX_NODES) fail(path, `a cena passa de ${MAX_NODES} nodes`);
  const n = obj(v, path);
  const hasGeometry = n.geometry != null;
  const asset = n.asset == null ? undefined : assetId(n.asset, `${path}.asset`, assets);
  if (hasGeometry && asset) fail(path, "use geometry ou asset, não os dois");
  const children = n.children == null ? [] : arr(n.children, `${path}.children`, MAX_NODES);
  return {
    geometry: hasGeometry ? parseGeometry(n.geometry, `${path}.geometry`) : undefined,
    asset,
    material: parseMaterial(n.material, `${path}.material`),
    position: vec3(n.position, `${path}.position`, [0, 0, 0], -1e5, 1e5),
    rotation: vec3(n.rotation, `${path}.rotation`, [0, 0, 0], -1e4, 1e4),
    scale: vec3(n.scale, `${path}.scale`, [1, 1, 1], -1e4, 1e4),
    spin: n.spin == null ? undefined : vec3(n.spin, `${path}.spin`, [0, 0, 0], -20, 20),
    orbit: n.orbit == null ? undefined : parseOrbit(n.orbit, `${path}.orbit`),
    bob: n.bob == null ? undefined : parseBob(n.bob, `${path}.bob`),
    children: children.map((c, i) => parseNode(c, `${path}.children[${i}]`, depth + 1, count, assets)),
  };
}

function parseOrbit(v: unknown, path: string): Orbit {
  const o = obj(v, path);
  const axis = text(o.axis ?? "y", `${path}.axis`, 1).toLowerCase();
  if (axis !== "x" && axis !== "y" && axis !== "z") fail(`${path}.axis`, `eixo inválido "${axis}"`);
  return {
    radius: num(o.radius, `${path}.radius`, 2, 0, 1e5),
    speed: num(o.speed, `${path}.speed`, 0.5, -100, 100),
    axis,
    phase: num(o.phase, `${path}.phase`, 0, -1e4, 1e4),
  };
}

function parseBob(v: unknown, path: string): { amplitude: number; speed: number } {
  const b = obj(v, path);
  return {
    amplitude: num(b.amplitude, `${path}.amplitude`, 0.2, 0, 1e4),
    speed: num(b.speed, `${path}.speed`, 1, -100, 100),
  };
}

const GREY = "#8a8f9e";

function parseMaterial(v: unknown, path: string): Material {
  const m = obj(v ?? {}, path);
  return {
    color: color(m.color, `${path}.color`, GREY),
    metalness: num(m.metalness, `${path}.metalness`, 0.1, 0, 1),
    roughness: num(m.roughness, `${path}.roughness`, 0.6, 0, 1),
    opacity: num(m.opacity, `${path}.opacity`, 1, 0, 1),
    emissive: color(m.emissive, `${path}.emissive`, "#000000"),
    emissiveIntensity: num(m.emissiveIntensity, `${path}.emissiveIntensity`, 1, 0, 100),
    wireframe: bool(m.wireframe, `${path}.wireframe`, false),
    flatShading: bool(m.flatShading, `${path}.flatShading`, false),
  };
}

function parseGeometry(v: unknown, path: string): Geometry {
  const g = obj(v, path);
  const kind = text(g.kind, `${path}.kind`, 32);
  switch (kind) {
    case "box":
      return { kind, size: vec3(g.size, `${path}.size`, [1, 1, 1], 0.001, 1e4) };
    case "sphere": {
      const segments = Math.round(num(g.segments, `${path}.segments`, 32, 3, 128));
      return { kind, radius: num(g.radius, `${path}.radius`, 1, 0.001, 1e4), segments };
    }
    case "cylinder": {
      const segments = Math.round(num(g.segments, `${path}.segments`, 32, 3, 128));
      return {
        kind,
        radiusTop: num(g.radiusTop, `${path}.radiusTop`, 1, 0, 1e4),
        radiusBottom: num(g.radiusBottom, `${path}.radiusBottom`, 1, 0, 1e4),
        height: num(g.height, `${path}.height`, 1, 0.001, 1e4),
        segments,
      };
    }
    case "cone": {
      const segments = Math.round(num(g.segments, `${path}.segments`, 32, 3, 128));
      return {
        kind,
        radius: num(g.radius, `${path}.radius`, 1, 0.001, 1e4),
        height: num(g.height, `${path}.height`, 1, 0.001, 1e4),
        segments,
      };
    }
    case "torus": {
      const raw = g.segments;
      let segments: [number, number] = [96, 16];
      if (Array.isArray(raw)) {
        const pair = arr(raw, `${path}.segments`, 2);
        if (pair.length !== 2) fail(`${path}.segments`, "esperava [radial, tubular]");
        segments = [
          Math.round(num(pair[0], `${path}.segments[0]`, 96, 3, 256)),
          Math.round(num(pair[1], `${path}.segments[1]`, 16, 3, 64)),
        ];
      } else if (raw != null) {
        segments = [Math.round(num(raw, `${path}.segments`, 96, 3, 256)), 16];
      }
      return {
        kind,
        radius: num(g.radius, `${path}.radius`, 1, 0.001, 1e4),
        tube: num(g.tube, `${path}.tube`, 0.1, 0.001, 1e4),
        segments,
      };
    }
    case "plane":
      return { kind, size: pair2(g.size, `${path}.size`, [1, 1], 0.001, 1e4) };
    case "capsule":
      return {
        kind,
        radius: num(g.radius, `${path}.radius`, 0.5, 0.001, 1e4),
        length: num(g.length, `${path}.length`, 1, 0, 1e4),
      };
    case "icosahedron":
      return {
        kind,
        radius: num(g.radius, `${path}.radius`, 1, 0.001, 1e4),
        detail: Math.round(num(g.detail, `${path}.detail`, 0, 0, 5)),
      };
    case "polyhedron":
      return parsePolyhedron(g, path);
    case "points":
      return parsePoints(g, path);
    case "lines":
      return parseLines(g, path);
    default:
      fail(`${path}.kind`, `geometria desconhecida "${kind}"`);
  }
}

function parsePolyhedron(g: Record<string, unknown>, path: string): Geometry {
  const rawV = arr(g.vertices, `${path}.vertices`, MAX_VERTICES);
  if (rawV.length < 3) fail(`${path}.vertices`, "precisa de pelo menos 3 vértices");
  const vertices = rawV.map((v, i) => vec3(v, `${path}.vertices[${i}]`, [0, 0, 0], -1e4, 1e4));
  const rawF = arr(g.faces, `${path}.faces`, MAX_FACES);
  if (rawF.length === 0) fail(`${path}.faces`, "precisa de pelo menos uma face");
  const faces: [number, number, number][] = [];
  rawF.forEach((f, i) => {
    const idx = arr(f, `${path}.faces[${i}]`, 12);
    if (idx.length < 3) fail(`${path}.faces[${i}]`, "uma face precisa de pelo menos 3 índices");
    const points = idx.map((x, j) => index(x, `${path}.faces[${i}][${j}]`, vertices.length));
    for (let k = 1; k + 1 < points.length; k++) {
      faces.push([points[0], points[k], points[k + 1]]);
      if (faces.length > MAX_FACES) fail(`${path}.faces`, `limite de ${MAX_FACES} triângulos`);
    }
  });
  return { kind: "polyhedron", vertices, faces, flat: bool(g.flat, `${path}.flat`, false) };
}

function parsePoints(g: Record<string, unknown>, path: string): Geometry {
  const raw = arr(g.positions, `${path}.positions`, MAX_POINTS);
  if (raw.length === 0) fail(`${path}.positions`, "precisa de pelo menos um ponto");
  const positions = raw.map((p, i) => vec3(p, `${path}.positions[${i}]`, [0, 0, 0], -1e5, 1e5));
  let colors: string[] | undefined;
  if (g.colors != null) {
    const list = arr(g.colors, `${path}.colors`, MAX_POINTS);
    if (list.length !== positions.length) fail(`${path}.colors`, "precisa ter a mesma quantidade de positions");
    colors = list.map((c, i) => color(c, `${path}.colors[${i}]`, "#ffffff"));
  }
  return {
    kind: "points",
    positions,
    colors,
    size: num(g.size, `${path}.size`, 0.06, 0.001, 2),
  };
}

function parseLines(g: Record<string, unknown>, path: string): Geometry {
  const raw = arr(g.positions, `${path}.positions`, MAX_LINES);
  if (raw.length < 2) fail(`${path}.positions`, "precisa de pelo menos dois pontos");
  return {
    kind: "lines",
    positions: raw.map((p, i) => vec3(p, `${path}.positions[${i}]`, [0, 0, 0], -1e5, 1e5)),
    closed: bool(g.closed, `${path}.closed`, false),
    color: color(g.color, `${path}.color`, GREY),
  };
}

function assetId(v: unknown, path: string, assets: Set<string>): string {
  const id = text(v, path, 512);
  if (
    !/^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(id) ||
    id.split("/").some((part) => !part || part === "." || part === "..")
  ) {
    fail(path, "id de asset inválido (use caminho relativo sem ..)");
  }
  assets.add(id);
  if (assets.size > MAX_ASSETS) fail(path, `no máximo ${MAX_ASSETS} assets por cena`);
  return id;
}

function obj(v: unknown, path: string): Record<string, unknown> {
  if (!v || typeof v !== "object" || Array.isArray(v)) fail(path, "esperava um objeto");
  return v as Record<string, unknown>;
}

function arr(v: unknown, path: string, max: number): unknown[] {
  if (!Array.isArray(v)) fail(path, "esperava uma lista");
  if (v.length > max) fail(path, `limite de ${max} itens`);
  return v;
}

function num(v: unknown, path: string, def: number | undefined, min: number, max: number): number {
  if (v == null) {
    if (def === undefined) fail(path, "número obrigatório");
    return def;
  }
  if (typeof v !== "number" || !Number.isFinite(v)) fail(path, "esperava um número finito");
  return Math.min(max, Math.max(min, v));
}

function text(v: unknown, path: string, max: number): string {
  if (typeof v !== "string") fail(path, "esperava um texto");
  const s = v.trim();
  if (!s) fail(path, "texto vazio");
  if (s.length > max) fail(path, `texto com mais de ${max} caracteres`);
  return s;
}

function bool(v: unknown, path: string, def: boolean): boolean {
  if (v == null) return def;
  if (typeof v !== "boolean") fail(path, "esperava true ou false");
  return v;
}

function vec3(v: unknown, path: string, def: Vec3, min: number, max: number): Vec3 {
  if (v == null) return [...def];
  const list = arr(v, path, 3);
  if (list.length !== 3) fail(path, "esperava [x, y, z]");
  return [
    num(list[0], `${path}[0]`, undefined, min, max),
    num(list[1], `${path}[1]`, undefined, min, max),
    num(list[2], `${path}[2]`, undefined, min, max),
  ];
}

function pair2(v: unknown, path: string, def: [number, number], min: number, max: number): [number, number] {
  if (v == null) return [...def];
  const list = arr(v, path, 2);
  if (list.length !== 2) fail(path, "esperava [largura, profundidade]");
  return [num(list[0], `${path}[0]`, undefined, min, max), num(list[1], `${path}[1]`, undefined, min, max)];
}

function index(v: unknown, path: string, len: number): number {
  if (typeof v !== "number" || !Number.isInteger(v) || v < 0 || v >= len) {
    fail(path, `índice fora de 0..${len - 1}`);
  }
  return v;
}

function color(v: unknown, path: string, def = GREY): string {
  if (v == null) return def;
  if (typeof v !== "string") fail(path, "cor deve ser hexadecimal (#rrggbb)");
  const s = v.trim().toLowerCase();
  if (/^#[0-9a-f]{6}$/.test(s)) return s;
  if (/^#[0-9a-f]{3}$/.test(s)) return `#${s[1]}${s[1]}${s[2]}${s[2]}${s[3]}${s[3]}`;
  fail(path, `cor inválida "${s}" (use #rrggbb)`);
}
