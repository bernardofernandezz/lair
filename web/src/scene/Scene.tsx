import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Component, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { loadAsset } from "./assets";
import type { Geometry, Light, Material, Node, Scene, Vec3 } from "./schema";

type Props = {
  scene: Scene;
  autoRotate: boolean;
  resetKey: number;
  onAssetError?: (id: string, message: string) => void;
  /** The three.js scene once it exists, for exporting it. */
  onReady?: (scene: THREE.Scene) => void;
};

/** Key, fill and a little ambient; the room environment does most of the soft lighting. */
const DEFAULT_LIGHTS: Light[] = [
  { kind: "ambient", color: "#ffffff", intensity: 0.15 },
  { kind: "directional", color: "#fff4e6", intensity: 1.6, position: [5, 9, 5] },
  { kind: "directional", color: "#dfe8ff", intensity: 0.45, position: [-6, 4, -3] },
];

/** Grid and shadow catcher are view aids: exports leave them out. */
export const HELPER = "lair-helper";

export function SceneView({ scene, autoRotate, resetKey, onAssetError, onReady }: Props) {
  const reduced = useReducedMotion();
  const bounds = useMemo(() => sceneBounds(scene), [scene]);
  const home = useMemo(() => cameraHome(scene, bounds), [scene, bounds]);
  const gridSize = Math.max(10, Math.ceil(bounds.radius * 2.5));
  const gridDivisions = Math.min(48, Math.max(4, Math.round(gridSize / 2)));

  return (
    <CanvasBoundary>
      <Canvas
        className="scene-canvas-inner"
        camera={{ fov: 45, near: 0.01, far: 100_000 }}
        dpr={[1, 2]}
        shadows
        gl={{ antialias: true, alpha: !scene.background }}
        frameloop="always"
        onCreated={(state) => onReady?.(state.scene)}
      >
        {scene.background && <color attach="background" args={[scene.background]} />}
        <RoomLight />
        {(scene.lights.length ? scene.lights : DEFAULT_LIGHTS).map((light, i) => (
          <LightView key={i} light={light} reach={bounds.radius} first={i === firstShadow(scene)} />
        ))}
        {scene.grid && (
          <gridHelper
            name={HELPER}
            args={[gridSize, gridDivisions, "#3a3f4b", "#2a2e38"]}
            position={[bounds.center.x, 0, bounds.center.z]}
          />
        )}
        <mesh name={HELPER} rotation={[-Math.PI / 2, 0, 0]} position={[bounds.center.x, bounds.minY - 0.002, bounds.center.z]} receiveShadow>
          <planeGeometry args={[bounds.radius * 8, bounds.radius * 8]} />
          <shadowMaterial opacity={0.22} />
        </mesh>
        <CameraRig home={home} autoRotate={autoRotate && !reduced} resetKey={resetKey} />
        {scene.nodes.map((node, i) => (
          <NodeView key={i} node={node} reduced={reduced} onAssetError={onAssetError} />
        ))}
      </Canvas>
    </CanvasBoundary>
  );
}

/** Approximate framing for scenes without a camera: one sphere per object, moved by its transforms. */
function sceneBounds(scene: Scene): { center: THREE.Vector3; radius: number; minY: number } {
  const box = new THREE.Box3();
  const local = new THREE.Matrix4();
  const visit = (node: Node, parent: THREE.Matrix4) => {
    local.compose(
      new THREE.Vector3(...node.position),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(...node.rotation)),
      new THREE.Vector3(...node.scale),
    );
    const world = parent.clone().multiply(local);
    let radius = node.geometry ? geometryRadius(node.geometry) : node.asset ? 1 : 0;
    radius *= world.getMaxScaleOnAxis();
    if (radius > 0) {
      const center = new THREE.Vector3().setFromMatrixPosition(world);
      box.expandByPoint(center.clone().addScalar(radius));
      box.expandByPoint(center.clone().addScalar(-radius));
    }
    for (const child of node.children) visit(child, world);
  };
  for (const node of scene.nodes) visit(node, new THREE.Matrix4());
  if (box.isEmpty()) return { center: new THREE.Vector3(), radius: 1, minY: 0 };
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(1, box.getSize(new THREE.Vector3()).length() / 2);
  return { center, radius, minY: box.min.y };
}

function geometryRadius(g: Geometry): number {
  switch (g.kind) {
    case "box":
      return Math.hypot(g.size[0], g.size[1], g.size[2]) / 2;
    case "sphere":
      return g.radius;
    case "cylinder":
      return Math.hypot(Math.max(g.radiusTop, g.radiusBottom), g.height / 2);
    case "cone":
      return Math.hypot(g.radius, g.height / 2);
    case "torus":
      return g.radius + g.tube;
    case "plane":
      return Math.hypot(g.size[0], g.size[1]) / 2;
    case "capsule":
      return g.radius + g.length / 2;
    case "icosahedron":
      return g.radius;
    case "polyhedron":
      return maxLength(g.vertices);
    case "points":
    case "lines":
      return maxLength(g.positions);
  }
}

function maxLength(points: Vec3[]): number {
  let max = 0;
  for (const p of points) max = Math.max(max, Math.hypot(p[0], p[1], p[2]));
  return max;
}

function cameraHome(scene: Scene, bounds: { center: THREE.Vector3; radius: number }) {
  if (scene.camera) {
    return {
      position: new THREE.Vector3(...scene.camera.position),
      target: new THREE.Vector3(...scene.camera.target),
    };
  }
  const dir = new THREE.Vector3(0.85, 0.55, 1.3).normalize();
  const distance = Math.max(4, bounds.radius * 2.3);
  return {
    position: bounds.center.clone().add(dir.multiplyScalar(distance)),
    target: bounds.center.clone(),
  };
}

function CameraRig({
  home,
  autoRotate,
  resetKey,
}: {
  home: { position: THREE.Vector3; target: THREE.Vector3 };
  autoRotate: boolean;
  resetKey: number;
}) {
  const { camera, gl } = useThree();
  const controls = useRef<OrbitControls | null>(null);

  useEffect(() => {
    const c = new OrbitControls(camera, gl.domElement);
    c.enableDamping = true;
    c.dampingFactor = 0.08;
    c.autoRotate = autoRotate;
    c.autoRotateSpeed = 0.9;
    c.minDistance = 0.3;
    c.maxDistance = 50_000;
    controls.current = c;
    return () => {
      c.dispose();
      controls.current = null;
    };
  }, [camera, gl]);

  useEffect(() => {
    if (controls.current) controls.current.autoRotate = autoRotate;
  }, [autoRotate]);

  useEffect(() => {
    const c = controls.current;
    if (!c) return;
    camera.position.copy(home.position);
    c.target.copy(home.target);
    c.update();
    camera.lookAt(home.target);
  }, [camera, home, resetKey]);

  useFrame(() => controls.current?.update());
  return null;
}

/** Soft image-based light from a neutral room, so metal and gloss have something to reflect. */
function RoomLight() {
  const { gl, scene } = useThree();
  useEffect(() => {
    const pmrem = new THREE.PMREMGenerator(gl);
    const env = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = env;
    scene.environmentIntensity = 0.55;
    return () => {
      scene.environment = null;
      env.dispose();
      pmrem.dispose();
    };
  }, [gl, scene]);
  return null;
}

/** Only one light casts shadows, the first directional (or spot) one: more costs frames for little. */
function firstShadow(scene: Scene): number {
  const lights = scene.lights.length ? scene.lights : DEFAULT_LIGHTS;
  return lights.findIndex((l) => l.kind === "directional" || l.kind === "spot");
}

function LightView({ light, reach, first }: { light: Light; reach: number; first: boolean }) {
  const position = light.position;
  const r = reach * 1.5;
  switch (light.kind) {
    case "ambient":
      return <ambientLight color={light.color} intensity={light.intensity} />;
    case "directional":
      return (
        <directionalLight
          color={light.color}
          intensity={light.intensity}
          position={position}
          castShadow={first}
          shadow-mapSize={[2048, 2048]}
          shadow-bias={-0.0004}
          shadow-normalBias={0.02}
          shadow-camera-left={-r}
          shadow-camera-right={r}
          shadow-camera-top={r}
          shadow-camera-bottom={-r}
          shadow-camera-far={Math.max(50, reach * 10)}
        />
      );
    case "point":
      return (
        <pointLight
          color={light.color}
          intensity={light.intensity}
          position={position}
          distance={light.distance}
          decay={light.decay}
        />
      );
    case "spot":
      return (
        <spotLight
          color={light.color}
          intensity={light.intensity}
          position={position}
          distance={light.distance}
          decay={light.decay}
          angle={light.angle}
          penumbra={light.penumbra}
          castShadow={first}
        />
      );
  }
}

function NodeView({
  node,
  reduced,
  onAssetError,
}: {
  node: Node;
  reduced: boolean;
  onAssetError?: (id: string, message: string) => void;
}) {
  const group = useRef<THREE.Group>(null);
  const { position, rotation, spin, orbit, bob } = node;

  useFrame(({ clock }) => {
    const g = group.current;
    if (!g) return;
    const t = clock.elapsedTime;
    let x = position[0];
    let y = position[1];
    let z = position[2];
    if (orbit && !reduced) {
      const a = t * orbit.speed + orbit.phase;
      const ox = Math.cos(a) * orbit.radius;
      const oz = Math.sin(a) * orbit.radius;
      if (orbit.axis === "y") {
        x += ox;
        z += oz;
      } else if (orbit.axis === "x") {
        y += ox;
        z += oz;
      } else {
        x += ox;
        y += oz;
      }
    }
    if (bob && !reduced) y += Math.sin(t * bob.speed) * bob.amplitude;
    g.position.set(x, y, z);

    if (spin) {
      if (reduced) {
        g.rotation.set(rotation[0], rotation[1], rotation[2]);
      } else {
        g.rotation.set(rotation[0] + t * spin[0], rotation[1] + t * spin[1], rotation[2] + t * spin[2]);
      }
    }
  });

  return (
    <group ref={group} position={position} rotation={rotation} scale={node.scale}>
      {node.geometry && <GeometryView geometry={node.geometry} material={node.material} />}
      {node.asset && <AssetView id={node.asset} onError={onAssetError} />}
      {node.children.map((child, i) => (
        <NodeView key={i} node={child} reduced={reduced} onAssetError={onAssetError} />
      ))}
    </group>
  );
}

function MaterialView({ material, flat }: { material: Material; flat?: boolean }) {
  return (
    <meshStandardMaterial
      color={material.color}
      metalness={material.metalness}
      roughness={material.roughness}
      transparent={material.opacity < 1}
      opacity={material.opacity}
      emissive={material.emissive}
      emissiveIntensity={material.emissiveIntensity}
      wireframe={material.wireframe}
      flatShading={flat || material.flatShading}
      side={THREE.DoubleSide}
    />
  );
}

function GeometryView({ geometry, material }: { geometry: Geometry; material: Material }) {
  switch (geometry.kind) {
    case "box":
      return (
        <mesh castShadow receiveShadow>
          <boxGeometry args={geometry.size} />
          <MaterialView material={material} />
        </mesh>
      );
    case "sphere":
      return (
        <mesh castShadow receiveShadow>
          <sphereGeometry args={[geometry.radius, geometry.segments, Math.max(8, Math.round(geometry.segments / 2))]} />
          <MaterialView material={material} />
        </mesh>
      );
    case "cylinder":
      return (
        <mesh castShadow receiveShadow>
          <cylinderGeometry
            args={[geometry.radiusTop, geometry.radiusBottom, geometry.height, geometry.segments]}
          />
          <MaterialView material={material} />
        </mesh>
      );
    case "cone":
      return (
        <mesh castShadow receiveShadow>
          <coneGeometry args={[geometry.radius, geometry.height, geometry.segments]} />
          <MaterialView material={material} />
        </mesh>
      );
    case "torus":
      return (
        <mesh castShadow receiveShadow>
          <torusGeometry args={[geometry.radius, geometry.tube, geometry.segments[1], geometry.segments[0]]} />
          <MaterialView material={material} />
        </mesh>
      );
    case "plane":
      return (
        <mesh castShadow receiveShadow>
          <planeGeometry args={geometry.size} />
          <MaterialView material={material} />
        </mesh>
      );
    case "capsule":
      return (
        <mesh castShadow receiveShadow>
          <capsuleGeometry args={[geometry.radius, geometry.length, 8, 16]} />
          <MaterialView material={material} />
        </mesh>
      );
    case "icosahedron":
      return (
        <mesh castShadow receiveShadow>
          <icosahedronGeometry args={[geometry.radius, geometry.detail]} />
          <MaterialView material={material} />
        </mesh>
      );
    case "polyhedron":
      return <PolyhedronView geometry={geometry} material={material} />;
    case "points":
      return <PointsView geometry={geometry} material={material} />;
    case "lines":
      return <LinesView geometry={geometry} material={material} />;
  }
}

function PolyhedronView({
  geometry: g,
  material,
}: {
  geometry: Extract<Geometry, { kind: "polyhedron" }>;
  material: Material;
}) {
  const geo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    if (g.flat) {
      const positions = new Float32Array(g.faces.length * 9);
      let k = 0;
      for (const face of g.faces) {
        for (const idx of face) {
          const v = g.vertices[idx];
          positions[k++] = v[0];
          positions[k++] = v[1];
          positions[k++] = v[2];
        }
      }
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    } else {
      const positions = new Float32Array(g.vertices.length * 3);
      g.vertices.forEach((v, i) => {
        positions[i * 3] = v[0];
        positions[i * 3 + 1] = v[1];
        positions[i * 3 + 2] = v[2];
      });
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setIndex(g.faces.flat());
    }
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
    return geo;
  }, [g]);

  useEffect(() => () => geo.dispose(), [geo]);
  return (
    <mesh geometry={geo} castShadow receiveShadow>
      <MaterialView material={material} flat={g.flat} />
    </mesh>
  );
}

function PointsView({
  geometry: g,
  material,
}: {
  geometry: Extract<Geometry, { kind: "points" }>;
  material: Material;
}) {
  const geo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(g.positions.length * 3);
    g.positions.forEach((p, i) => {
      positions[i * 3] = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];
    });
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (g.colors) {
      const colors = new Float32Array(g.colors.length * 3);
      const c = new THREE.Color();
      g.colors.forEach((hex, i) => {
        c.set(hex);
        colors[i * 3] = c.r;
        colors[i * 3 + 1] = c.g;
        colors[i * 3 + 2] = c.b;
      });
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    }
    geo.computeBoundingSphere();
    return geo;
  }, [g]);

  useEffect(() => () => geo.dispose(), [geo]);
  return (
    <points geometry={geo}>
      <pointsMaterial
        size={g.size}
        sizeAttenuation
        vertexColors={!!g.colors}
        color={g.colors ? undefined : material.color}
        transparent={material.opacity < 1}
        opacity={material.opacity}
      />
    </points>
  );
}

function LinesView({
  geometry: g,
  material,
}: {
  geometry: Extract<Geometry, { kind: "lines" }>;
  material: Material;
}) {
  const geo = useMemo(() => {
    const points = g.closed ? [...g.positions, g.positions[0]] : g.positions;
    const positions: number[] = [];
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1];
      const b = points[i];
      positions.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    return geo;
  }, [g]);

  useEffect(() => () => geo.dispose(), [geo]);
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial
        color={g.color}
        transparent={material.opacity < 1}
        opacity={material.opacity}
      />
    </lineSegments>
  );
}

function AssetView({ id, onError }: { id: string; onError?: (id: string, message: string) => void }) {
  const [object, setObject] = useState<THREE.Group | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    setObject(null);
    setFailed(false);
    loadAsset(id).then(
      (source) => {
        if (!alive) return;
        const copy = source.clone(true);
        copy.traverse((o) => (o.castShadow = o.receiveShadow = true));
        setObject(copy);
      },
      (e) => {
        if (!alive) return;
        setFailed(true);
        onError?.(id, (e as Error).message);
      },
    );
    return () => {
      alive = false;
    };
  }, [id, onError]);

  if (object) return <primitive object={object} dispose={null} />;
  if (failed) {
    return (
      <mesh>
        <boxGeometry args={[0.5, 0.5, 0.5]} />
        <meshBasicMaterial color="#ff5c5c" wireframe />
      </mesh>
    );
  }
  return null;
}

function useReducedMotion() {
  const [reduced, setReduced] = useState(
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return reduced;
}

class CanvasBoundary extends Component<{ children: ReactNode }, { error?: string }> {
  state: { error?: string } = {};
  static getDerivedStateFromError(error: Error) {
    return { error: error.message };
  }
  render() {
    if (this.state.error) {
      return <p className="scene-error">Não foi possível abrir o WebGL: {this.state.error}</p>;
    }
    return this.props.children;
  }
}
