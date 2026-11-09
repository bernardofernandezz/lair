import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, type RefObject } from "react";
import * as THREE from "three";

const noise = /* glsl */ `
// Simplex 3D noise — Ashima Arts / Stefan Gustavson, MIT.
vec4 permute(vec4 x){ return mod(((x*34.0)+1.0)*x, 289.0); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }
float snoise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + 2.0 * C.xxx;
  vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
  i = mod(i, 289.0);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 1.0/7.0;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
`;

const vertexShader = /* glsl */ `
uniform float uTime;
uniform float uActivity;
varying vec3 vNormal;
varying vec3 vView;
varying float vNoise;
${noise}

float field(vec3 dir) {
  float f = 1.1 + uActivity * 1.4;
  return snoise(dir * f + uTime * 0.55) + 0.45 * snoise(dir * f * 2.4 - uTime * 0.8);
}

vec3 displaced(vec3 dir) {
  return dir * (1.0 + field(dir) * (0.06 + uActivity * 0.14));
}

void main() {
  vec3 n = normalize(position);
  vec3 t = normalize(abs(n.y) < 0.99 ? cross(n, vec3(0.0, 1.0, 0.0)) : cross(n, vec3(1.0, 0.0, 0.0)));
  vec3 b = cross(n, t);
  float e = 0.01;
  vec3 p = displaced(n);
  vec3 pt = displaced(normalize(n + t * e));
  vec3 pb = displaced(normalize(n + b * e));
  vNormal = normalize(normalMatrix * cross(pt - p, pb - p));
  vNoise = field(n);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vView = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}
`;

// Thin-film interference look: the film colour shifts with viewing angle, like oil on obsidian.
const fragmentShader = /* glsl */ `
uniform float uActivity;
uniform float uThinking;
uniform float uDark;
varying vec3 vNormal;
varying vec3 vView;
varying float vNoise;

vec3 film(float t) {
  const vec3 teal = vec3(0.25, 0.72, 0.69);
  const vec3 violet = vec3(0.48, 0.36, 0.84);
  const vec3 gold = vec3(0.89, 0.72, 0.34);
  float s = fract(t) * 3.0;
  return s < 1.0 ? mix(teal, violet, s) : s < 2.0 ? mix(violet, gold, s - 1.0) : mix(gold, teal, s - 2.0);
}

void main() {
  vec3 n = normalize(vNormal);
  vec3 v = normalize(vView);
  float facing = max(dot(n, v), 0.0);
  float rim = pow(1.0 - facing, 2.0);
  vec3 base = mix(vec3(0.075, 0.068, 0.095), vec3(0.11, 0.10, 0.14), uDark);
  vec3 sheen = film(rim * 0.9 + vNoise * 0.25 + 0.15 + uThinking * 0.22) * rim * (0.8 + uActivity * 0.7);
  vec3 l = normalize(vec3(-0.45, 0.8, 0.55));
  float spec = pow(max(dot(reflect(-l, n), v), 0.0), 48.0);
  float soft = pow(max(dot(reflect(-l, n), v), 0.0), 6.0) * 0.08;
  gl_FragColor = vec4(base + sheen + vec3(spec * 0.7 + soft), 1.0);
}
`;

/** Written by the chat as tokens stream: `pulse` is bumped per token and decays here. */
export type OrbSignal = { pulse: number; thinking: boolean };

type Props = {
  signal: RefObject<OrbSignal>;
  docked: boolean;
};

const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const darkScheme = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

function Pearl({ signal, docked }: Props) {
  const mesh = useRef<THREE.Mesh>(null!);
  const { gl, viewport } = useThree();
  const uniforms = useMemo(
    () => ({ uTime: { value: 0 }, uActivity: { value: 0 }, uThinking: { value: 0 }, uDark: { value: darkScheme() ? 1 : 0 } }),
    [],
  );
  const motion = useRef({ still: reducedMotion(), dragging: false, lastX: 0, lastY: 0, vx: 0, vy: 0, rx: 0, ry: 0, tiltX: 0, tiltY: 0, px: 0, py: 0 });

  useEffect(() => {
    const m = motion.current;
    const el = gl.domElement;
    const down = (e: PointerEvent) => {
      m.dragging = true;
      m.lastX = e.clientX;
      m.lastY = e.clientY;
      el.setPointerCapture(e.pointerId);
      el.style.cursor = "grabbing";
    };
    const move = (e: PointerEvent) => {
      m.px = (e.clientX / window.innerWidth) * 2 - 1;
      m.py = (e.clientY / window.innerHeight) * 2 - 1;
      if (!m.dragging) return;
      m.vy = (e.clientX - m.lastX) * 0.006;
      m.vx = (e.clientY - m.lastY) * 0.006;
      m.ry += m.vy;
      m.rx += m.vx;
      m.lastX = e.clientX;
      m.lastY = e.clientY;
    };
    const up = () => {
      m.dragging = false;
      el.style.cursor = "";
    };
    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const schemeQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const onMotion = () => (m.still = motionQuery.matches);
    const onScheme = () => (uniforms.uDark.value = schemeQuery.matches ? 1 : 0);
    el.addEventListener("pointerdown", down);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    motionQuery.addEventListener("change", onMotion);
    schemeQuery.addEventListener("change", onScheme);
    return () => {
      el.removeEventListener("pointerdown", down);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      motionQuery.removeEventListener("change", onMotion);
      schemeQuery.removeEventListener("change", onScheme);
    };
  }, [gl, uniforms]);

  useFrame((_, rawDt) => {
    const dt = Math.min(rawDt, 0.05);
    const m = motion.current;
    const ease = (k: number) => 1 - Math.exp(-dt * k);

    const sig = signal.current;
    sig.pulse *= Math.exp(-dt * 2.2);
    uniforms.uActivity.value += (Math.min(sig.pulse, 1) - uniforms.uActivity.value) * ease(6);
    uniforms.uThinking.value += ((sig.thinking ? 1 : 0) - uniforms.uThinking.value) * ease(2);
    const speed = 0.22 + uniforms.uActivity.value * (1.6 - uniforms.uThinking.value * 1.1);
    if (!m.still) uniforms.uTime.value += dt * speed;

    if (!m.dragging) {
      m.vx *= Math.exp(-dt * 2.5);
      m.vy *= Math.exp(-dt * 2.5);
      m.rx += m.vx;
      m.ry += m.vy + (m.still ? 0 : dt * 0.07);
    }
    m.tiltX += (m.py * 0.22 - m.tiltX) * ease(3);
    m.tiltY += (m.px * 0.3 - m.tiltY) * ease(3);

    const o = mesh.current;
    o.rotation.set(m.rx + m.tiltX, m.ry + m.tiltY, 0);

    const fit = Math.min(1, viewport.width / 2.2);
    const dockedScale = Math.min(0.16, viewport.width * 0.08);
    const scale = docked ? dockedScale : 0.62 * fit;
    const y = docked ? viewport.height / 2 - dockedScale * 2 : 0.35;
    const k = m.still ? 1 : ease(3.2);
    o.position.y += (y - o.position.y) * k;
    o.scale.setScalar(o.scale.x + (scale - o.scale.x) * k);
  });

  return (
    <mesh ref={mesh} scale={0.001}>
      <icosahedronGeometry args={[1, 64]} />
      <shaderMaterial vertexShader={vertexShader} fragmentShader={fragmentShader} uniforms={uniforms} />
    </mesh>
  );
}

export function Orb(props: Props) {
  return (
    <Canvas className="orb" camera={{ position: [0, 0, 4.2], fov: 40 }} dpr={[1, 2]} gl={{ antialias: true, alpha: true }} aria-hidden>
      <Pearl {...props} />
    </Canvas>
  );
}
