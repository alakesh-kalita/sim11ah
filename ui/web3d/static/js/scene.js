// Renderer, camera, lighting, ground plane and the scene-graph groups
// every other module attaches its meshes to. Blocky-game look: flat sky,
// no atmospheric scattering, no bloom/PBR gloss -- one hard sun over
// nearest-filtered pixel-art blocks.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SKY_COLOR, tiledClone } from './core.js';
import { TEX } from './textures.js';

const canvas = document.getElementById('app-canvas');
export const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.BasicShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;

export const scene = new THREE.Scene();
scene.background = new THREE.Color(SKY_COLOR);
// FAR raised alongside every other AP-range-derived scale constant here
// (controls.maxDistance, the shadow frustum below) once topology_canvas.py's
// _ENV_PATH_LOSS_EXP gave each environment a realistic outdoor link budget
// (up to ~2.97km for Paddy Field) instead of one shared ~1.22km figure --
// the old 3200 far plane started fogging out real, in-range nodes well
// before the edge of a large scatter.
const DEFAULT_FOG_NEAR = 700, DEFAULT_FOG_FAR = 6000;
scene.fog = new THREE.Fog(SKY_COLOR, DEFAULT_FOG_NEAR, DEFAULT_FOG_FAR);

// Military Zone's real extent (802.11ah's long range routinely pushes it
// past 1km) can put the actual playable area, mountains and all, well
// inside the default 700-3200 fog band -- most of the visible battlefield
// reads as a hazy "smog" instead of clear ground. Lets a caller push the
// fog band out to fit the real scene extent; falls back to the default
// band for every other (much more compact) environment.
export function setFogRange(near, far) {
  scene.fog.near = near ?? DEFAULT_FOG_NEAR;
  scene.fog.far = far ?? DEFAULT_FOG_FAR;
}

export const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 1, 8000);
camera.position.set(320, 260, 420);

export const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 20;
// Was 3200 -- too tight to zoom out far enough to frame a large Paddy
// Field/Smart City scatter now that AP range can nominally reach ~3km (see
// DEFAULT_FOG_FAR above); frameCameraOnNodes() can otherwise request a
// distance that immediately gets clamped back in on the first
// controls.update(), undoing its own auto-frame.
controls.maxDistance = 6500;
controls.target.set(0, 15, 0);

export function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener('resize', resize);

const hemi = new THREE.HemisphereLight(0xbfe3ff, 0x7c6b4e, 0.95);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 1.35);
sun.position.set(500, 700, 260);
sun.castShadow = true;
// Frustum widened from +-1400/far 3000 alongside every other AP-range-
// derived scale constant above (DEFAULT_FOG_FAR, controls.maxDistance) --
// a scene whose real nodes now nominally reach ~3km was already losing
// shadows past the old edge. mapSize doubled to 2048 to keep texel density
// roughly where it was at the old, smaller frustum (a fixed map size over
// a wider frustum alone would just blur every shadow more).
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -3600;
sun.shadow.camera.right = 3600;
sun.shadow.camera.top = 3600;
sun.shadow.camera.bottom = -3600;
sun.shadow.camera.near = 100;
sun.shadow.camera.far = 6500;
scene.add(sun, sun.target);

// Warm, hazy late-day battlefield mood for Military Zone -- a flat
// sky-blue sky/fog and neutral-white sun read as a calm midday park, not
// a dusty combat zone. Background and fog colour MUST move together
// (they already were the same SKY_COLOR): change one without the other
// and distant ground fades into a mismatched colour, a hard visible band
// where fog meets sky. Resets to the defaults for every other
// environment, which never call this.
const DEFAULT_BG_COLOR = new THREE.Color(SKY_COLOR);
const MIL_BG_COLOR = new THREE.Color(0xcdc2a0);
const DEFAULT_SUN_COLOR = 0xffffff, DEFAULT_SUN_INTENSITY = 1.35;
const MIL_SUN_COLOR = 0xffcf94, MIL_SUN_INTENSITY = 1.5;
const DEFAULT_HEMI_SKY = 0xbfe3ff, DEFAULT_HEMI_GROUND = 0x7c6b4e, DEFAULT_HEMI_INTENSITY = 0.95;
const MIL_HEMI_SKY = 0xd8cfa8, MIL_HEMI_GROUND = 0x6b5a3e, MIL_HEMI_INTENSITY = 0.85;
export function setBattleAtmosphere(active) {
  scene.background = active ? MIL_BG_COLOR : DEFAULT_BG_COLOR;
  scene.fog.color.set(active ? MIL_BG_COLOR : DEFAULT_BG_COLOR);
  sun.color.set(active ? MIL_SUN_COLOR : DEFAULT_SUN_COLOR);
  sun.intensity = active ? MIL_SUN_INTENSITY : DEFAULT_SUN_INTENSITY;
  hemi.color.set(active ? MIL_HEMI_SKY : DEFAULT_HEMI_SKY);
  hemi.groundColor.set(active ? MIL_HEMI_GROUND : DEFAULT_HEMI_GROUND);
  hemi.intensity = active ? MIL_HEMI_INTENSITY : DEFAULT_HEMI_INTENSITY;
}

// ---- ground -------------------------------------------------------------
// Large enough that Military Zone's mountain backdrop (placed relative to
// the base's real extent, which 802.11ah's long range can push past 1km)
// always sits ON textured ground rather than past its edge -- a mountain
// ring beyond GROUND_SIZE's boundary rendered as peaks floating over bare
// sky with a visible seam where the ground texture just stopped.
export const GROUND_SIZE = 12000;
const groundGeo = new THREE.PlaneGeometry(GROUND_SIZE, GROUND_SIZE);
const groundMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

const ENV_GROUND_ICON = {
  'Open Area': 'grass', 'Paddy Field': 'paddy',
  'Industrial Site': 'dirt_mining', 'Smart City': 'concrete',
  'Military Zone': 'grass',
};
// grass.png is a small, flat, highly-saturated tile -- fine at Open Area's
// residential scale, but Military Zone's field is thousands of units
// across and viewed from far enough that mipmapping collapses the whole
// ground into one perfectly flat, cartoonish saturated green ("looks
// odd"/fake). Tinting it down towards a duller, more natural field colour
// (multiplied against the texture, not replacing it) fixes that without
// touching any other environment's ground.
const ENV_GROUND_TINT = {
  'Military Zone': 0xa9c47a,
};
let lastEnv = null;
export function applyEnvironment(env) {
  if (env === lastEnv) return;
  lastEnv = env;
  const tex = TEX[ENV_GROUND_ICON[env]] ?? TEX.grass;
  const tile = 6; // metres per ground block -- chunky and visible, not a smooth field
  if (groundMat.map) groundMat.map.dispose();
  groundMat.map = tiledClone(tex, GROUND_SIZE / tile, GROUND_SIZE / tile);
  groundMat.color.set(ENV_GROUND_TINT[env] ?? 0xffffff);
  groundMat.needsUpdate = true;
}

// ---- scene groups -- every other module attaches its meshes to one of these
export const roadGroup = new THREE.Group(); scene.add(roadGroup);
export const buildingsGroup = new THREE.Group(); scene.add(buildingsGroup);
export const propsGroup = new THREE.Group(); scene.add(propsGroup);
export const nodesGroup = new THREE.Group(); scene.add(nodesGroup);
export const linksGroup = new THREE.Group(); scene.add(linksGroup);
export const packetsGroup = new THREE.Group(); scene.add(packetsGroup);
export const vehiclesGroup = new THREE.Group(); scene.add(vehiclesGroup);
export const smokeGroup = new THREE.Group(); scene.add(smokeGroup);
