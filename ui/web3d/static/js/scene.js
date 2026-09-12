// Renderer, camera, lighting, ground plane and the scene-graph groups
// every other module attaches its meshes to. Still a deliberately
// low-poly/blocky city layout (no geometry rewrite), but PBR-lit: every
// material in world.js/entities.js/smoke.js is MeshStandardMaterial
// (roughness/metalness per surface -- concrete and dirt stay matte, metal
// railings/panels/vehicle trim get real specular, glass/water get a glossy
// low-roughness response), filmic tone mapping instead of a flat linear
// output, and a vertical gradient sky (see paintSkyGradient below) instead
// of one flat background colour.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { SSAOPass } from 'three/addons/postprocessing/SSAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { SKY_COLOR, tiledClone, hashSeed, mulberry32 } from './core.js';
import { TEX } from './textures.js';

const canvas = document.getElementById('app-canvas');
// antialias:true smooths block silhouette edges (a geometry-edge fix, not
// a lighting/material change) -- previously off, and every screenshot of
// this scene showed visibly jagged block/roof edges as a result.
export const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
// PCFSoftShadowMap softens shadow EDGES -- BasicShadowMap's hard-edged,
// slightly aliased shadow boundary was the single most obviously
// "unpolished" thing in every screenshot of this scene; Minecraft's own
// default shadows are soft-edged too, so this isn't fighting the voxel
// geometry, just cleaning up an artifact of the cheapest shadow-map filter.
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
// ACESFilmic replaces the old flat/linear output -- it rolls off highlights
// (sun-lit metal/glass specular from the PBR materials below, the emissive
// fire/warning-light materials) instead of clipping them to solid white,
// which is most of what makes a scene read as "rendered", not "lit". Goes
// through OutputPass (composer, below), which is exactly what that pass
// exists for. Exposure nudged slightly above 1.0 to compensate for
// ACESFilmic's midtone compression against this scene's existing light
// intensities (sun/hemi below are unchanged from the pre-PBR pass).
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

export const scene = new THREE.Scene();

// ---- sky -----------------------------------------------------------------
// A flat single-colour background reads as a solid-colour backdrop, not
// sky -- real open-air haze always lightens toward the horizon (more
// atmosphere between the eye and the horizon than straight up). Cheap,
// low-risk stand-in for a full atmospheric-scattering dome: a vertical
// gradient (deep/saturated at the zenith, hazy/whitened at the horizon)
// painted onto a tiny canvas and used as scene.background. Regenerated
// (not just recoloured) whenever setBattleAtmosphere below swaps the
// active mood colour, same trigger as before.
let skyCanvas = null, skyCtx = null, skyTexture = null;
function ensureSkyCanvas() {
  if (skyTexture) return;
  skyCanvas = document.createElement('canvas');
  skyCanvas.width = 2;
  skyCanvas.height = 256;
  skyCtx = skyCanvas.getContext('2d');
  skyTexture = new THREE.CanvasTexture(skyCanvas);
  skyTexture.colorSpace = THREE.SRGBColorSpace;
}
// Returns the horizon colour actually painted, so callers can keep fog
// (which represents the same haze, just applied to distant geometry
// instead of the empty sky) matched to it rather than to the zenith tone.
function paintSkyGradient(baseHex) {
  ensureSkyCanvas();
  const base = new THREE.Color(baseHex);
  const zenith = base.clone().multiplyScalar(0.72);
  const horizon = base.clone().lerp(new THREE.Color(0xffffff), 0.55);
  const grad = skyCtx.createLinearGradient(0, 0, 0, skyCanvas.height);
  grad.addColorStop(0, `#${zenith.getHexString()}`);
  grad.addColorStop(1, `#${horizon.getHexString()}`);
  skyCtx.fillStyle = grad;
  skyCtx.fillRect(0, 0, skyCanvas.width, skyCanvas.height);
  skyTexture.needsUpdate = true;
  return horizon;
}
const initialHorizon = paintSkyGradient(SKY_COLOR);
scene.background = skyTexture;

// FAR raised alongside every other AP-range-derived scale constant here
// (controls.maxDistance, the shadow frustum below) once topology_canvas.py's
// _ENV_PATH_LOSS_EXP gave each environment a realistic outdoor link budget
// (up to ~2.97km for Paddy Field) instead of one shared ~1.22km figure --
// the old 3200 far plane started fogging out real, in-range nodes well
// before the edge of a large scatter.
const DEFAULT_FOG_NEAR = 700, DEFAULT_FOG_FAR = 6000;
scene.fog = new THREE.Fog(initialHorizon, DEFAULT_FOG_NEAR, DEFAULT_FOG_FAR);

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

// near=4, not 1 -- a standard (non-logarithmic) WebGL depth buffer's
// precision is dominated by the near:far ratio, and controls.minDistance
// (below) already keeps the camera's orbit at least 20 units from its
// target, so nothing the user can actually navigate to sits anywhere
// near a 1-unit near plane. Quadrupling it noticeably tightens that
// ratio (8000:1 -> 2000:1) with no practical clipping risk, which
// directly helps the flat, large, nearly-coplanar ground layers
// (road/sidewalk/centreline, building bases) that flicker worst at the
// larger camera distances this scene's 1100m+ cars_uavs city now needs.
export const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 4, 8000);
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

// Contact-shadow-only SSAO: small kernel radius so it darkens the seams
// where blocks actually touch (ground/building corners, prop clusters)
// rather than producing the big soft grey halos a large-radius AO pass
// casts around every object -- that broad-halo look reads as hazy/washed
// out and would fight the PBR materials' own specular response (see the
// top-of-file comment). Geometry/layout untouched; this only adds a
// multiplicative darkening term in the few pixels where geometry actually
// occludes itself.
export const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const ssaoPass = new SSAOPass(scene, camera, window.innerWidth, window.innerHeight);
ssaoPass.kernelRadius = 6;
ssaoPass.minDistance = 0.0005;
ssaoPass.maxDistance = 0.03;
composer.addPass(ssaoPass);
// Threshold high (0.87) so only the BRIGHTEST pixels bloom -- the sun
// disc, emissive warning/beacon lights, headlights, direct sun-glint off
// the PBR metal surfaces -- not the whole scene. Strength/radius kept
// modest (0.35/0.4) for a subtle glow, not a haze over everything;
// tune these two first if it reads as too much or too little.
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight), 0.35, 0.4, 0.87,
);
composer.addPass(bloomPass);
// Must be the last pass -- EffectComposer's intermediate render targets
// don't apply the renderer's own sRGB output conversion automatically;
// without this the composited frame comes out visibly washed out
// relative to a plain renderer.render() call using the same
// outputColorSpace setting above.
composer.addPass(new OutputPass());

export function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  ssaoPass.setSize(w, h);
  bloomPass.setSize(w, h);
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
  const horizon = paintSkyGradient(active ? MIL_BG_COLOR : DEFAULT_BG_COLOR);
  scene.background = skyTexture;
  scene.fog.color.set(horizon);
  sun.color.set(active ? MIL_SUN_COLOR : DEFAULT_SUN_COLOR);
  sun.intensity = active ? MIL_SUN_INTENSITY : DEFAULT_SUN_INTENSITY;
  hemi.color.set(active ? MIL_HEMI_SKY : DEFAULT_HEMI_SKY);
  hemi.groundColor.set(active ? MIL_HEMI_GROUND : DEFAULT_HEMI_GROUND);
  hemi.intensity = active ? MIL_HEMI_INTENSITY : DEFAULT_HEMI_INTENSITY;
  sunSpriteMat.color.copy(sun.color);
}

// ---- sun disc + clouds ---------------------------------------------------
// The gradient sky above reads as atmosphere but had nothing IN it -- a
// visible sun disc and a scattering of clouds are the two cheapest,
// highest-impact things a sky can have that a flat gradient alone can't.
function makeSunTexture() {
  const size = 128;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,250,1)');
  grad.addColorStop(0.22, 'rgba(255,244,214,0.95)');
  grad.addColorStop(0.55, 'rgba(255,230,180,0.28)');
  grad.addColorStop(1, 'rgba(255,230,180,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
// Additive + no depth-write: a soft glow that composites onto the sky
// behind it rather than an opaque disc that could clip through terrain.
// fog:false -- a real sun is effectively at infinite distance, so unlike
// everything else in the scene, atmospheric haze shouldn't dim it.
const sunSpriteMat = new THREE.SpriteMaterial({
  map: makeSunTexture(), color: DEFAULT_SUN_COLOR, transparent: true,
  depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
});
const sunSprite = new THREE.Sprite(sunSpriteMat);
sunSprite.scale.set(900, 900, 1);
scene.add(sunSprite);
const SUN_SPRITE_DIST = 5200;
const _sunDir = new THREE.Vector3();
// Camera-relative position, not a fixed world point -- a real sun stays
// in the same DIRECTION no matter where you stand, so as the camera
// roams a multi-km scene (free-fly/chase-cam can travel far from the
// origin) the disc must move with it rather than visibly drift the way
// a finite-distance prop would. Called every animate() frame (app.js),
// not just on environment change, since the camera moves every frame
// under those newer modes.
export function updateSunSprite() {
  _sunDir.copy(sun.position).normalize();
  sunSprite.position.copy(camera.position).addScaledVector(_sunDir, SUN_SPRITE_DIST);
}
updateSunSprite();

// A handful of scattered cloud clusters -- the same "a few overlapping
// low-poly icosahedra" technique world.js's addTree uses for canopies,
// just bigger, flatter and high up. Fixed world positions (unlike the
// sun disc above) -- these represent real objects at a real altitude
// over a real part of the map, not something at effectively infinite
// distance, so they DO get fogged at a distance like everything else
// (no fog:false here) and don't need to track the camera.
const cloudMat = new THREE.MeshStandardMaterial({ color: 0xfbfcff, roughness: 0.95, metalness: 0 });
const cloudLobeGeo = [1, 0.82, 0.66].map((r) => new THREE.IcosahedronGeometry(r, 1));
const cloudsGroup = new THREE.Group();
scene.add(cloudsGroup);
const _cloudLobeOffsets = [[0, 0, 0], [1.3, -0.15, 0.5], [-1.2, -0.1, -0.45], [0.4, 0.1, -0.9]];
function addCloud(cx, cy, cz, scale, rng) {
  const g = new THREE.Group();
  for (let i = 0; i < _cloudLobeOffsets.length; i++) {
    const [ox, oy, oz] = _cloudLobeOffsets[i];
    const lobe = new THREE.Mesh(cloudLobeGeo[i % cloudLobeGeo.length], cloudMat);
    lobe.position.set(ox, oy * 0.4, oz);
    const s = 0.8 + rng() * 0.5;
    lobe.scale.set(s * 1.6, s * 0.7, s * 1.6);
    g.add(lobe);
  }
  g.position.set(cx, cy, cz);
  g.scale.setScalar(scale);
  cloudsGroup.add(g);
}
// Seeded, not Math.random() -- reproducible across reloads like every
// other procedural layout here. Scattered in a ring from 400 to 3600
// units out (never right overhead at the origin, where the densest node
// cluster/camera framing usually is) so clouds read as background sky
// dressing instead of competing with the scene itself for attention.
{
  const cloudRng = mulberry32(hashSeed('sim11ah-clouds'));
  for (let i = 0; i < 18; i++) {
    const ang = cloudRng() * Math.PI * 2;
    const r = 400 + cloudRng() * 3200;
    const cy = 320 + cloudRng() * 140;
    addCloud(Math.cos(ang) * r, cy, Math.sin(ang) * r, 60 + cloudRng() * 70, cloudRng);
  }
}

// ---- ground -------------------------------------------------------------
// Large enough that Military Zone's mountain backdrop (placed relative to
// the base's real extent, which 802.11ah's long range can push past 1km)
// always sits ON textured ground rather than past its edge -- a mountain
// ring beyond GROUND_SIZE's boundary rendered as peaks floating over bare
// sky with a visible seam where the ground texture just stopped.
export const GROUND_SIZE = 12000;
const groundGeo = new THREE.PlaneGeometry(GROUND_SIZE, GROUND_SIZE);
const groundMat = new THREE.MeshStandardMaterial({ roughness: 0.92, metalness: 0, color: 0xffffff });
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
