// Static/environment geometry built from server state: buildings, roads,
// and decorative props (trees, mining stockpiles, floodlights, city
// street furniture). Anything that's "part of the map" rather than a
// live simulated entity lives here.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clone as skeletonClone } from 'three/addons/utils/SkeletonUtils.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { hashSeed, mulberry32, tiledClone, toScene } from './core.js';
import { TEX, pickWallVariant } from './textures.js';
import {
  buildingsGroup, propsGroup, roadGroup, vehiclesGroup, smokeGroup, setFogRange, setBattleAtmosphere,
} from './scene.js';
import { clearSmokestacks, registerSmokestack, clearDustEmitters, registerDustEmitter } from './smoke.js';

// Real modelled+textured+animated low-poly soldier (MIT-licensed, vendored
// from three.js's own official example assets -- see ui/web3d/static/models/)
// used for Military Zone's patrol infantry instead of a hand-built
// capsule+sphere figure. Loaded once, eagerly, at module load; clones
// share the source geometry/skeleton (see SkeletonUtils) but each gets
// its own AnimationMixer so every soldier can be at a different point in
// the walk cycle.
let soldierGLTF = null;
let soldierLoadPromise = null;
function loadSoldierModel() {
  if (soldierLoadPromise) return soldierLoadPromise;
  soldierLoadPromise = new Promise((resolve) => {
    new GLTFLoader().load('/models/Soldier.glb', (gltf) => {
      soldierGLTF = gltf;
      resolve(gltf);
    }, undefined, () => resolve(null));
  });
  return soldierLoadPromise;
}
loadSoldierModel();

// Module-level materials get constructed when this module is first
// imported, which happens synchronously before app.js's main() awaits
// loadAllTextures() -- so `new THREE.MeshStandardMaterial({ map: TEX.foo })`
// at top level bakes in `map: undefined` forever (TEX.foo isn't populated
// yet). lazyMat() defers the map assignment: it hands back a plain
// material immediately and records it, then applyLoadedTextures() (called
// once textures actually finish loading) fills in every recorded map.
const deferredTexMats = [];
function lazyMat(key, extra) {
  const m = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0.0, ...extra });
  deferredTexMats.push([m, key]);
  return m;
}
export function applyLoadedTextures() {
  for (const [m, key] of deferredTexMats) {
    m.map = TEX[key];
    m.needsUpdate = true;
  }
}

// ---- building rooftop registry -- every box building records its own
// axis-aligned scene-space footprint + roof height here as a byproduct of
// being built, so entities.js can ask "is (x,z) standing on a roof, and
// how high" to place devices realistically without world.js needing to
// know anything about nodes. Rebuilt from scratch each rebuildBuildings()
// call alongside the meshes themselves.
let buildingFootprints = [];
function registerFootprint(x, z, w, d, topY) {
  buildingFootprints.push({ x, z, hw: w / 2, hd: d / 2, topY });
}
export function roofHeightAt(x, z) {
  let best = null;
  for (const f of buildingFootprints) {
    if (Math.abs(x - f.x) <= f.hw && Math.abs(z - f.z) <= f.hd) {
      if (best === null || f.topY > best) best = f.topY;
    }
  }
  return best;
}

// ---- decorative props: real-tree-silhouette trees (tapered trunk +
// overlapping low-poly canopy lobes, not a log column + leaf cube),
// mining stockpiles, floodlight towers ------------------------------------
const trunkMat = lazyMat('log');
const leafMat = lazyMat('leaves');
// Tapered hexagonal prism (6 radial segments -- faceted, not round; stays
// inside this scene's low-poly look) instead of a uniform box: wider at
// the base, narrower at the top, the single biggest thing that used to
// read as "a stick" rather than a trunk.
const trunkGeo = new THREE.CylinderGeometry(0.42, 0.62, 6, 6);
// Three overlapping icosahedra (20-tri faceted spheres -- the standard
// low-poly-game foliage shape) instead of 6 axis-aligned leaf cubes: a
// cube cluster reads as a stacked box no matter how it's arranged, an
// icosahedron already reads as a rounded canopy lobe on its own, and
// three overlapping, differently-sized ones at offset positions give an
// asymmetric, organic silhouette instead of a symmetric block. Also one
// FEWER mesh per tree than the old 6-cube cluster (3 lobes vs 6 cubes),
// so this is a rendering-cost win too, not just a look one.
const canopyLobeGeo = [3.3, 2.5, 2.0].map((r) => new THREE.IcosahedronGeometry(r, 0));
const _CANOPY_LOBE_OFFSETS = [[0, 0.35, 0], [1.05, -0.25, 0.55], [-0.95, -0.15, -0.65]];
// Boosts every addTree() call site at once -- trees sized to their raw
// call-site scale read as roughly the same size as a 9m/13m-tall STA
// node marker, which made a whole treeline look like a row of network
// icons instead of vegetation. Bumping the shared base geometry instead
// of touching each of the dozen call sites keeps their relative variety
// (near-camera vs background trees) while making all of them read as
// clearly larger than a node marker.
const _TREE_SCALE_BOOST = 1.45;
function addTree(x, z, scaleIn) {
  const scale = scaleIn * _TREE_SCALE_BOOST;
  const g = new THREE.Group();
  const trunk = new THREE.Mesh(trunkGeo, trunkMat);
  trunk.position.y = 3 * scale;
  trunk.scale.set(scale, scale, scale);
  trunk.castShadow = true;
  g.add(trunk);
  // Leaf clusters don't cast shadows -- 6 extra shadow-casters PER TREE
  // (on top of the trunk, which still does, for basic ground grounding)
  // added up fast at Smart City's filler density (hundreds of trees
  // across a large scattered scene), and a missing leaf-cluster shadow
  // is not a detail anyone notices at normal viewing distance, especially
  // under PCFSoftShadowMap's already-blurred edges. Real, broadly
  // applicable shadow-pass cost cut (every environment using addTree
  // benefits, not just one) for a visual cost nobody will see. Still
  // true with 3 lobes instead of 6 cubes.
  //
  // Per-tree jitter seeded from this tree's own (x, z) -- addTree's
  // call sites only ever pass a position/scale, no rng function, and
  // adding one would touch every one of them; deriving determinism from
  // position instead (core.js's hashSeed/mulberry32, the same PRNG this
  // file already uses elsewhere) keeps the signature unchanged while
  // still making neighbouring trees look like distinct individuals
  // instead of identical clones, and stays fully reproducible for a
  // given layout (same positions -> same jitter, every run).
  const jitter = mulberry32(hashSeed(`tree:${x.toFixed(2)}:${z.toFixed(2)}`));
  for (let i = 0; i < canopyLobeGeo.length; i++) {
    const [ox, oy, oz] = _CANOPY_LOBE_OFFSETS[i];
    const jx = (jitter() - 0.5) * 1.1, jz = (jitter() - 0.5) * 1.1;
    const lobe = new THREE.Mesh(canopyLobeGeo[i], leafMat);
    lobe.position.set((ox + jx) * scale, (6.6 + oy) * scale, (oz + jz) * scale);
    const s = scale * (0.85 + jitter() * 0.3);
    lobe.scale.set(s, s * (0.8 + jitter() * 0.25), s);
    lobe.rotation.y = jitter() * Math.PI * 2;
    g.add(lobe);
  }
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

const stockpileMat = lazyMat('gravel');
function addStockpile(x, z, radius, rng) {
  const blocks = Math.max(6, Math.round(radius / 3));
  const g = new THREE.Group();
  for (let i = 0; i < blocks; i++) {
    const ang = rng() * Math.PI * 2, rr = rng() * radius;
    const s = 3 + rng() * 3;
    const b = new THREE.Mesh(new THREE.BoxGeometry(s, s * 0.7, s), stockpileMat);
    b.position.set(Math.cos(ang) * rr, s * 0.32, Math.sin(ang) * rr);
    b.rotation.y = rng() * Math.PI;
    b.castShadow = true; b.receiveShadow = true;
    g.add(b);
  }
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

const floodMastMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.7, color: 0x2e3238 });
const floodLampMat = new THREE.MeshBasicMaterial({ color: 0xfff2c2 });
function addFloodlight(x, z, rng) {
  const g = new THREE.Group();
  const mastH = 16 + rng() * 4;
  const mast = new THREE.Mesh(new THREE.BoxGeometry(1.2, mastH, 1.2), floodMastMat);
  mast.position.y = mastH / 2;
  mast.castShadow = true;
  g.add(mast);
  for (let i = 0; i < 4; i++) {
    const lamp = new THREE.Mesh(new THREE.BoxGeometry(1.8, 1.0, 0.5), floodLampMat);
    const ang = (i / 4) * Math.PI * 2;
    lamp.position.set(Math.cos(ang) * 1.7, mastH + 0.6, Math.sin(ang) * 1.7);
    lamp.rotation.y = ang;
    g.add(lamp);
  }
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

// Shipping-container stack -- a small colourful cluster of boxes, the
// single most recognisable "industrial yard is full of stuff" prop.
const containerMats = [0xc0392b, 0x2a6f97, 0x3a7d5c, 0xe0a13a, 0x8a5fa8]
  .map(color => new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.45, color }));
const containerGeo = new THREE.BoxGeometry(6, 2.6, 2.4);
function addContainerStack(x, z, rng) {
  const g = new THREE.Group();
  const cols = 2 + Math.floor(rng() * 3);
  const rows = rng() < 0.45 ? 2 : 1;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (r > 0 && rng() < 0.3) continue; // gaps in the top row
      const mat = containerMats[(rng() * containerMats.length) | 0];
      const box = new THREE.Mesh(containerGeo, mat);
      box.position.set(c * 6.3 - ((cols - 1) * 6.3) / 2, r * 2.7 + 1.3, 0);
      box.castShadow = true; box.receiveShadow = true;
      g.add(box);
    }
  }
  g.rotation.y = rng() * Math.PI * 2;
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

// Oil-drum cluster -- quick scattered detail between the bigger props.
const barrelMats = [
  new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.5, color: 0x3a4a52 }),
  new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.5, color: 0x8a4a2a }),
  new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.5, color: 0x4a5c2a }),
];
function addBarrelCluster(x, z, rng) {
  const g = new THREE.Group();
  const n = 3 + Math.floor(rng() * 5);
  for (let i = 0; i < n; i++) {
    const mat = barrelMats[(rng() * barrelMats.length) | 0];
    const b = new THREE.Mesh(new THREE.CylinderGeometry(0.9, 0.9, 1.6, 8), mat);
    const ang = rng() * Math.PI * 2, r = rng() * 3;
    b.position.set(Math.cos(ang) * r, 0.8, Math.sin(ang) * r);
    b.castShadow = true; b.receiveShadow = true;
    g.add(b);
  }
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

// Parked heavy truck -- cab + flatbed + wheels, the "mining site" traffic
// that's always sitting around a real yard even when not moving.
const truckBodyMats = [0xc0392b, 0x2a6f97, 0xe0a13a, 0x5c6672].map(color => new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.5, color }));
const truckBedMat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.4, color: 0x4a4f55 });
const truckWheelMat = new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0.15, color: 0x15171b });
const truckCabGeo = new THREE.BoxGeometry(3, 2.6, 2.6);
const truckBedGeo = new THREE.BoxGeometry(6, 1.8, 2.7);
const truckWheelGeo = new THREE.BoxGeometry(0.9, 1.3, 1.3);
function buildTruckMesh(rng) {
  const g = new THREE.Group();
  const bodyMat = truckBodyMats[(rng() * truckBodyMats.length) | 0];
  const cab = new THREE.Mesh(truckCabGeo, bodyMat);
  cab.position.set(4, 1.5, 0);
  cab.castShadow = true;
  g.add(cab);
  const bed = new THREE.Mesh(truckBedGeo, truckBedMat);
  bed.position.set(-2, 1.2, 0);
  bed.castShadow = true;
  g.add(bed);
  for (const wx of [4.5, 1.5, -2, -4.5]) {
    for (const wz of [-1.5, 1.5]) {
      const wheel = new THREE.Mesh(truckWheelGeo, truckWheelMat);
      wheel.position.set(wx, 0.65, wz);
      g.add(wheel);
    }
  }
  return { group: g, bodyMat };
}

function addTruck(x, z, ry, rng) {
  const built = buildTruckMesh(rng);
  built.group.rotation.y = ry;
  built.group.position.set(x, 0, z);
  propsGroup.add(built.group);
}

// Worker's car -- low two-box sedan, visibly half the truck's bulk. A
// busy site is full of staff vehicles, not just plant machinery.
const carBodyMats = [0xd8dade, 0x9aa2ab, 0xb03a2e, 0x2e5fa3, 0x23272c, 0x74833f]
  .map(color => new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.5, color }));
const carGlassMat = new THREE.MeshStandardMaterial({ roughness: 0.12, metalness: 0.35, color: 0x9fc4d8 });
const carBodyGeo = new THREE.BoxGeometry(3.8, 0.9, 1.8);
const carCabinGeo = new THREE.BoxGeometry(1.9, 0.75, 1.6);
const carWheelGeo = new THREE.BoxGeometry(0.6, 0.75, 0.75);
function buildCarMesh(rng) {
  const g = new THREE.Group();
  const bodyMat = carBodyMats[(rng() * carBodyMats.length) | 0];
  const body = new THREE.Mesh(carBodyGeo, bodyMat);
  body.position.set(0, 0.85, 0);
  body.castShadow = true;
  g.add(body);
  const cabin = new THREE.Mesh(carCabinGeo, carGlassMat);
  cabin.position.set(-0.35, 1.62, 0);
  cabin.castShadow = true;
  g.add(cabin);
  for (const wx of [1.25, -1.25]) {
    for (const wz of [-0.95, 0.95]) {
      const wheel = new THREE.Mesh(carWheelGeo, truckWheelMat);
      wheel.position.set(wx, 0.4, wz);
      g.add(wheel);
    }
  }
  return { group: g, bodyMat };
}

function addCar(x, z, ry, rng) {
  const built = buildCarMesh(rng);
  built.group.rotation.y = ry;
  built.group.position.set(x, 0, z);
  propsGroup.add(built.group);
}

// Tractor-trailer -- long-nose cab hauling a container-coloured box. No
// articulated hitch: a rigid cab+trailer pair with a small gap between
// them reads fine at this scale, and it's the heavy-haul silhouette a
// mining/logistics yard is actually full of.
const semiCabMats = [0xc0392b, 0x1f6f50, 0x2a6f97, 0xd9dbe0, 0xe0a13a]
  .map(color => new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.5, color }));
const semiHoodGeo = new THREE.BoxGeometry(2.2, 1.5, 2.3);
const semiCabGeo = new THREE.BoxGeometry(2.2, 3.1, 2.6);
const semiStackGeo = new THREE.BoxGeometry(0.35, 2.4, 0.35);
const semiTrailerGeo = new THREE.BoxGeometry(9.5, 3.0, 2.6);
function buildSemiMesh(rng) {
  const g = new THREE.Group();
  const bodyMat = semiCabMats[(rng() * semiCabMats.length) | 0];
  const hood = new THREE.Mesh(semiHoodGeo, bodyMat);
  hood.position.set(6.5, 1.35, 0);
  hood.castShadow = true;
  g.add(hood);
  const cab = new THREE.Mesh(semiCabGeo, bodyMat);
  cab.position.set(4.4, 2.15, 0);
  cab.castShadow = true;
  g.add(cab);
  const stack = new THREE.Mesh(semiStackGeo, truckWheelMat);
  stack.position.set(3.5, 3.2, 1.1);
  g.add(stack);
  const trailer = new THREE.Mesh(semiTrailerGeo, containerMats[(rng() * containerMats.length) | 0]);
  trailer.position.set(-2.6, 2.1, 0);
  trailer.castShadow = true;
  g.add(trailer);
  for (const wx of [6.3, 2.0, 0.9, -5.6, -6.7]) {
    for (const wz of [-1.5, 1.5]) {
      const wheel = new THREE.Mesh(truckWheelGeo, truckWheelMat);
      wheel.position.set(wx, 0.65, wz);
      g.add(wheel);
    }
  }
  return { group: g, bodyMat };
}

function addSemi(x, z, ry, rng) {
  const built = buildSemiMesh(rng);
  built.group.rotation.y = ry;
  built.group.position.set(x, 0, z);
  propsGroup.add(built.group);
}

// Marked-out staff parking -- two facing rows of cars with the odd empty
// stall; ordered rows say "shift is on site" in a way scatter can't.
function addParkingLot(x, z, ry, cols, rng) {
  const g = new THREE.Group();
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < cols; c++) {
      if (rng() < 0.2) continue;
      const built = buildCarMesh(rng);
      built.group.position.set((r - 0.5) * 9, 0, (c - (cols - 1) / 2) * 2.9);
      built.group.rotation.y = (r ? 0 : Math.PI) + (rng() - 0.5) * 0.08;
      g.add(built.group);
    }
  }
  g.rotation.y = ry;
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

function addSemiRow(x, z, ry, n, rng) {
  const g = new THREE.Group();
  for (let i = 0; i < n; i++) {
    const built = buildSemiMesh(rng);
    built.group.position.set((rng() - 0.5) * 2, 0, (i - (n - 1) / 2) * 5.4);
    built.group.rotation.y = (rng() - 0.5) * 0.06;
    g.add(built.group);
  }
  g.rotation.y = ry;
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

// ---- Smart City street furniture -- the sidewalk-scale clutter (lamps,
// benches, signals, shelters) that makes a downtown read as inhabited
// rather than a bare road ring between towers ------------------------------
const lampPoleMat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.7, color: 0x33383e });
const lampHeadMat = new THREE.MeshBasicMaterial({ color: 0xffe9b0 });
const lampPoleGeo = new THREE.BoxGeometry(0.5, 8.5, 0.5);
const lampArmGeo = new THREE.BoxGeometry(2.4, 0.4, 0.4);
const lampHeadGeo = new THREE.BoxGeometry(1.2, 0.45, 0.8);
function addStreetlamp(x, z, ry) {
  const g = new THREE.Group();
  const pole = new THREE.Mesh(lampPoleGeo, lampPoleMat);
  pole.position.y = 4.25;
  pole.castShadow = true;
  g.add(pole);
  const arm = new THREE.Mesh(lampArmGeo, lampPoleMat);
  arm.position.set(1.1, 8.3, 0);
  g.add(arm);
  const head = new THREE.Mesh(lampHeadGeo, lampHeadMat);
  head.position.set(2.1, 8.05, 0);
  g.add(head);
  g.rotation.y = ry;
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

const benchWoodMat = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0, color: 0x8a6a42 });
const benchLegMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.6, color: 0x2c2f33 });
const benchSeatGeo = new THREE.BoxGeometry(1.1, 0.3, 3);
const benchBackGeo = new THREE.BoxGeometry(0.25, 1.0, 3);
const benchLegGeo = new THREE.BoxGeometry(1.0, 0.9, 0.3);
function addBench(x, z, ry) {
  const g = new THREE.Group();
  const seat = new THREE.Mesh(benchSeatGeo, benchWoodMat);
  seat.position.y = 1.0;
  seat.castShadow = true;
  g.add(seat);
  const back = new THREE.Mesh(benchBackGeo, benchWoodMat);
  back.position.set(-0.5, 1.6, 0);
  g.add(back);
  for (const lz of [-1.2, 1.2]) {
    const leg = new THREE.Mesh(benchLegGeo, benchLegMat);
    leg.position.set(0, 0.45, lz);
    g.add(leg);
  }
  g.rotation.y = ry;
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

const binMat = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.25, color: 0x37503f });
const binGeo = new THREE.CylinderGeometry(0.65, 0.55, 1.4, 6);
function addTrashCan(x, z) {
  const bin = new THREE.Mesh(binGeo, binMat);
  bin.position.set(x, 0.7, z);
  bin.castShadow = true;
  propsGroup.add(bin);
}

const hydrantMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.35, color: 0xc23b2e });
const hydrantGeo = new THREE.BoxGeometry(0.7, 1.1, 0.7);
const hydrantCapGeo = new THREE.BoxGeometry(0.45, 0.35, 0.45);
function addHydrant(x, z) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(hydrantGeo, hydrantMat);
  body.position.y = 0.55;
  g.add(body);
  const cap = new THREE.Mesh(hydrantCapGeo, hydrantMat);
  cap.position.y = 1.25;
  g.add(cap);
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

const signalDarkMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.3, color: 0x23262b });
const signalPoleGeo = new THREE.BoxGeometry(0.45, 7, 0.45);
const signalHeadGeo = new THREE.BoxGeometry(0.8, 2.3, 0.9);
const signalLightGeo = new THREE.BoxGeometry(0.3, 0.5, 0.5);
const signalMats = [0xff4d43, 0xffc23d, 0x35d06a].map(c => new THREE.MeshBasicMaterial({ color: c }));
function addTrafficLight(x, z, ry) {
  const g = new THREE.Group();
  const pole = new THREE.Mesh(signalPoleGeo, signalDarkMat);
  pole.position.y = 3.5;
  pole.castShadow = true;
  g.add(pole);
  const head = new THREE.Mesh(signalHeadGeo, signalDarkMat);
  head.position.y = 7.9;
  g.add(head);
  signalMats.forEach((m, i) => {
    const lamp = new THREE.Mesh(signalLightGeo, m);
    lamp.position.set(0.5, 8.6 - i * 0.7, 0);
    g.add(lamp);
  });
  g.rotation.y = ry;
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

const shelterRoofMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.35, color: 0x2a6f97 });
const shelterPostGeo = new THREE.BoxGeometry(0.35, 3.1, 0.35);
const shelterRoofGeo = new THREE.BoxGeometry(2.6, 0.35, 5.6);
const shelterBackGeo = new THREE.BoxGeometry(0.25, 2.2, 5.6);
const shelterSeatGeo = new THREE.BoxGeometry(1.0, 0.35, 4.4);
function addBusShelter(x, z, ry) {
  const g = new THREE.Group();
  const roof = new THREE.Mesh(shelterRoofGeo, shelterRoofMat);
  roof.position.y = 3.2;
  roof.castShadow = true;
  g.add(roof);
  const back = new THREE.Mesh(shelterBackGeo, carGlassMat);
  back.position.set(-1.1, 1.45, 0);
  g.add(back);
  for (const pz of [-2.6, 2.6]) {
    const post = new THREE.Mesh(shelterPostGeo, benchLegMat);
    post.position.set(1.05, 1.55, pz);
    g.add(post);
  }
  const seat = new THREE.Mesh(shelterSeatGeo, benchWoodMat);
  seat.position.set(-0.55, 0.75, 0);
  g.add(seat);
  g.rotation.y = ry;
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

const kioskAwningMat = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0, color: 0xe8e4da });
const kioskBodyGeo = new THREE.BoxGeometry(3.2, 2.7, 2.6);
const kioskAwningGeo = new THREE.BoxGeometry(1.4, 0.25, 3.2);
function addKiosk(x, z, ry, rng) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(kioskBodyGeo, containerMats[(rng() * containerMats.length) | 0]);
  body.position.y = 1.35;
  body.castShadow = true;
  g.add(body);
  const awning = new THREE.Mesh(kioskAwningGeo, kioskAwningMat);
  awning.position.set(2.0, 2.45, 0);
  awning.rotation.z = -0.18;
  g.add(awning);
  const sign = new THREE.Mesh(lampHeadGeo, lampHeadMat);
  sign.position.set(0, 2.95, 0);
  g.add(sign);
  g.rotation.y = ry;
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

const fountainStoneMat = new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0, color: 0x9aa0a6 });
const fountainWaterMat = new THREE.MeshBasicMaterial({ color: 0x6cc8e8 });
const fountainBasinGeo = new THREE.CylinderGeometry(7, 7.6, 1.4, 8);
const fountainWaterGeo = new THREE.CylinderGeometry(5.9, 5.9, 0.5, 8);
const fountainPillarGeo = new THREE.CylinderGeometry(0.9, 1.2, 3.4, 6);
const fountainTopGeo = new THREE.CylinderGeometry(2.4, 2.8, 0.7, 6);
function addFountain(x, z) {
  const g = new THREE.Group();
  const basin = new THREE.Mesh(fountainBasinGeo, fountainStoneMat);
  basin.position.y = 0.7;
  basin.castShadow = true; basin.receiveShadow = true;
  g.add(basin);
  const water = new THREE.Mesh(fountainWaterGeo, fountainWaterMat);
  water.position.y = 1.15;
  g.add(water);
  const pillar = new THREE.Mesh(fountainPillarGeo, fountainStoneMat);
  pillar.position.y = 2.4;
  pillar.castShadow = true;
  g.add(pillar);
  const top = new THREE.Mesh(fountainTopGeo, fountainStoneMat);
  top.position.y = 4.3;
  g.add(top);
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

// Football/soccer pitch -- flat turf rect, painted boundary/centre lines
// and a centre circle, two low-poly goal frames at the short ends.
const pitchGrassMat = lazyMat('grass');
const pitchLineMat = new THREE.MeshBasicMaterial({ color: 0xf4f6f2 });
const goalPostMat = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.7, color: 0xe8ebee });
function addFootballField(x, z, ry, halfW, halfD) {
  const g = new THREE.Group();
  const turf = new THREE.Mesh(new THREE.BoxGeometry(halfW * 2, 0.15, halfD * 2), pitchGrassMat);
  turf.position.y = 0.08;
  turf.receiveShadow = true;
  g.add(turf);

  const lineY = 0.17, lw = 0.25;
  const addStrip = (w, d, lx, lz) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, 0.03, d), pitchLineMat);
    m.position.set(lx, lineY, lz);
    g.add(m);
  };
  addStrip(halfW * 2, lw, 0, halfD);
  addStrip(halfW * 2, lw, 0, -halfD);
  addStrip(lw, halfD * 2, halfW, 0);
  addStrip(lw, halfD * 2, -halfW, 0);
  addStrip(lw, halfD * 2, 0, 0);

  const cr = Math.min(halfW, halfD) * 0.32;
  const ring = new THREE.Mesh(new THREE.RingGeometry(cr, cr + lw, 20), pitchLineMat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = lineY;
  g.add(ring);

  const postGeo = new THREE.BoxGeometry(0.25, 2.4, 0.25);
  const barGeo = new THREE.BoxGeometry(0.25, 0.25, 7.3);
  for (const gx of [-halfW, halfW]) {
    const goal = new THREE.Group();
    for (const gz of [-3.65, 3.65]) {
      const post = new THREE.Mesh(postGeo, goalPostMat);
      post.position.set(0, 1.2, gz);
      goal.add(post);
    }
    const bar = new THREE.Mesh(barGeo, goalPostMat);
    bar.position.set(0, 2.4, 0);
    goal.add(bar);
    goal.position.set(gx, 0, 0);
    g.add(goal);
  }

  g.rotation.y = ry;
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

// Park pond -- low-segment cylinders read as an organic rounded shape at
// this scale (cheaper and better-looking here than a plain rectangle),
// stone rim, a short wooden dock over the water.
const pondWaterMat = new THREE.MeshBasicMaterial({ color: 0x4a8fc0 });
const pondRimMat = lazyMat('gravel');
const pondDockMat = lazyMat('planks');
function addPond(x, z, r) {
  const g = new THREE.Group();
  const segs = 10;
  const rim = new THREE.Mesh(new THREE.CylinderGeometry(r * 1.15, r * 1.25, 0.6, segs), pondRimMat);
  rim.position.y = 0.15;
  rim.receiveShadow = true;
  g.add(rim);
  const water = new THREE.Mesh(new THREE.CylinderGeometry(r, r * 1.02, 0.3, segs), pondWaterMat);
  water.position.y = 0.32;
  g.add(water);

  const dock = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.2, r * 0.9), pondDockMat);
  dock.position.set(r * 0.9, 0.4, 0);
  dock.castShadow = true;
  g.add(dock);
  for (const dz of [-r * 0.4, 0, r * 0.4]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.5, 0.2), pondDockMat);
    post.position.set(r * 0.55, 0.15, dz);
    g.add(post);
  }

  g.position.set(x, 0, z);
  propsGroup.add(g);
}

// Playground -- a sandpit with a blocky swing set and slide. Neither
// needs to be literal, just readable as "playground" at this scale.
const playSandMat = new THREE.MeshStandardMaterial({ roughness: 0.95, metalness: 0, color: 0xd8c48a });
const playFrameMat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.6, color: 0xc23b2e });
const playSlideMat = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.5, color: 0x2a6f97 });
function addPlayground(x, z, ry) {
  const g = new THREE.Group();
  const sand = new THREE.Mesh(new THREE.BoxGeometry(16, 0.1, 12), playSandMat);
  sand.position.y = 0.05;
  sand.receiveShadow = true;
  g.add(sand);

  const swingX = -4;
  for (const sx of [-1, 1]) {
    for (const dz of [-1.6, 1.6]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.18, 2.6, 0.18), playFrameMat);
      leg.position.set(swingX + sx * 1, 1.3, dz);
      leg.rotation.z = sx * 0.18;
      g.add(leg);
    }
  }
  const swingBar = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.18, 3.6), playFrameMat);
  swingBar.position.set(swingX, 2.55, 0);
  g.add(swingBar);
  for (const dz of [-0.9, 0.9]) {
    const seat = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.08, 0.35), playFrameMat);
    seat.position.set(swingX, 1.1, dz);
    g.add(seat);
  }

  const slideX = 4;
  const platform = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.15, 1.4), playFrameMat);
  platform.position.set(slideX, 2.0, -1.5);
  g.add(platform);
  const slide = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.12, 3.4), playSlideMat);
  slide.position.set(slideX, 1.1, 0.3);
  slide.rotation.x = -0.55;
  g.add(slide);
  for (const lx of [-0.6, 0.6]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.15, 2.0, 0.15), playFrameMat);
    leg.position.set(slideX + lx, 1.0, -1.5);
    g.add(leg);
  }

  g.rotation.y = ry;
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

// Six-post gazebo -- reuses the bench wood/leg materials and the existing
// shingle-roof texture rather than inventing new ones.
const gazeboRoofMat = lazyMat('roof_shingle');
const gazeboDeckMat = lazyMat('planks');
function addGazebo(x, z) {
  const g = new THREE.Group();
  const roof = new THREE.Mesh(new THREE.ConeGeometry(4.2, 2.6, 6), gazeboRoofMat);
  roof.position.y = 3.6;
  roof.castShadow = true;
  g.add(roof);
  const deck = new THREE.Mesh(new THREE.CylinderGeometry(3.6, 3.6, 0.3, 6), gazeboDeckMat);
  deck.position.y = 0.25;
  deck.receiveShadow = true;
  g.add(deck);
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.3, 2.3, 0.3), benchLegMat);
    post.position.set(Math.cos(a) * 3.2, 1.5, Math.sin(a) * 3.2);
    post.castShadow = true;
    g.add(post);
  }
  const seat = new THREE.Mesh(new THREE.CylinderGeometry(3.0, 3.0, 0.35, 6), benchWoodMat);
  seat.position.y = 0.85;
  g.add(seat);
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

// Flowerbed with a low hedge border -- garden filler between the bigger
// park zones.
const gardenBedMat = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0, color: 0x5c4630 });
const gardenHedgeMat = lazyMat('leaves');
const gardenFlowerMats = [0xc0392b, 0xe0a72e, 0x8b5cf6, 0xffffff].map(c => new THREE.MeshBasicMaterial({ color: c }));
function addGardenBed(x, z, ry, rng) {
  const g = new THREE.Group();
  const bed = new THREE.Mesh(new THREE.BoxGeometry(6, 0.35, 4), gardenBedMat);
  bed.position.y = 0.17;
  bed.receiveShadow = true;
  g.add(bed);
  for (let i = 0; i < 10; i++) {
    const flower = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.3, 0.3), gardenFlowerMats[i % gardenFlowerMats.length]);
    flower.position.set((rng() - 0.5) * 5, 0.5, (rng() - 0.5) * 3);
    g.add(flower);
  }
  for (const hz of [-2.1, 2.1]) {
    const hedge = new THREE.Mesh(new THREE.BoxGeometry(6.4, 0.9, 0.5), gardenHedgeMat);
    hedge.position.set(0, 0.6, hz);
    g.add(hedge);
  }
  g.rotation.y = ry;
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

// Flat lawn patch -- fills a vacant downtown lot with grass instead of
// leaving it bare pavement; trees/benches/gardens get scattered on top
// of it by the caller (addCityFillerBuildings) rather than baked in here,
// same "small atomic builder, orchestration at the call site" pattern as
// the rest of the file.
const grassPatchMat = lazyMat('grass');
function addGrassPatch(x, z, w, d) {
  const lawn = new THREE.Mesh(new THREE.BoxGeometry(w, 0.12, d), grassPatchMat);
  lawn.position.set(x, 0.06, z);
  lawn.receiveShadow = true;
  propsGroup.add(lawn);
}

// A point on a road loop's edge: +offset metres toward the loop centre,
// -offset outside it (same right-hand normal stepIndustrialTraffic uses
// for its lane offset). Heading matches the vehicle rotation convention,
// so parked cars/benches orient with `rotation.y = heading + ...`.
function loopEdgePoint(l, t, offset) {
  const [wx, wy] = rectLoopPos(l.cx, l.cy, l.hw, l.hh, t);
  const [wx2, wy2] = rectLoopPos(l.cx, l.cy, l.hw, l.hh, t + 0.002);
  const hx = wx2 - wx, hy = wy2 - wy, hl = Math.hypot(hx, hy) || 1;
  const p = toScene(wx + (hy / hl) * offset, wy - (hx / hl) * offset);
  return { x: p.x, z: p.z, heading: Math.atan2(hy, hx) };
}

// Perimeter chain-link fence -- posts + wire strands around a fixed
// boundary centred on the AP, the "this is a contained facility, not
// buildings floating in a void" cue a real industrial site always has.
const fencePostMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.6, color: 0x55585c });
const fencePostGeo = new THREE.BoxGeometry(0.5, 3, 0.5);
const fenceWireMat = new THREE.LineBasicMaterial({ color: 0x9aa0a6 });
function addFencePerimeter(hw, hd) {
  const corners = [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd], [-hw, -hd]];
  const spacing = 22;
  for (let i = 0; i < 4; i++) {
    const [x0, z0] = corners[i], [x1, z1] = corners[i + 1];
    const len = Math.hypot(x1 - x0, z1 - z0);
    const n = Math.max(2, Math.round(len / spacing));
    for (let k = 0; k < n; k++) {
      const t = k / n;
      const post = new THREE.Mesh(fencePostGeo, fencePostMat);
      post.position.set(x0 + (x1 - x0) * t, 1.5, z0 + (z1 - z0) * t);
      post.castShadow = true;
      propsGroup.add(post);
    }
  }
  for (const hgt of [0.8, 1.6, 2.4]) {
    const pts = corners.map(([x, z]) => new THREE.Vector3(x, hgt, z));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    propsGroup.add(new THREE.Line(geo, fenceWireMat));
  }
}

// Elevated pipe-rack -- twin parallel pipe runs on support posts (plus a
// cross-brace at every post), linking two points the way a real process
// plant's tank farm feeds its reactor units. Same "merged oriented-box-
// segment strip" pattern as addPaddyStrip (Paddy Field's channel/track),
// just elevated and doubled, and built straight from world coordinates
// (like addFencePerimeter) rather than taking a `parent` -- it always
// belongs directly in propsGroup, which sits at scene-root identity.
const pipeMat = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.75, color: 0x8a97a0 });
const pipePostMat = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.7, color: 0x454f55 });
function addPipeRack(pts, elevation) {
  const runGeos = [], postGeos = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const [x0, z0] = pts[i], [x1, z1] = pts[i + 1];
    const dx = x1 - x0, dz = z1 - z0;
    const len = Math.hypot(dx, dz);
    if (len < 1e-3) continue;
    const ry = -Math.atan2(dz, dx);
    const nx = -dz / len, nz = dx / len;
    for (const off of [-1.1, 1.1]) {
      const run = new THREE.BoxGeometry(len, 0.7, 0.7);
      run.rotateY(ry);
      run.translate((x0 + x1) / 2 + nx * off, elevation, (z0 + z1) / 2 + nz * off);
      runGeos.push(run);
    }
    const nposts = Math.max(2, Math.round(len / 16));
    for (let k = 1; k < nposts; k++) {
      const t = k / nposts;
      const px = x0 + dx * t, pz = z0 + dz * t;
      const post = new THREE.BoxGeometry(0.4, elevation, 0.4);
      post.translate(px, elevation / 2, pz);
      postGeos.push(post);
      const brace = new THREE.BoxGeometry(3.0, 0.3, 0.3);
      brace.rotateY(ry);
      brace.translate(px, elevation, pz);
      postGeos.push(brace);
    }
  }
  addMergedMesh(runGeos, pipeMat, true);
  addMergedMesh(postGeos, pipePostMat, true);
}

// Ground-level hazard-striped pad -- a concrete apron with alternating
// yellow hazard stripes, the ground-level counterpart to the tiled
// TEX.warning bands addBuilding already puts on a smokestack -- a
// process-plant containment pad or loading apron reads the same cue.
// lazyMat, not a direct tiledClone(TEX.concrete, ...) -- this module-level
// const runs at import time, before app.js's main() has awaited
// loadAllTextures(), so TEX.concrete is still undefined here (see this
// file's own lazyMat comment up top for the full explanation of the bug
// that would otherwise cause).
const hazardApronMat = lazyMat('concrete');
const hazardStripeMat = new THREE.MeshBasicMaterial({ color: 0xe0b23d });
function addHazardStripe(x, z, w, d, ry) {
  const g = new THREE.Group();
  const apron = new THREE.Mesh(new THREE.BoxGeometry(w, 0.1, d), hazardApronMat);
  apron.position.y = 0.05;
  apron.receiveShadow = true;
  g.add(apron);
  const n = Math.max(3, Math.round(w / 4));
  for (let i = 0; i < n; i++) {
    if (i % 2 === 0) continue;
    const stripe = new THREE.Mesh(
      new THREE.BoxGeometry(w / n, 0.02, d),
      hazardStripeMat,
    );
    stripe.position.set(-w / 2 + (i + 0.5) * (w / n), 0.11, 0);
    g.add(stripe);
  }
  g.rotation.y = ry;
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

// Obstacle footprints scale with however spread-out the current node
// layout happens to be (~150m up to ~900m+ wide, see addTankFarm's
// comment) -- basing prop placement on a fixed radius meant scattered
// content either buried itself inside a huge building or, for a small
// site, spread out into visible empty gaps. Deriving the radius from the
// actual obstacles keeps the yard filled either way.
export function siteRadius(obstacles) {
  // Uses obstacle CENTRES, not their (possibly huge, see addBuilding's
  // comment) raw corner extents -- this only needs to reflect how far
  // apart the buildings actually sit so props scatter around them, not
  // how big the underlying obstacle rect is. Capped so a compact
  // "vertical world" site stays compact even if the node layout that
  // drove the obstacle positions happens to be very spread out.
  //
  // Was capped at 700, then 1400 -- both physically implausible for a real
  // downtown/industrial park once topology_canvas.py's _ENV_PATH_LOSS_EXP
  // gave each environment its own realistic outdoor link budget instead of
  // one shared "indoor/mixed" exponent: the AP's nominal range now runs
  // from ~1.8km (Industrial Site) up to ~2.97km (Paddy Field), and measured
  // node scatter routinely reaches most of that -- past the old 1400 cap,
  // meaning the decorative buildings were again being squeezed into a
  // visibly smaller box than the simulation itself spans. 3600 comfortably
  // covers the new realistic scatter range with headroom instead of
  // clipping against it; filler building COUNTS are mostly fixed per
  // environment (see addFillerBuildings/addCityFillerBuildings), so a
  // larger radius spreads the same buildings further apart rather than
  // adding more of them -- this is a density fix, not a "bigger city" fix.
  let maxR = 180;
  for (const o of obstacles || []) {
    const cx = o.kind === 'circle' ? o.cx : (o.x0 + o.x1) / 2;
    const cy = o.kind === 'circle' ? o.cy : (o.y0 + o.y1) / 2;
    maxR = Math.max(maxR, Math.hypot(cx, cy));
  }
  return Math.min(maxR, 3600);
}

// Smart City's road_loops are sized server-side from the live node/STA
// scatter (snapshot.py's _stable_bounds), which can run far past
// siteRadius(obstacles) -- that only reflects the ~5 real, tightly
// clustered campus buildings near the origin (ism_campus_obstacles()),
// not how far the actual node scatter (and therefore the roads) reach.
// At 802.11ah's real range this gap is routine, not an edge case: using
// siteRadius() alone as the filler-grid radius left the ENTIRE grid
// smaller than the inner loop's own plaza-exclusion zone, silently
// placing zero filler buildings and zero pocket-park content -- bare
// pavement out to the road loops with nothing on it, regardless of
// layout_variant. Sized off the loops themselves (when given) so the
// filler grid always actually reaches as far as the roads it sits on.
export function cityExtentR(obstacles, loops) {
  const base = siteRadius(obstacles);
  if (!loops || !loops.length) return base;
  const outer = loops.reduce((a, b) => (a.hw * a.hh >= b.hw * b.hh ? a : b));
  return Math.max(base, Math.hypot(outer.hw, outer.hh));
}

// Military Zone's own extent, UNCAPPED unlike siteRadius() above -- 802.11ah's
// real long range can put checkpoint bunkers 900m+ from the AP, well past
// siteRadius's 700 cap (that cap exists for other environments' prop-scatter
// density, not for "how far out does this base actually reach"). Exported so
// app.js's camera framing uses the exact same numbers rebuildMilitaryPatrol
// builds the perimeter/patrol loops/filler from -- they used to be computed
// two different ways, which could leave the patrolling tanks/soldiers
// sitting just outside the camera's default view.
export function militaryExtent(obstacles) {
  let hw = 220, hd = 220;
  for (const o of obstacles || []) {
    const cx = o.kind === 'circle' ? o.cx : (o.x0 + o.x1) / 2;
    const cy = o.kind === 'circle' ? o.cy : (o.y0 + o.y1) / 2;
    hw = Math.max(hw, Math.abs(cx) * 1.18 + 40);
    hd = Math.max(hd, Math.abs(cy) * 1.18 + 40);
  }
  return { hw, hd };
}

// At this prop density a blind radial scatter would constantly drop piles
// and parked vehicles onto the pavement the traffic loops drive, so
// placements are rejection-sampled against the road band (rect-outline
// SDF). Scene z = -world y, so an off-origin loop centre (Smart City's
// come from the server-side node bounds) lands at (cx, -cy) here;
// industrial loops are origin-centred and unaffected.
// Margin is per-prop -- a stockpile needs far more standoff than a barrel.
function loopDist(l, x, z) {
  const qx = Math.abs(x - (l.cx || 0)) - l.hw, qz = Math.abs(z + (l.cy || 0)) - l.hh;
  return Math.abs(Math.hypot(Math.max(qx, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qz), 0));
}
function clearOfRoads(x, z, loops, margin) {
  for (const l of loops) {
    if (loopDist(l, x, z) < ROAD_HALF_W + SIDEWALK_W + margin) return false;
  }
  return true;
}

function scatterPos(rng, rMin, rMax, loops, margin, cx = 0, cz = 0) {
  let x = cx, z = cz;
  for (let tries = 0; tries < 14; tries++) {
    const ang = rng() * Math.PI * 2, r = rMin + rng() * (rMax - rMin);
    x = cx + Math.cos(ang) * r; z = cz + Math.sin(ang) * r;
    if (clearOfRoads(x, z, loops, margin)) break;
  }
  return [x, z];
}

// Smart City's loops are server-derived from live node bounds, so they
// can drift continuously (e.g. aerial_relay/uav topologies keep flying
// nodes orbiting forever) -- bucketed coarsely so props/fillers only
// rebuild when the road has moved enough to actually matter visually.
// Even the server side now excludes flying relay/UAV nodes from this
// bounding box (see snapshot.py's _stable_bounds), so this bucket mostly
// guards against float jitter and the residual drift left in "uav" mode
// (every node flies there, so there's no fully static reference); it
// used to be 20m, which was tight enough that a real rebuild -- and a
// full building/prop reshuffle, since a different site radius shifts the
// whole filler grid's alignment -- could fire with no user action at all.
// Empty string for every other environment keeps their rebuild keys
// unchanged.
function cityLoopSig(env, loops) {
  if (env !== 'Smart City') return '';
  return '|' + (loops || []).map(l =>
    [l.cx, l.cy, l.hw, l.hh].map(v => Math.round(v / 60)).join(',')).join(';');
}

// Paddy Field has no RF obstacles for siteRadius() to size itself off of
// (it's open terrain, no buildings), so it stays pinned at that
// function's floor regardless of how far the STAs actually scatter --
// this bucket brings the node-derived extent (see app.js's
// nodeExtentRadius, passed in as rebuildProps's nodeExtentR) into the key
// so the plot mosaic actually grows/shrinks with the real node spread.
function paddyExtentSig(env, nodeExtentR) {
  if (env !== 'Paddy Field') return '';
  return '|' + Math.round((nodeExtentR || 0) / 50);
}

// ---- Paddy Field: a working rice-farm landscape -- a shared-corner
// jittered plot lattice (flooded, seedling, growing, ripening and
// stubble plots in zoned patches on gently terraced ground), a river
// with a plank footbridge, an irrigation canal, a dirt farm track, a
// fenced farmstead, field life (farmers, buffalo, egrets, a scarecrow)
// and a hazed mountain-and-terrace backdrop. The palette comes straight
// from the 2D topology canvas's three paddy layouts (_PADDY_*/_HUT_* in
// _bg_paddy_v1, _RIVER_*/_BRIDGE_*/_MTN_* in _bg_paddy_v2 "River
// Valley", _TERRACE_* in _bg_paddy_v3 "Highland Terraces") so both
// views read as the same place. Previously this environment had no
// geometry of its own at all: rebuildProps fell through to Open Area's
// bare tree-ring scatter. -------------------------------------------------
const paddyBundMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0x9c8a5e });
const paddyWaterMat = new THREE.MeshStandardMaterial({ roughness: 0.12, metalness: 0, color: 0x7fb6cf });
const paddyGlintMat = new THREE.MeshBasicMaterial({ color: 0xeef8fb });
// One near-white row-striped tile tinted per stage (Lambert multiplies
// map x colour), so all seven growth stages share a single texture yet
// keep the exact _PADDY_CROPS hex progression green -> ripe gold ->
// stubble, with the transplant-row grain visible in each.
const paddyCropMats = ['#6fb04e', '#83c15f', '#9ed36f', '#57964a', '#b6de84', '#c8c266', '#dcd28e']
  .map((c) => lazyMat('paddy_crop', { color: c }));
const paddySproutMat = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0, color: 0xffffff });
const paddySproutColors = [0x4c7a3e, 0x57964a].map((c) => new THREE.Color(c));
const paddyChannelMat = new THREE.MeshStandardMaterial({ roughness: 0.15, metalness: 0, color: 0x5f9dbf });
const paddyChannelLtMat = new THREE.MeshBasicMaterial({ color: 0xa9d8e8 });
const paddyTrackMat = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, color: 0xc7ae7f });
const paddyRutMat = new THREE.MeshBasicMaterial({ color: 0xa3967c });
// Doubles as straw (haystacks, figure heads) -- mud-plaster and rice
// straw share a tone in the 2D hut palette.
const paddyHutWallMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0xc9a36a });
const paddyRiverMat = new THREE.MeshStandardMaterial({ roughness: 0.15, metalness: 0, color: 0x63849a });
const paddyRiverBankMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0xb2b995 });
const paddyRiverLtMat = new THREE.MeshBasicMaterial({ color: 0xa4bbc7 });
const paddyBridgeDeckMat = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0, color: 0x8b7761 });
const paddyBridgeRailMat = new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0, color: 0x645440 });
const paddyHatMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0xdcd28e });
const paddyShirtMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0x63849a });
const paddyTrouserMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0x645440 });
const paddyBuffaloMat = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0, color: 0x5d6156 });
const paddyHornMat = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0, color: 0xabaea1 });
const paddyEgretMat = new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0, color: 0xe7ebeb });
const paddyBeakMat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0, color: 0xc7ae7f });
const paddyYardMat = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, 
  color: 0xc7ae7f, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
});
// flatShading so the low-segment backdrop cones read as faceted low-poly
// massifs rather than smooth plastic humps. The back range is pre-tinted
// with the 2D canvas's aerial-perspective haze tone; distance fog then
// finishes the job.
const paddyMtnBaseMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0x7d8175, flatShading: true });
const paddyMtnMidMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0x93968b, flatShading: true });
const paddyMtnHazeMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0xc3c6bb, flatShading: true });
const paddyMtnSnowMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0, color: 0xe7ebeb, flatShading: true });
const paddyTerraceMats = ['#4f8a48', '#5f9852', '#71a75e', '#85b76e', '#9bc880', '#b3d494', '#c9dda8']
  .map((c) => new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: c, flatShading: true }));
const paddyCanopyMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0xffffff });
const paddyTreeGreens = [0x3f7a3f, 0x2e5c30, 0x4f9450].map((c) => new THREE.Color(c));
const PADDY_SPROUT_GEO = new THREE.ConeGeometry(0.5, 1, 5);
const PADDY_TRUNK_GEO = new THREE.CylinderGeometry(0.5, 0.7, 1, 5);
const PADDY_CANOPY_GEO = new THREE.IcosahedronGeometry(1, 0);
const PADDY_MAX_PLOTS = 720;
const PADDY_MAX_SPROUTS = 4200;
// [near, far] while a Paddy Field scene is built, null otherwise -- the
// mist band is applied through rebuildMilitaryPatrol's per-poll fog
// reset (which runs after rebuildProps and would stomp a direct
// setFogRange call here on the very same poll).
let paddyFogBand = null;

// Order-independent per-cell hash (same role as the 2D canvas's
// _hash01(gx, gy, k)) -- lattice corners are shared between neighbouring
// plots, so their jitter must depend only on the corner's own grid
// coordinates, never on iteration order or on how many rng() draws
// happened before them.
function paddyHash01(gx, gz, k) {
  let h = (Math.imul(gx, 0x27d4eb2d) ^ Math.imul(gz, 0x165667b1) ^ Math.imul(k + 1, 0x9e3779b9)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function polylineDist(x, z, pts) {
  let best = Infinity;
  for (let i = 0; i < pts.length - 1; i++) {
    const [x0, z0] = pts[i], [x1, z1] = pts[i + 1];
    const dx = x1 - x0, dz = z1 - z0;
    const lenSq = dx * dx + dz * dz || 1;
    let t = ((x - x0) * dx + (z - z0) * dz) / lenSq;
    t = Math.max(0, Math.min(1, t));
    best = Math.min(best, Math.hypot(x - (x0 + dx * t), z - (z0 + dz * t)));
  }
  return best;
}

// Vertical prism from a flat polygon, baked into world coordinates so
// hundreds of plots can merge into a handful of draw calls. Corners go
// +z-first (c00, c01, c11, c10): the shape plane's y axis is -worldZ, so
// that order keeps the polygon CCW and the top cap's normals pointing
// up. holePts (reverse-wound automatically) turns the prism into a
// basin rim whose inner walls stay visible -- a solid slab's top cap
// would hide the recessed water surface entirely. uvSpin rotates and
// scales the cap UVs into 6m-per-tile world space, giving each plot its
// own transplant-row direction from one shared texture -- which is what
// stops the merged mosaic reading as one giant wallpapered plane.
function paddyPrism(pts, baseY, topY, uvSpin, holePts) {
  const shape = new THREE.Shape();
  shape.moveTo(pts[0][0], -pts[0][1]);
  for (let i = 1; i < pts.length; i++) shape.lineTo(pts[i][0], -pts[i][1]);
  if (holePts) {
    const hole = new THREE.Path();
    hole.moveTo(holePts[holePts.length - 1][0], -holePts[holePts.length - 1][1]);
    for (let i = holePts.length - 2; i >= 0; i--) hole.lineTo(holePts[i][0], -holePts[i][1]);
    shape.holes.push(hole);
  }
  const geo = new THREE.ExtrudeGeometry(shape, { depth: topY - baseY, bevelEnabled: false });
  geo.rotateX(-Math.PI / 2);
  geo.translate(0, baseY, 0);
  if (uvSpin) {
    const uv = geo.attributes.uv;
    const c = Math.cos(uvSpin.ang), s = Math.sin(uvSpin.ang);
    for (let i = 0; i < uv.count; i++) {
      const du = uv.getX(i) - uvSpin.cu, dv = uv.getY(i) - uvSpin.cv;
      uv.setXY(i, (c * du - s * dv) / 6, (s * du + c * dv) / 6);
    }
  }
  return geo;
}

function addMergedMesh(geos, mat, castShadow = false) {
  if (!geos.length) return;
  const merged = mergeGeometries(geos, false);
  for (const g of geos) g.dispose();
  const mesh = new THREE.Mesh(merged, mat);
  mesh.receiveShadow = true;
  mesh.castShadow = castShadow;
  propsGroup.add(mesh);
}

// A winding watercourse or dirt track through a chain of world-space
// points -- overlapping oriented box segments baked into one merged mesh
// (plus one for the sheen/wheel-rut accent) rather than a mesh per
// segment, but still the same "many simple blocky pieces" read the rest
// of this file uses for roads and fences.
function addPaddyStrip(pts, halfWidth, mat, thickness, y, ltMat) {
  const segs = [], accents = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const [x0, z0] = pts[i], [x1, z1] = pts[i + 1];
    const dx = x1 - x0, dz = z1 - z0;
    const len = Math.hypot(dx, dz);
    if (len < 1e-3) continue;
    const ry = -Math.atan2(dz, dx);
    const seg = new THREE.BoxGeometry(len + halfWidth, thickness, halfWidth * 2);
    seg.rotateY(ry);
    seg.translate((x0 + x1) / 2, y, (z0 + z1) / 2);
    segs.push(seg);
    if (ltMat) {
      const sheen = new THREE.BoxGeometry(len + halfWidth, 0.02, halfWidth * 0.35);
      sheen.rotateY(ry);
      sheen.translate((x0 + x1) / 2, y + thickness / 2 + 0.01, (z0 + z1) / 2);
      accents.push(sheen);
    }
  }
  addMergedMesh(segs, mat);
  if (ltMat) addMergedMesh(accents, ltMat);
}

// River centreline z at a given x (riverPts run left-to-right) -- used to
// find where the farm track crosses so the bridge lands exactly on the
// water rather than merely near it.
function paddyRiverZAt(pts, x) {
  if (x <= pts[0][0]) return pts[0][1];
  for (let i = 0; i < pts.length - 1; i++) {
    if (x <= pts[i + 1][0]) {
      const f = (x - pts[i][0]) / (pts[i + 1][0] - pts[i][0] || 1);
      return pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f;
    }
  }
  return pts[pts.length - 1][1];
}

// Farmhouse: mud-plaster walls on a low earthen plinth under a thatch
// gable -- the 2D canvas's hut palette is straw-brown, so the city's
// terracotta shingle would read as the wrong building out here.
function addFarmhouse(x, z, rng) {
  const g = new THREE.Group();
  // Sized to clearly read as a building next to a 9m-footprint/13m-tall
  // STA node marker, not a same-scale box beside it -- a life-size 9-12m
  // farmhouse was practically indistinguishable from the node icons in
  // the 3D view.
  const w = 17 + rng() * 6, d = 12 + rng() * 3, h = 7.0;
  const plinth = new THREE.Mesh(new THREE.BoxGeometry(w + 1.4, 0.5, d + 1.4), paddyBundMat);
  plinth.position.y = 0.25;
  plinth.receiveShadow = true;
  g.add(plinth);
  const wall = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), paddyHutWallMat);
  wall.position.y = 0.5 + h / 2;
  wall.castShadow = true;
  wall.receiveShadow = true;
  g.add(wall);
  const door = new THREE.Mesh(new THREE.BoxGeometry(2.0, 3.0, 0.2), paddyBridgeRailMat);
  door.position.set(0, 2.0, d / 2 + 0.05);
  g.add(door);
  addGableRoof(g, w, d, 4.6, h + 0.5, TEX.thatch);
  g.position.set(x, 0, z);
  g.rotation.y = rng() * Math.PI * 2;
  propsGroup.add(g);
}

// ---- farm life: blocky figures at people/animal scale. Sub-pixel from
// a whole-network view, but 802.11ah paddy scenarios are exactly the
// ones users zoom into a single sensor on -- this is what's standing
// next to it when they do.
function addFarmer(x, z, y, ry) {
  const g = new THREE.Group();
  const legs = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.9, 0.42), paddyTrouserMat);
  legs.position.y = 0.45;
  g.add(legs);
  const torso = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.85, 0.44), paddyShirtMat);
  torso.position.set(0, 1.12, 0.14);
  torso.rotation.x = 0.7; // bent over the seedlings, not standing at attention
  g.add(torso);
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), paddyHutWallMat);
  head.position.set(0, 1.42, 0.52);
  g.add(head);
  const hat = new THREE.Mesh(new THREE.ConeGeometry(0.6, 0.3, 6), paddyHatMat);
  hat.position.set(0, 1.64, 0.52);
  g.add(hat);
  g.traverse((o) => { o.castShadow = true; });
  g.position.set(x, y, z);
  g.rotation.y = ry;
  propsGroup.add(g);
}

function addBuffalo(x, z, y, ry) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(3.0, 1.5, 1.5), paddyBuffaloMat);
  body.position.y = 1.55;
  g.add(body);
  for (const [lx, lz] of [[-1.05, -0.5], [-1.05, 0.5], [1.05, -0.5], [1.05, 0.5]]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.9, 0.34), paddyBuffaloMat);
    leg.position.set(lx, 0.45, lz);
    g.add(leg);
  }
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.85, 0.85), paddyBuffaloMat);
  head.position.set(1.85, 1.35, 0);
  g.add(head);
  for (const side of [-1, 1]) {
    const horn = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.16, 0.16), paddyHornMat);
    horn.position.set(1.85, 1.85, side * 0.5);
    horn.rotation.y = side * 0.7;
    horn.rotation.z = 0.35;
    g.add(horn);
  }
  const tail = new THREE.Mesh(new THREE.BoxGeometry(0.14, 1.1, 0.14), paddyBuffaloMat);
  tail.position.set(-1.55, 1.35, 0);
  g.add(tail);
  g.traverse((o) => { o.castShadow = true; });
  g.position.set(x, y, z);
  g.rotation.y = ry;
  propsGroup.add(g);
}

function addEgret(x, z, y, ry) {
  const g = new THREE.Group();
  const legs = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.5, 0.06), paddyBeakMat);
  legs.position.y = 0.25;
  g.add(legs);
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.34, 0.3), paddyEgretMat);
  body.position.y = 0.62;
  g.add(body);
  const neck = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.5, 0.09), paddyEgretMat);
  neck.position.set(0.26, 0.95, 0);
  g.add(neck);
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.16, 0.14), paddyEgretMat);
  head.position.set(0.3, 1.2, 0);
  g.add(head);
  const beak = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.05, 0.05), paddyBeakMat);
  beak.position.set(0.5, 1.2, 0);
  g.add(beak);
  g.position.set(x, y, z);
  g.rotation.y = ry;
  propsGroup.add(g);
}

function addScarecrow(x, z, y, ry) {
  const g = new THREE.Group();
  const pole = new THREE.Mesh(new THREE.BoxGeometry(0.16, 2.3, 0.16), paddyBridgeRailMat);
  pole.position.y = 1.15;
  g.add(pole);
  const arms = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.16, 0.16), paddyBridgeRailMat);
  arms.position.y = 1.62;
  g.add(arms);
  const shirt = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.85, 0.5), paddyShirtMat);
  shirt.position.y = 1.35;
  g.add(shirt);
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), paddyHutWallMat);
  head.position.y = 2.02;
  g.add(head);
  const hat = new THREE.Mesh(new THREE.ConeGeometry(0.55, 0.3, 6), paddyHatMat);
  hat.position.y = 2.26;
  g.add(hat);
  g.traverse((o) => { o.castShadow = true; });
  g.position.set(x, y, z);
  g.rotation.y = ry;
  propsGroup.add(g);
}

function addHaystack(x, z, y, s) {
  const stack = new THREE.Mesh(new THREE.ConeGeometry(1.5 * s, 2.4 * s, 7), paddyHutWallMat);
  stack.position.set(x, y + 1.2 * s, z);
  stack.castShadow = true;
  propsGroup.add(stack);
}

// Blocky palm -- trunk from the shared log texture, fronds as drooping
// slats. The round broadleaf addTree stays on the river banks; palms
// mark the farmstead the way they do around real paddy homesteads.
function addPalmTree(x, z, s, rng) {
  const g = new THREE.Group();
  const h = (10 + rng() * 4) * s;
  const trunk = new THREE.Mesh(new THREE.BoxGeometry(0.8, h, 0.8), trunkMat);
  trunk.position.y = h / 2;
  trunk.castShadow = true;
  g.add(trunk);
  for (let i = 0; i < 6; i++) {
    const sub = new THREE.Group();
    sub.rotation.y = (i / 6) * Math.PI * 2 + rng() * 0.5;
    const frond = new THREE.Mesh(new THREE.BoxGeometry(3.6 * s, 0.16, 0.9 * s), leafMat);
    frond.position.set(1.7 * s, h - 0.1, 0);
    frond.rotation.z = -0.5;
    frond.castShadow = true;
    sub.add(frond);
    g.add(sub);
  }
  g.position.set(x, 0, z);
  propsGroup.add(g);
}

// Rice-drying rack: sheaves hung over a bar to dry in the dooryard.
function addDryingRack(x, z, ry, rng) {
  const g = new THREE.Group();
  for (const side of [-1, 1]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.22, 2.2, 0.22), paddyBridgeRailMat);
    post.position.set(side * 3, 1.1, 0);
    g.add(post);
  }
  const bar = new THREE.Mesh(new THREE.BoxGeometry(6.4, 0.16, 0.16), paddyBridgeRailMat);
  bar.position.y = 2.1;
  g.add(bar);
  for (let i = 0; i < 6; i++) {
    const sheaf = new THREE.Mesh(new THREE.BoxGeometry(0.7, 1.2, 0.34), paddyHatMat);
    sheaf.position.set(-2.4 + i * (0.96 + rng() * 0.1), 1.5, 0);
    g.add(sheaf);
  }
  g.traverse((o) => { o.castShadow = true; });
  g.position.set(x, 0, z);
  g.rotation.y = ry;
  propsGroup.add(g);
}

// Farmstead: house + tool shed + drying rack + haystacks + palms inside
// a post-and-rail fence, all on a packed-earth blob so the compound sits
// in a yard rather than directly on field base.
function addFarmstead(x, z, rng) {
  const yard = new THREE.Mesh(makeBlobGeometry(10, 0.3, 0x9a77), paddyYardMat);
  yard.rotation.x = -Math.PI / 2;
  yard.scale.set(24, 19, 1);
  yard.position.set(x, 0.05, z);
  yard.receiveShadow = true;
  propsGroup.add(yard);
  addFarmhouse(x, z, rng);
  const shed = new THREE.Group();
  const sw = 8 + rng() * 2, sd = 6 + rng() * 1.5, sh = 3.6;
  const sWall = new THREE.Mesh(new THREE.BoxGeometry(sw, sh, sd), paddyHutWallMat);
  sWall.position.y = sh / 2;
  sWall.castShadow = true;
  shed.add(sWall);
  addGableRoof(shed, sw, sd, 2.0, sh, TEX.thatch);
  shed.position.set(x + 13 + rng() * 2, 0, z - 9);
  shed.rotation.y = rng() * Math.PI;
  propsGroup.add(shed);
  addDryingRack(x - 12, z + 9, rng() * Math.PI, rng);
  addHaystack(x + 8, z + 10, 0, 1.1 + rng() * 0.5);
  addHaystack(x + 12.5, z + 8, 0, 0.9 + rng() * 0.4);
  addPalmTree(x - 13, z - 8, 1.1 + rng() * 0.3, rng);
  addPalmTree(x + 15, z + 2, 1.0 + rng() * 0.3, rng);
  // Post-and-rail fence on three sides, open toward the track.
  const fGeos = [];
  const sides = [
    [x - 19, z - 15, x + 19, z - 15],
    [x - 19, z - 15, x - 19, z + 15],
    [x - 19, z + 15, x + 19, z + 15],
  ];
  for (const [x0, z0, x1, z1] of sides) {
    const len = Math.hypot(x1 - x0, z1 - z0);
    const ry = -Math.atan2(z1 - z0, x1 - x0);
    const nPosts = Math.round(len / 3.5);
    for (let i = 0; i <= nPosts; i++) {
      const t = i / nPosts;
      const post = new THREE.BoxGeometry(0.28, 1.3, 0.28);
      post.translate(x0 + (x1 - x0) * t, 0.65, z0 + (z1 - z0) * t);
      fGeos.push(post);
    }
    for (const railY of [0.6, 1.1]) {
      const rail = new THREE.BoxGeometry(len, 0.12, 0.12);
      rail.rotateY(ry);
      rail.translate((x0 + x1) / 2, railY, (z0 + z1) / 2);
      fGeos.push(rail);
    }
  }
  addMergedMesh(fGeos, paddyBridgeRailMat, true);
}

// Plank footbridge where the farm track crosses the river -- straight
// from _bg_paddy_v2's bridge treatment: deck over the water, side rails,
// piles, and packed-earth ramps meeting the track at both ends.
function addPaddyBridge(x, z, ry, span) {
  const g = new THREE.Group();
  const deck = new THREE.Mesh(new THREE.BoxGeometry(span, 0.28, 3.8), paddyBridgeDeckMat);
  deck.position.y = 1.35;
  deck.castShadow = true;
  g.add(deck);
  for (const side of [-1, 1]) {
    const rail = new THREE.Mesh(new THREE.BoxGeometry(span, 0.14, 0.14), paddyBridgeRailMat);
    rail.position.set(0, 2.15, side * 1.75);
    g.add(rail);
    for (let i = -2; i <= 2; i++) {
      const post = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.85, 0.18), paddyBridgeRailMat);
      post.position.set(i * span * 0.22, 1.75, side * 1.75);
      g.add(post);
    }
    const pile = new THREE.Mesh(new THREE.BoxGeometry(0.4, 1.6, 0.4), paddyBridgeRailMat);
    pile.position.set(side * span * 0.22, 0.6, side * 1.2);
    g.add(pile);
    const ramp = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.7, 3.8), paddyTrackMat);
    ramp.position.set(side * (span / 2 + 1.0), 0.35, 0);
    g.add(ramp);
  }
  g.position.set(x, 0, z);
  g.rotation.y = ry;
  propsGroup.add(g);
}

function addPaddyPeak(x, z, h, mat, rng, snow) {
  const r = h * (1.5 + rng() * 0.5);
  const cone = new THREE.Mesh(new THREE.ConeGeometry(r, h, 6), mat);
  cone.position.set(x, h / 2 - 1, z);
  cone.rotation.y = rng() * Math.PI;
  propsGroup.add(cone);
  if (snow) {
    const cap = new THREE.Mesh(new THREE.ConeGeometry(r * 0.38, h * 0.38, 6), paddyMtnSnowMat);
    cap.position.set(x, h * 0.81, z);
    cap.rotation.y = cone.rotation.y;
    propsGroup.add(cap);
  }
}

// Stepped terrace hill -- _bg_paddy_v3's "paddies climbing the hillside"
// as a stack of shrinking discs, valley-dark at the base to pale at the
// crest, the same shade progression as the 2D _TERRACE_SHADES ramp.
function addTerraceHill(x, z, r0, rng) {
  let r = r0, y = 0;
  for (let t = 0; t < 6; t++) {
    const th = 3.0 + rng() * 1.6;
    const tier = new THREE.Mesh(
      new THREE.CylinderGeometry(r * 0.82, r, th, 9),
      paddyTerraceMats[Math.min(1 + t, paddyTerraceMats.length - 1)]);
    tier.position.set(x + (rng() - 0.5) * r * 0.2, y + th / 2, z + (rng() - 0.5) * r * 0.2);
    tier.rotation.y = rng() * Math.PI;
    tier.receiveShadow = true;
    propsGroup.add(tier);
    y += th;
    r *= 0.68;
  }
}

// Broadleaf treeline closing the horizon everywhere the mountains
// don't -- two InstancedMesh calls total, same low-draw-call pattern as
// the military terrain filler, but with round icosahedron canopies in
// the 2D canvas's tree greens instead of conifer cones.
function addPaddyTreeline(rng, R, north) {
  const pts = [];
  for (let c = 0; c < 15; c++) {
    const a = rng() * Math.PI * 2;
    const nT = 3 + ((rng() * 5) | 0);
    // The mountain range owns its stretch of horizon.
    if (Math.abs(Math.atan2(Math.sin(a - north), Math.cos(a - north))) < 0.85) continue;
    const cr = R * (1.06 + rng() * 0.38);
    for (let t = 0; t < nT; t++) {
      pts.push([Math.cos(a) * cr + (rng() - 0.5) * 46, Math.sin(a) * cr + (rng() - 0.5) * 46]);
    }
  }
  if (!pts.length) return;
  const trunks = new THREE.InstancedMesh(PADDY_TRUNK_GEO, MIL_TRUNK_MAT, pts.length);
  const canopies = new THREE.InstancedMesh(PADDY_CANOPY_GEO, paddyCanopyMat, pts.length);
  trunks.frustumCulled = false;
  canopies.frustumCulled = false;
  trunks.castShadow = true;
  canopies.castShadow = true;
  pts.forEach(([x, z], i) => {
    const h = 10 + rng() * 8;
    dummy.position.set(x, h / 2, z);
    dummy.rotation.set(0, rng() * Math.PI, 0);
    dummy.scale.set(1, h, 1);
    dummy.updateMatrix();
    trunks.setMatrixAt(i, dummy.matrix);
    const cr = h * (0.45 + rng() * 0.18);
    dummy.position.set(x, h * 0.92, z);
    dummy.scale.set(cr, cr * (0.8 + rng() * 0.3), cr);
    dummy.updateMatrix();
    canopies.setMatrixAt(i, dummy.matrix);
    canopies.setColorAt(i, paddyTreeGreens[(rng() * paddyTreeGreens.length) | 0]);
  });
  trunks.instanceMatrix.needsUpdate = true;
  canopies.instanceMatrix.needsUpdate = true;
  if (canopies.instanceColor) canopies.instanceColor.needsUpdate = true;
  propsGroup.add(trunks);
  propsGroup.add(canopies);
}

// Mountain range walling off the field's north, as in both the 2D
// "River Valley" and "Highland Terraces" layouts: a haze-tinted back
// row, a darker front row with snow on its tallest peaks, and terraced
// foothills at the arc's shoulders. Everything sits past the plot
// mosaic, so the field never just stops on bare ground.
function addPaddyBackdrop(rng, R) {
  const north = -Math.PI / 2; // scene -z == world north, where the 2D variants put the range
  for (let i = 0; i < 6; i++) {
    const a = north + (i / 5 - 0.5) * 2.4 + (rng() - 0.5) * 0.1;
    const r = R * (2.25 + rng() * 0.25);
    const h = Math.min(430, Math.max(70, r * (0.13 + rng() * 0.05)));
    addPaddyPeak(Math.cos(a) * r, Math.sin(a) * r, h, paddyMtnHazeMat, rng, false);
  }
  const peaks = [];
  for (let i = 0; i < 8; i++) {
    const a = north + (i / 7 - 0.5) * 2.0 + (rng() - 0.5) * 0.08;
    const r = R * (1.85 + rng() * 0.2);
    const h = Math.min(380, Math.max(55, r * (0.11 + rng() * 0.07)));
    peaks.push([Math.cos(a) * r, Math.sin(a) * r, h, rng() < 0.5]);
  }
  peaks.sort((p, q) => q[2] - p[2]);
  peaks.forEach(([px, pz, h, mid], i) => {
    addPaddyPeak(px, pz, h, mid ? paddyMtnMidMat : paddyMtnBaseMat, rng, i < 3);
  });
  for (const side of [-1, 1]) {
    addTerraceHill(
      Math.cos(north + side * 0.85) * R * 1.62,
      Math.sin(north + side * 0.85) * R * 1.62,
      R * (0.15 + rng() * 0.04), rng);
  }
  addPaddyTreeline(rng, R, north);
}

function addPaddyField(rng, R) {
  const cell = Math.max(34, R / 15);

  // Watercourses and the track come first: the lattice carves plots away
  // from them, leaving bare dry-earth corridors the way real field
  // blocks give way to a river's floodplain and road verges.
  const n = 16;
  const riverPts = [], canalPts = [], trackPts = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    riverPts.push([-R * 1.04 + R * 2.08 * t, Math.sin(t * 3.1 + 0.4) * R * 0.2 - R * 0.12]);
    const z = -R * 1.04 + R * 2.08 * t;
    canalPts.push([Math.sin(t * 2.6 + 2.1) * R * 0.11 - R * 0.4, z]);
    trackPts.push([Math.sin(t * 2.1 + 1.3) * R * 0.14 + R * 0.3, z]);
  }
  const riverHalfW = Math.max(4.5, cell * 0.16);
  const canalHalfW = Math.max(2.2, cell * 0.07);
  const fi = Math.floor(n * 0.72);
  const farmX = trackPts[fi][0] + 22, farmZ = trackPts[fi][1];

  // Plot lattice with shared jittered corners (the 2D canvas's own
  // corner(gx, gy) trick): neighbouring plots reuse the same displaced
  // corner, so field boundaries wander organically and never open gaps.
  // Everything gets merged per material -- one bund mesh, one water
  // mesh, one mesh per crop stage -- so plot count buys richness, not
  // draw calls.
  const half = Math.ceil((R * 0.99) / cell) + 1;
  const corner = (gx, gz) => [
    gx * cell + (paddyHash01(gx, gz, 1) - 0.5) * 0.34 * cell,
    gz * cell + (paddyHash01(gx, gz, 2) - 0.5) * 0.34 * cell,
  ];
  const bundGeos = [], waterGeos = [], glintGeos = [];
  const cropGeos = paddyCropMats.map(() => []);
  const sprouts = [];
  const waterPlots = [], seedlingPlots = [], ripePlots = [], stubblePlots = [];
  let placed = 0;
  for (let gx = -half; gx < half && placed < PADDY_MAX_PLOTS; gx++) {
    for (let gz = -half; gz < half && placed < PADDY_MAX_PLOTS; gz++) {
      const c00 = corner(gx, gz), c10 = corner(gx + 1, gz);
      const c11 = corner(gx + 1, gz + 1), c01 = corner(gx, gz + 1);
      const cx = (c00[0] + c10[0] + c11[0] + c01[0]) / 4;
      const cz = (c00[1] + c10[1] + c11[1] + c01[1]) / 4;
      const dist = Math.hypot(cx, cz);
      if (dist > R || dist < 34) continue; // keep clear of the AP mast's own footing
      // Carve against every corner, not just the centroid: a snug
      // per-corner test lets plots run right up to the banks (a blanket
      // centroid margin wide enough to be safe left huge bare corridors
      // through the middle of the field).
      const quad = [c00, c01, c11, c10];
      const near = (pts, keep) => polylineDist(cx, cz, pts) < keep
        || quad.some(([px, pz]) => polylineDist(px, pz, pts) < keep);
      if (near(riverPts, riverHalfW * 1.9 + 1)) continue;
      if (near(canalPts, canalHalfW * 1.8 + 1)) continue;
      if (near(trackPts, 4.2)) continue;
      if (quad.some(([px, pz]) => Math.hypot(px - farmX, pz - farmZ) < 30)) continue;
      placed++;

      // Growth stages arrive in zoned patches, not per-plot noise -- a
      // real paddy landscape is planted block by block, so neighbouring
      // plots usually share a stage. Ground height rides a second,
      // slower field, quantised so the land steps in dead-level terraces
      // (each paddy must hold standing water) instead of tilting.
      const wob = Math.sin(cx * 0.9 / cell + 1.7) * Math.cos(cz * 0.75 / cell - 0.4)
        + 0.45 * Math.sin(cx * 2.3 / cell - 0.6) * Math.cos(cz * 1.9 / cell + 1.1);
      const v = Math.min(1, Math.max(0, 0.5 + 0.34 * wob + (paddyHash01(gx, gz, 3) - 0.5) * 0.3));
      const ter = 0.5 + 0.5 * Math.sin(cx * 0.5 / cell + 0.3) * Math.cos(cz * 0.42 / cell + 2.0);
      const baseY = Math.floor(ter * 3.999) * 0.24;
      const bundTop = baseY + 0.5 + paddyHash01(gx, gz, 4) * 0.12;

      const insetF = Math.max(0.84, 1 - 2.6 / cell); // ~2.6m bund crown at any plot size
      const rim = quad.map(([px, pz]) => [cx + (px - cx) * insetF, cz + (pz - cz) * insetF]);
      // The surface slab is cut slightly wider than the rim opening so
      // its edges bury inside the bund solid instead of z-fighting
      // against coincident basin walls.
      const slab = quad.map(([px, pz]) => [cx + (px - cx) * (insetF + 0.02), cz + (pz - cz) * (insetF + 0.02)]);
      bundGeos.push(paddyPrism(quad, 0, bundTop, null, rim));
      const ang = paddyHash01(gx, gz, 5) * Math.PI;

      if (v < 0.34) {
        // Flooded: water sits recessed inside the bund basin. The
        // youngest plots get transplant rows of seedlings poking out of
        // the water -- the single most paddy-defining sight there is.
        const wy = bundTop - 0.26;
        waterGeos.push(paddyPrism(slab, wy - 0.08, wy));
        if (v < 0.13) {
          waterPlots.push([cx, cz, wy]);
          if (rng() < 0.6) {
            const glint = new THREE.BoxGeometry(cell * insetF * (0.2 + rng() * 0.2), 0.02, 0.45);
            glint.translate(cx + (rng() - 0.5) * cell * 0.3, wy + 0.03, cz + (rng() - 0.5) * cell * 0.3);
            glintGeos.push(glint);
          }
        } else {
          seedlingPlots.push([cx, cz, wy]);
          const ca = Math.cos(ang), sa = Math.sin(ang);
          const ext = cell * insetF * 0.4;
          const pitch = Math.max(3.0, cell / 8);
          let row = 0;
          for (let rz = -ext; rz <= ext; rz += pitch, row++) {
            for (let rx = -ext; rx <= ext; rx += pitch * 0.85) {
              if (sprouts.length >= PADDY_MAX_SPROUTS) break;
              const jx = rx + (rng() - 0.5) * 0.8, jz = rz + (rng() - 0.5) * 0.5;
              sprouts.push([cx + jx * ca - jz * sa, wy, cz + jx * sa + jz * ca,
                0.6 + rng() * 0.45, row & 1]);
            }
          }
        }
      } else {
        // Planted: the surface slab is the canopy itself, rising with
        // the growth stage until harvest drops it back to pale stubble.
        const stage = Math.min(6, ((v - 0.34) / 0.66 * 7) | 0);
        const canopyH = stage === 6 ? 0.12 : 0.16 + (stage / 5) * 0.72;
        const sy = bundTop - 0.3;
        cropGeos[stage].push(paddyPrism(slab, sy, sy + canopyH, { cu: cx, cv: -cz, ang }));
        if (stage === 5) ripePlots.push([cx, cz, sy + canopyH]);
        if (stage === 6) stubblePlots.push([cx, cz, sy + canopyH]);
      }
    }
  }
  addMergedMesh(bundGeos, paddyBundMat);
  addMergedMesh(waterGeos, paddyWaterMat);
  addMergedMesh(glintGeos, paddyGlintMat);
  cropGeos.forEach((geos, i) => addMergedMesh(geos, paddyCropMats[i]));

  if (sprouts.length) {
    const mesh = new THREE.InstancedMesh(PADDY_SPROUT_GEO, paddySproutMat, sprouts.length);
    mesh.frustumCulled = false;
    sprouts.forEach(([x, y, z, s, ci], i) => {
      dummy.position.set(x, y + s * 0.4, z);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, paddySproutColors[ci]);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    propsGroup.add(mesh);
  }

  // River layering straight from the 2D _river(): floodplain grass band,
  // then the channel, then a thin sheen. The water slab's top must clear
  // the wider bank slab's top or the bank swallows the water whole and
  // the whole river reads as a pale road. Same recipe, narrower, for the
  // irrigation canal (bund-earth banks) and the track (wheel-rut accent).
  addPaddyStrip(riverPts, riverHalfW * 1.9, paddyRiverBankMat, 0.16, 0.08);
  addPaddyStrip(riverPts, riverHalfW, paddyRiverMat, 0.3, 0.17, paddyRiverLtMat);
  addPaddyStrip(canalPts, canalHalfW * 1.8, paddyBundMat, 0.2, 0.1);
  addPaddyStrip(canalPts, canalHalfW, paddyChannelMat, 0.3, 0.17, paddyChannelLtMat);
  addPaddyStrip(trackPts, 2.6, paddyTrackMat, 0.14, 0.08, paddyRutMat);

  for (let i = 0; i < trackPts.length - 1; i++) {
    const d0 = trackPts[i][1] - paddyRiverZAt(riverPts, trackPts[i][0]);
    const d1 = trackPts[i + 1][1] - paddyRiverZAt(riverPts, trackPts[i + 1][0]);
    if (d0 === 0 || d0 * d1 < 0) {
      const f = Math.abs(d0) / (Math.abs(d0) + Math.abs(d1) || 1);
      const bx = trackPts[i][0] + (trackPts[i + 1][0] - trackPts[i][0]) * f;
      const bz = trackPts[i][1] + (trackPts[i + 1][1] - trackPts[i][1]) * f;
      const ry = -Math.atan2(trackPts[i + 1][1] - trackPts[i][1], trackPts[i + 1][0] - trackPts[i][0]);
      addPaddyBridge(bx, bz, ry, riverHalfW * 3.8 + 3);
      addBuffalo(bx + 14, bz + riverHalfW * 2.6, 0, rng() * Math.PI * 2);
      break;
    }
  }

  addFarmstead(farmX, farmZ, rng);

  // Bank trees: full blocky pixel-art trees close to the action,
  // matching the 2D canvas's tree-lined watercourses.
  for (let i = 1; i < riverPts.length - 1; i += 2) {
    const [tx, tz] = riverPts[i];
    const side = rng() < 0.5 ? 1 : -1;
    addTree(tx + (rng() - 0.5) * 10, tz + side * (riverHalfW * 2.2 + 4 + rng() * 5), 0.8 + rng() * 0.5);
  }
  for (let i = 2; i < canalPts.length - 2; i += 4) {
    const [tx, tz] = canalPts[i];
    addTree(tx + (rng() < 0.5 ? 1 : -1) * (canalHalfW * 2 + 3 + rng() * 4), tz, 0.7 + rng() * 0.4);
  }

  // Field life, placed on plots that fit the activity: farmers bent over
  // the transplant beds, egrets stalking the open water, a scarecrow in
  // the ripening grain, straw stacked on harvested stubble, one buffalo
  // wallowing chest-deep.
  const pick = (arr) => (arr.length ? arr[(rng() * arr.length) | 0] : null);
  for (let i = 0; i < 3; i++) {
    const p = pick(seedlingPlots);
    if (p) addFarmer(p[0] + (rng() - 0.5) * 8, p[1] + (rng() - 0.5) * 8, p[2], rng() * Math.PI * 2);
  }
  for (let i = 0; i < 3; i++) {
    const p = pick(waterPlots);
    if (p) addEgret(p[0] + (rng() - 0.5) * 10, p[1] + (rng() - 0.5) * 10, p[2], rng() * Math.PI * 2);
  }
  const sc = pick(ripePlots);
  if (sc) addScarecrow(sc[0], sc[1], sc[2] - 0.4, rng() * Math.PI * 2);
  for (let i = 0; i < 2; i++) {
    const p = pick(stubblePlots);
    if (p) {
      addHaystack(p[0] + (rng() - 0.5) * 6, p[1] + (rng() - 0.5) * 6, p[2], 0.9 + rng() * 0.4);
      addHaystack(p[0] + 3 + (rng() - 0.5) * 4, p[1] + 3 + (rng() - 0.5) * 4, p[2], 0.6 + rng() * 0.3);
    }
  }
  const wallow = pick(waterPlots);
  if (wallow) addBuffalo(wallow[0], wallow[1], wallow[2] - 0.5, rng() * Math.PI * 2);

  addPaddyBackdrop(rng, R);
}

// Suburban Corridor (Smart City variant 3) street furniture: a single
// wide boulevard gets sparse lamps/benches, a continuous green verge
// between the road and the building line, and a couple of surface
// parking lots -- deliberately much lighter overall density than the
// downtown variants' dense lamp/bench/hydrant/civic-plaza treatment in
// rebuildProps' main Smart City branch, which assumes a multi-loop
// downtown block grid this variant doesn't have (see _road_loops in
// snapshot.py, which gives this variant a single flattened loop).
function addSuburbanCorridorProps(rng, loops, obstacles) {
  const l = loops[0];
  if (!l) return;
  const curb = ROAD_HALF_W + SIDEWALK_W * 0.55;
  const perim = 2 * (2 * l.hw + 2 * l.hh);
  const nLamp = Math.max(8, Math.round(perim / 50));
  for (let i = 0; i < nLamp; i++) {
    for (const side of [1, -1]) {
      const p = loopEdgePoint(l, (i + (side === 1 ? 0.5 : 0)) / nLamp, side * curb);
      if (loopDist(l, p.x, p.z) < ROAD_HALF_W + 0.5) continue;
      addStreetlamp(p.x, p.z, p.heading + side * Math.PI / 2);
    }
  }
  const nBench = Math.max(4, Math.round(perim / 150));
  for (let i = 0; i < nBench; i++) {
    for (const side of [1, -1]) {
      const p = loopEdgePoint(l, (i + 0.3 + (side === 1 ? 0.5 : 0)) / nBench, side * (ROAD_HALF_W + SIDEWALK_W - 1.2));
      if (loopDist(l, p.x, p.z) < ROAD_HALF_W + 0.8) continue;
      addBench(p.x, p.z, p.heading + side * Math.PI / 2);
    }
  }
  // Continuous green verge between the road and the building/parking
  // line on both sides -- the "wide boulevard" cue, not a downtown kerb.
  const nVerge = Math.max(12, Math.round(perim / 34));
  for (let i = 0; i < nVerge; i++) {
    for (const side of [1, -1]) {
      const p = loopEdgePoint(l, (i + (side === 1 ? 0 : 0.5)) / nVerge, side * (ROAD_HALF_W + SIDEWALK_W + 6));
      addGrassPatch(p.x, p.z, (perim / nVerge) * 1.1, 10);
    }
  }
  const nTree = Math.max(10, Math.round(perim / 55));
  for (let i = 0; i < nTree; i++) {
    for (const side of [1, -1]) {
      const p = loopEdgePoint(l, (i + 0.22) / nTree, side * (ROAD_HALF_W + SIDEWALK_W + 13));
      addTree(p.x, p.z, 0.55 + rng() * 0.4);
    }
  }
  const avoid = (obstacles || [])
    .filter(o => o.kind !== 'circle')
    .map(o => toScene((o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2));
  const nLot = 4;
  for (let i = 0; i < nLot; i++) {
    const side = i % 2 === 0 ? 1 : -1;
    const p = loopEdgePoint(l, i / nLot + 0.12, side * (ROAD_HALF_W + SIDEWALK_W + 30));
    if (Math.hypot(p.x, p.z) < 55) continue;
    if (avoid.some(a => Math.hypot(p.x - a.x, p.z - a.z) < 65)) continue;
    addParkingLot(p.x, p.z, p.heading + side * Math.PI / 2, 5 + ((rng() * 3) | 0), rng);
  }
}

let propsBuiltFor = null;
export function rebuildProps(env, obstacles, roadLoops, nodeExtentR = 0, variant = 1) {
  // Bucketed to the nearest 50m so a topology rebuild that changes the
  // node scatter extent (and therefore siteRadius) re-scatters props,
  // without rebuilding every single poll over sub-metre float jitter.
  // nodeExtentR only actually affects the key for Paddy Field (see
  // paddyExtentSig) -- every other environment already derives its own
  // scale from real obstacle rects or road loops, so folding it in
  // unconditionally would just add unrelated rebuild churn whenever node
  // positions drift for those. variant is folded in unconditionally --
  // Industrial Site and Smart City's prop scatter is now variant-aware
  // (see below), and there's no cheap way to tell from here whether a
  // variant switch also happened to change the rounded siteRadius bucket.
  const key = env + '|' + variant + '|' + Math.round(siteRadius(obstacles) / 50)
    + cityLoopSig(env, roadLoops) + paddyExtentSig(env, nodeExtentR);
  if (propsBuiltFor === key) return;
  propsBuiltFor = key;
  paddyFogBand = null; // reinstated below iff this rebuild is Paddy Field
  for (const child of [...propsGroup.children]) {
    propsGroup.remove(child);
    child.traverse?.(o => o.geometry?.dispose?.());
  }
  if (env === 'Open Area') {
    const rng = mulberry32(0x1234);
    // Ceiling raised alongside siteRadius()'s (Open Area's own AP-range
    // budget -- topology_canvas.py's _ENV_PATH_LOSS_EXP -- now nominally
    // reaches ~2.3km) so trees still surround nodes near the edge of a
    // large scatter instead of thinning out to bare ground past ~1.4km.
    for (let i = 0; i < 30; i++) {
      const ang = rng() * Math.PI * 2, r = 260 + rng() * 2600;
      addTree(Math.cos(ang) * r, Math.sin(ang) * r, 1.0 + rng() * 0.8);
    }
  } else if (env === 'Paddy Field') {
    // siteRadius(obstacles) alone stays pinned at its own 180m floor here
    // -- Paddy Field has no RF obstacles to size itself off of -- so the
    // node-derived extent (same 80th-percentile distance app.js's camera
    // framing already uses) fills in the actual scale of a spread-out
    // network. Capped the same way siteRadius caps itself, so an extreme
    // scatter doesn't balloon the plot count.
    const R = Math.min(3600, Math.max(siteRadius(obstacles), nodeExtentR));
    // Morning-mist band scaled to the real field: far plots soften, the
    // mountain backdrop fades toward the sky (the 2D canvas fakes the
    // same aerial perspective with its _MTN_HAZE wash), and the horizon
    // dissolves before the textured ground plane visibly ends.
    paddyFogBand = [R * 1.6, R * 4.0];
    addPaddyField(mulberry32(0x9a1dd7), R);
  } else if (env === 'Industrial Site') {
    // Yard-prop scatter differs by variant now too -- fillerLayoutFor
    // (rebuildBuildings) already varies the FILLER BUILDINGS, but the
    // real "site is full of stuff" read comes from this dense prop
    // scatter, which used to be one identical pile of containers/
    // barrels/trucks/floodlights regardless of variant. Distinct seed per
    // variant so switching layouts doesn't just relabel the same
    // relative pile positions.
    const rng = mulberry32(hashSeed('industrial-props|' + variant));
    const R = siteRadius(obstacles);
    const scale = Math.min(3.4, Math.max(0.6, R / 500));
    const cnt = (base) => Math.max(4, Math.round(base * Math.min(3, scale)));
    const loops = industrialRoadLoops(obstacles, variant);

    if (variant === 3) {
      // Business Park: light-industry campus -- lawns, trees, garden
      // beds and real parking lots instead of yard clutter. No fence, no
      // floodlights, no containers/barrels/tanks/heavy trucks -- this is
      // the one variant that should visibly read as lower-industrial-
      // intensity than the other two.
      for (let i = 0; i < cnt(9); i++) {
        const [x, z] = scatterPos(rng, R * 0.1, R * 0.85, loops, 24);
        const gw = 28 + rng() * 22, gd = 22 + rng() * 18;
        addGrassPatch(x, z, gw, gd);
        const nTree = 2 + ((rng() * 3) | 0);
        for (let k = 0; k < nTree; k++) {
          const a = rng() * Math.PI * 2, rr = rng() * Math.min(gw, gd) * 0.4;
          addTree(x + Math.cos(a) * rr, z + Math.sin(a) * rr, 0.55 + rng() * 0.5);
        }
        if (rng() < 0.6) addGardenBed(x + (rng() - 0.5) * gw * 0.4, z + (rng() - 0.5) * gd * 0.4, rng() * Math.PI * 2, rng);
        if (rng() < 0.5) addBench(x + (rng() - 0.5) * gw * 0.4, z + (rng() - 0.5) * gd * 0.4, rng() * Math.PI * 2);
      }
      const nLots = scale > 1 ? 4 : 3;
      for (let i = 0; i < nLots; i++) {
        const [x, z] = scatterPos(rng, R * 0.18, R * 0.82, loops, 20);
        addParkingLot(x, z, rng() * Math.PI * 2, 7 + ((rng() * 5) | 0), rng);
      }
      for (let i = 0; i < cnt(10); i++) {
        const [x, z] = scatterPos(rng, R * 0.1, R * 0.9, loops, 5);
        addCar(x, z, rng() * Math.PI * 2, rng);
      }
    } else if (variant === 2) {
      // Process Plant: heavy industry -- hazard-striped pads, a scatter
      // of standalone tank clusters linked by pipe-racks, and far more
      // barrels than v1, but only a token amount of parking/light
      // traffic -- this is a working plant, not a logistics yard. (The
      // real process-unit/tank-farm obstacles already get their own
      // chimneys/tanks/apron via addBuilding/addTankFarm; this pass adds
      // the surrounding yard character.)
      addFencePerimeter(R * 1.05, R * 1.05);
      const nFlood = cnt(9);
      for (let i = 0; i < nFlood; i++) {
        const ang = (i / nFlood) * Math.PI * 2 + rng() * 0.3;
        let r = R * 0.85 + rng() * R * 0.1;
        if (!clearOfRoads(Math.cos(ang) * r, Math.sin(ang) * r, loops, 2)) r = R;
        addFloodlight(Math.cos(ang) * r, Math.sin(ang) * r, rng);
      }
      for (let i = 0; i < cnt(46); i++) {
        const [x, z] = scatterPos(rng, R * 0.08, R * 0.9, loops, 4);
        addBarrelCluster(x, z, rng);
      }
      for (let i = 0; i < cnt(6); i++) {
        const [x, z] = scatterPos(rng, R * 0.14, R * 0.55, loops, 14);
        const w = 30 + rng() * 22, d = 22 + rng() * 14;
        addHazardStripe(x, z, w, d, rng() * Math.PI * 2);
      }
      const tankClusters = [];
      for (let i = 0; i < cnt(6); i++) {
        const [x, z] = scatterPos(rng, R * 0.15, R * 0.75, loops, 12);
        addTank(propsGroup, x, z, 3 + rng() * 2, 6 + rng() * 5, rng);
        tankClusters.push([x, z]);
      }
      // Pipe-racks strung between successive standalone tank clusters --
      // reads as a real linked plant instead of loose scattered props.
      for (let i = 0; i + 1 < tankClusters.length; i++) {
        if (rng() < 0.7) addPipeRack([tankClusters[i], tankClusters[i + 1]], 3.0 + rng() * 1.0);
      }
      for (let i = 0; i < cnt(9); i++) {
        const [x, z] = scatterPos(rng, R * 0.1, R * 0.9, loops, 9);
        addContainerStack(x, z, rng);
      }
      for (let i = 0; i < cnt(3); i++) {
        const [x, z] = scatterPos(rng, R * 0.12, R * 0.85, loops, 7);
        addTruck(x, z, rng() * Math.PI * 2, rng);
      }
      for (let i = 0; i < cnt(4); i++) {
        const [x, z] = scatterPos(rng, R * 0.15, R * 0.85, loops, 10);
        addSemi(x, z, rng() * Math.PI * 2, rng);
      }
      const [px, pz] = scatterPos(rng, R * 0.2, R * 0.6, loops, 20);
      addParkingLot(px, pz, rng() * Math.PI * 2, 4, rng);
    } else {
      // Logistics Park (variant 1, and the default): warehouses, a real
      // loading-dock scatter of containers/barrels, a staffed parking
      // lot, perimeter fence + floodlights -- the original mixed yard.
      addFencePerimeter(R * 1.05, R * 1.05);
      for (let i = 0; i < cnt(22); i++) {
        const pileR = 12 + rng() * 16;
        const [x, z] = scatterPos(rng, R * 0.15, R * 0.9, loops, pileR * 0.7);
        addStockpile(x, z, pileR, rng);
      }
      const nFlood = cnt(14);
      for (let i = 0; i < nFlood; i++) {
        const ang = (i / nFlood) * Math.PI * 2 + rng() * 0.3;
        let r = R * 0.85 + rng() * R * 0.1;
        // Near the outer loop's corners the perimeter ring can graze the
        // pavement -- push those masts out toward the fence instead.
        if (!clearOfRoads(Math.cos(ang) * r, Math.sin(ang) * r, loops, 2)) r = R;
        addFloodlight(Math.cos(ang) * r, Math.sin(ang) * r, rng);
      }
      for (let i = 0; i < cnt(6); i++) {
        const [x, z] = scatterPos(rng, R * 0.15, R * 0.75, loops, 3);
        addFloodlight(x, z, rng);
      }
      for (let i = 0; i < cnt(38); i++) {
        const [x, z] = scatterPos(rng, R * 0.1, R * 0.9, loops, 9);
        addContainerStack(x, z, rng);
      }
      for (let i = 0; i < cnt(26); i++) {
        const [x, z] = scatterPos(rng, R * 0.08, R * 0.9, loops, 4);
        addBarrelCluster(x, z, rng);
      }
      for (let i = 0; i < cnt(13); i++) {
        const [x, z] = scatterPos(rng, R * 0.12, R * 0.85, loops, 7);
        addTruck(x, z, rng() * Math.PI * 2, rng);
      }
      for (let i = 0; i < cnt(9); i++) {
        const [x, z] = scatterPos(rng, R * 0.15, R * 0.85, loops, 10);
        addSemi(x, z, rng() * Math.PI * 2, rng);
      }
      for (let i = 0; i < cnt(16); i++) {
        const [x, z] = scatterPos(rng, R * 0.1, R * 0.9, loops, 5);
        addCar(x, z, rng() * Math.PI * 2, rng);
      }
      const nLots = scale > 1 ? 3 : 2;
      for (let i = 0; i < nLots; i++) {
        const [x, z] = scatterPos(rng, R * 0.18, R * 0.7, loops, 20);
        addParkingLot(x, z, rng() * Math.PI * 2, 7 + ((rng() * 5) | 0), rng);
      }
      for (let i = 0; i < 2; i++) {
        const [x, z] = scatterPos(rng, R * 0.2, R * 0.8, loops, 16);
        addSemiRow(x, z, rng() * Math.PI * 2, 3 + ((rng() * 3) | 0), rng);
      }
    }
  } else if (env === 'Smart City' && (roadLoops || []).length) {
    const loops = roadLoops;
    const rng = mulberry32(hashSeed('city-props|' + variant));
    const R = cityExtentR(obstacles, loops);
    if (variant === 3) {
      // Suburban Corridor: a single wide boulevard, not a downtown grid --
      // dedicated sparse street furniture + green-verge treatment instead
      // of the dense lamps/benches/civic-plaza pass below, which assumes
      // a downtown block grid with an inner "plaza" loop. rebuildProps
      // returns here; this is the last branch in the env if/else chain.
      addSuburbanCorridorProps(rng, loops, obstacles);
      return;
    }
    // Business District (variant 2) reads as denser/busier street
    // furniture and skips the leisure civic plaza entirely (see below) --
    // "every block holds a tower, no parks" per its 2D design intent.
    const dense = variant === 2;
    const cnt = (base) => Math.max(3, Math.round(base * Math.min(3, Math.max(0.7, R / 400)) * (dense ? 1.35 : 1.0)));
    const inner = loops.reduce((a, b) => (a.hw * a.hh <= b.hw * b.hh ? a : b));
    const outer = loops.reduce((a, b) => (a.hw * a.hh >= b.hw * b.hh ? a : b));
    const ccx = inner.cx, ccz = -inner.cy;
    const curb = ROAD_HALF_W + SIDEWALK_W * 0.55;

    for (const l of loops) {
      const perim = 2 * (2 * l.hw + 2 * l.hh);
      // side=+1 is the loop's inner sidewalk, -1 the outer; heading +
      // side*PI/2 turns lamp arms and bench fronts toward the pavement
      // from either side. Inner-side offsets cut loop corners, which
      // would drop props onto the crossing pavement -- the loopDist
      // guards skip those few near-corner slots instead.
      for (const side of [1, -1]) {
        const nLamp = Math.max(10, Math.round(perim / 20));
        for (let i = 0; i < nLamp; i++) {
          const p = loopEdgePoint(l, (i + (side === 1 ? 0.5 : 0)) / nLamp, side * curb);
          if (loopDist(l, p.x, p.z) < ROAD_HALF_W + 0.5) continue;
          addStreetlamp(p.x, p.z, p.heading + side * Math.PI / 2);
        }
        const nBench = Math.max(6, Math.round(perim / 42));
        for (let i = 0; i < nBench; i++) {
          const p = loopEdgePoint(l, (i + 0.27 + (side === 1 ? 0.5 : 0)) / nBench, side * (ROAD_HALF_W + SIDEWALK_W - 1.2));
          if (loopDist(l, p.x, p.z) < ROAD_HALF_W + 0.8) continue;
          addBench(p.x, p.z, p.heading + side * Math.PI / 2);
          if (rng() < 0.7) addTrashCan(p.x + Math.cos(p.heading) * 2.4, p.z - Math.sin(p.heading) * 2.4);
        }
        const nHyd = Math.max(4, Math.round(perim / 60));
        for (let i = 0; i < nHyd; i++) {
          const p = loopEdgePoint(l, (i + 0.62) / nHyd, side * (ROAD_HALF_W + SIDEWALK_W * 0.3));
          if (loopDist(l, p.x, p.z) < ROAD_HALF_W + 0.4) continue;
          addHydrant(p.x, p.z);
        }
        // Kerbside parking hugs the pavement edge, well clear of the
        // centreline the live server-driven cars drive. A truck takes two
        // stalls, so the slot after one is always forced empty; no
        // parking within a car length of an intersection.
        const nPark = Math.max(8, Math.round(perim / 8.4));
        let skip = false;
        for (let i = 0; i < nPark; i++) {
          if (skip) { skip = false; continue; }
          if (rng() < 0.4) continue;
          const p = loopEdgePoint(l, (i + (side === 1 ? 0.4 : 0)) / nPark, side * (ROAD_HALF_W - 2.3));
          if (loopDist(l, p.x, p.z) < 6.2) continue;
          const ry = p.heading + (side === 1 ? 0 : Math.PI) + (rng() - 0.5) * 0.05;
          if (rng() < 0.12) { addTruck(p.x, p.z, ry, rng); skip = true; }
          else addCar(p.x, p.z, ry, rng);
        }
      }
      for (const sx of [-1, 1]) {
        for (const sy of [-1, 1]) {
          const o = ROAD_HALF_W + 2;
          const p = toScene(l.cx + sx * (l.hw + o), l.cy + sy * (l.hh + o));
          addTrafficLight(p.x, p.z, Math.atan2(-sy, -sx));
        }
      }
    }

    const nShel = 6;
    for (let i = 0; i < nShel; i++) {
      const p = loopEdgePoint(outer, (i + 0.32) / nShel, -(ROAD_HALF_W + SIDEWALK_W * 0.6));
      addBusShelter(p.x, p.z, p.heading - Math.PI / 2);
    }
    const nTree = Math.max(20, Math.round((2 * (2 * outer.hw + 2 * outer.hh)) / 14));
    for (let i = 0; i < nTree; i++) {
      const p = loopEdgePoint(outer, (i + 0.15) / nTree, -(ROAD_HALF_W + SIDEWALK_W + 4));
      addTree(p.x, p.z, 0.7 + rng() * 0.45);
      if (rng() < 0.55) {
        addGardenBed(p.x + Math.cos(p.heading) * 3.2, p.z - Math.sin(p.heading) * 3.2, p.heading, rng);
      }
    }
    // The inner loop only ever got lamps/benches/hydrants above -- its
    // plaza-facing sidewalk had zero tree-lining. Ring it too so the civic
    // square reads green right up to its own kerb, not just past it.
    if (inner !== outer) {
      const nInnerTree = Math.max(12, Math.round((2 * (2 * inner.hw + 2 * inner.hh)) / 16));
      for (let i = 0; i < nInnerTree; i++) {
        const p = loopEdgePoint(inner, (i + 0.2) / nInnerTree, ROAD_HALF_W + SIDEWALK_W + 3);
        if (loopDist(inner, p.x, p.z) < ROAD_HALF_W + 0.5) continue;
        addTree(p.x, p.z, 0.6 + rng() * 0.4);
      }
    }

    // Civic plaza inside the inner loop, offset so the AP mast at the
    // map centre isn't left standing in the fountain. Skipped when a
    // tight node cluster shrinks the loop below plaza size.
    if (!dense && inner.hw > 42 && inner.hh > 42) {
      const px = ccx + Math.min(20, inner.hw * 0.3), pz = ccz - Math.min(16, inner.hh * 0.25);
      addFountain(px, pz);
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * Math.PI * 2;
        addBench(px + Math.cos(a) * 13, pz + Math.sin(a) * 13, Math.PI - a);
        if (i % 2 === 0) addTrashCan(px + Math.cos(a + 0.28) * 13, pz + Math.sin(a + 0.28) * 13);
      }
      for (let i = 0; i < 18; i++) {
        const a = (i / 18) * Math.PI * 2 + 0.3;
        const rr = 21 + (i % 2) * 5;
        addTree(px + Math.cos(a) * rr, pz + Math.sin(a) * rr, 0.75 + rng() * 0.4);
      }
      for (let i = 0; i < 4; i++) {
        const a = (i / 4) * Math.PI * 2 + 0.8;
        addStreetlamp(px + Math.cos(a) * 17, pz + Math.sin(a) * 17, Math.PI - a);
      }

      // Three more park zones fill the rest of the plaza -- a football
      // pitch, a pond with a dock, and a playground/garden corner -- each
      // offset into its own quadrant so nothing overlaps the fountain or
      // each other. Sizes scale off the available radius so a modest
      // plaza still gets a (smaller) version of all three rather than
      // just skipping them; clearOfRoads gates each zone out entirely
      // if the plaza is too tight for even that reduced size to fit
      // without spilling onto the ring road.
      const zoneR = Math.min(inner.hw, inner.hh) * 0.48;

      const fx = ccx - zoneR, fz = ccz + zoneR * 0.7;
      const fieldHalfW = Math.min(52, zoneR * 0.95), fieldHalfD = Math.min(34, zoneR * 0.68);
      if (fieldHalfW > 14 && fieldHalfD > 9
        && clearOfRoads(fx, fz, loops, Math.hypot(fieldHalfW, fieldHalfD) + 6)) {
        addFootballField(fx, fz, rng() < 0.5 ? 0 : Math.PI / 2, fieldHalfW, fieldHalfD);
        for (let i = 0; i < 6; i++) {
          const a = (i / 6) * Math.PI * 2;
          const rr = Math.hypot(fieldHalfW, fieldHalfD) + 7;
          addBench(fx + Math.cos(a) * rr, fz + Math.sin(a) * rr, Math.PI - a);
        }
      }

      const wx = ccx - zoneR, wz = ccz - zoneR * 0.7;
      const pondR = Math.min(20, zoneR * 0.58);
      if (pondR > 7 && clearOfRoads(wx, wz, loops, pondR + 10)) {
        addPond(wx, wz, pondR);
        for (let i = 0; i < 6; i++) {
          const a = (i / 6) * Math.PI * 2 + 0.4;
          addBench(wx + Math.cos(a) * (pondR + 4), wz + Math.sin(a) * (pondR + 4), Math.PI - a);
        }
        for (let i = 0; i < 14; i++) {
          const a = (i / 14) * Math.PI * 2;
          const rr = pondR + 8 + (i % 2) * 4;
          addTree(wx + Math.cos(a) * rr, wz + Math.sin(a) * rr, 0.7 + rng() * 0.4);
        }
        for (let i = 0; i < 3; i++) {
          const a = (i / 3) * Math.PI * 2 + 0.9;
          addStreetlamp(wx + Math.cos(a) * (pondR + 12), wz + Math.sin(a) * (pondR + 12), Math.PI - a);
        }
      }

      const gx = ccx + zoneR * 0.9, gz = ccz + zoneR * 0.9;
      if (clearOfRoads(gx, gz, loops, 20)) {
        addPlayground(gx, gz, rng() * Math.PI * 2);
        addGardenBed(gx + 11, gz - 3, rng() * Math.PI * 2, rng);
        addGardenBed(gx - 10, gz + 4, rng() * Math.PI * 2, rng);
        addGardenBed(gx - 3, gz - 11, rng() * Math.PI * 2, rng);
        addGardenBed(gx + 9, gz + 9, rng() * Math.PI * 2, rng);
        addGazebo(gx + 2, gz + 11);
        for (let i = 0; i < 9; i++) {
          const a = (i / 9) * Math.PI * 2;
          const rr = 15 + (i % 3) * 4;
          addTree(gx + Math.cos(a) * rr, gz + Math.sin(a) * rr, 0.7 + rng() * 0.4);
        }
        for (let i = 0; i < 6; i++) {
          const a = rng() * Math.PI * 2;
          addBench(gx + Math.cos(a) * 9, gz + Math.sin(a) * 9, rng() * Math.PI * 2);
        }
      }

      // The four zones above only claim a fraction of a big inner loop --
      // blanket-fill whatever plaza floor is left with grass-and-tree
      // clumps so none of it reads as bare tarmac. Grid-scattered rather
      // than randomly scattered so coverage stays even out to the plaza's
      // own kerb, with each cell skipped near an existing zone (so it
      // doesn't stack a lawn on top of the fountain/pitch/pond/playground)
      // and occasionally left bare so it doesn't look like a solid green
      // slab either.
      const claimed = [
        [px, pz, 26], [fx, fz, Math.hypot(fieldHalfW, fieldHalfD) + 8],
        [wx, wz, pondR + 14], [gx, gz, 22],
      ];
      const fillCell = Math.max(16, Math.min(inner.hw, inner.hh) / 10);
      const fillMarginX = inner.hw - (ROAD_HALF_W + SIDEWALK_W + 6);
      const fillMarginZ = inner.hh - (ROAD_HALF_W + SIDEWALK_W + 6);
      const fillHalfX = Math.ceil(fillMarginX / fillCell), fillHalfZ = Math.ceil(fillMarginZ / fillCell);
      for (let fgx = -fillHalfX; fillMarginX > 0 && fgx <= fillHalfX; fgx++) {
        for (let fgz = -fillHalfZ; fillMarginZ > 0 && fgz <= fillHalfZ; fgz++) {
          const x = ccx + fgx * fillCell + (rng() - 0.5) * fillCell * 0.4;
          const z = ccz + fgz * fillCell + (rng() - 0.5) * fillCell * 0.4;
          if (Math.abs(x - ccx) > fillMarginX || Math.abs(z - ccz) > fillMarginZ) continue;
          if (claimed.some(([cx, cz, r]) => Math.hypot(x - cx, z - cz) < r)) continue;
          if (rng() < 0.12) continue;
          const gw = fillCell * (0.75 + rng() * 0.2), gd = fillCell * (0.75 + rng() * 0.2);
          addGrassPatch(x, z, gw, gd);
          const nTree = 2 + ((rng() * 4) | 0);
          for (let i = 0; i < nTree; i++) {
            const a = rng() * Math.PI * 2, rr = rng() * Math.min(gw, gd) * 0.42;
            addTree(x + Math.cos(a) * rr, z + Math.sin(a) * rr, 0.55 + rng() * 0.5);
          }
          if (rng() < 0.35) addGardenBed(x, z, rng() * Math.PI * 2, rng);
          if (rng() < 0.3) addBench(x + (rng() - 0.5) * gw * 0.4, z + (rng() - 0.5) * gd * 0.4, rng() * Math.PI * 2);
        }
      }
    }

    // Pocket parks, kiosks, lots and the odd delivery truck fill the
    // blocks between the filler towers.
    for (let i = 0; i < cnt(15); i++) {
      const [x, z] = scatterPos(rng, R * 0.2, R * 0.9, loops, 14, ccx, ccz);
      const nT = 5 + ((rng() * 5) | 0);
      for (let k = 0; k < nT; k++) {
        const a = rng() * Math.PI * 2, rr = 3 + rng() * 9;
        addTree(x + Math.cos(a) * rr, z + Math.sin(a) * rr, 0.7 + rng() * 0.5);
      }
      for (let k = 0; k < 1 + ((rng() * 2) | 0); k++) {
        const a = rng() * Math.PI * 2;
        addBench(x + Math.cos(a) * 6, z + Math.sin(a) * 6, rng() * Math.PI * 2);
      }
      if (rng() < 0.7) addTrashCan(x + 2, z - 2);
    }
    for (let i = 0; i < cnt(5); i++) {
      const [x, z] = scatterPos(rng, R * 0.15, R * 0.7, loops, 4, ccx, ccz);
      addKiosk(x, z, rng() * Math.PI * 2, rng);
    }
    for (let i = 0; i < cnt(3); i++) {
      const [x, z] = scatterPos(rng, R * 0.3, R * 0.8, loops, 20, ccx, ccz);
      addParkingLot(x, z, rng() * Math.PI * 2, 6 + ((rng() * 4) | 0), rng);
    }
    for (let i = 0; i < cnt(5); i++) {
      const [x, z] = scatterPos(rng, R * 0.25, R * 0.85, loops, 8, ccx, ccz);
      addTruck(x, z, rng() * Math.PI * 2, rng);
    }
  }
}

// ---- buildings: stacked flat-shaded blocks, chunky low-poly roof/tank/
// chimney furniture -- built from sim.obstacles (the same rects PhyLayer
// uses for real path-loss, so what you see is what actually blocks RF) --
const beaconGeo = new THREE.BoxGeometry(1.6, 1.6, 1.6);

function addBlockBox(parent, w, h, d, wallTex, roofTex, centerY) {
  const geo = new THREE.BoxGeometry(Math.max(1, w), Math.max(1, h), Math.max(1, d));
  const tileM = 4;
  const mkWallMat = (span) => new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, 
    map: tiledClone(wallTex, Math.max(1, Math.round(span / tileM)), Math.max(1, Math.round(h / tileM))),
  });
  const side = mkWallMat(d), front = mkWallMat(w);
  const roofMat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05, 
    map: tiledClone(roofTex, Math.max(1, Math.round(w / tileM)), Math.max(1, Math.round(d / tileM))),
  });
  const mesh = new THREE.Mesh(geo, [side, side, roofMat, roofMat, front, front]);
  mesh.position.y = centerY;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

// Rooftop clutter for a FLAT roof only (a pitched/gabled or setback-
// tower roof has no single flat plane to scatter this on) -- AC units,
// vent pipes, a small water tank, scattered per-building via the same
// seeded rng addBuilding/addFillerBuilding already derive from the
// building's own label, so it's stable across rebuilds like everything
// else here. A flat-topped box on its own reads as an empty rooftop no
// real building has; this is the single cheapest way to break that up.
const roofAcMat = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.4, color: 0xc9ccd1 });
const roofVentMat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.5, color: 0x6b6f76 });
function addRoofClutter(parent, w, d, roofY, rng) {
  if (rng() >= 0.7) return; // not every rooftop -- an empty one now and then reads as real variety, not a bug
  const marginX = Math.min(w, d) * 0.16 + 1.6;
  const halfW = Math.max(0.1, w / 2 - marginX), halfD = Math.max(0.1, d / 2 - marginX);
  const n = 1 + Math.floor(rng() * 3);
  for (let i = 0; i < n; i++) {
    const x = (rng() * 2 - 1) * halfW, z = (rng() * 2 - 1) * halfD;
    const kind = rng();
    if (kind < 0.55) {
      const s = 1.2 + rng() * 1.0;
      const box = new THREE.Mesh(new THREE.BoxGeometry(s * 1.4, s * 0.8, s * 1.1), roofAcMat);
      box.position.set(x, roofY + s * 0.4, z);
      box.castShadow = true;
      parent.add(box);
    } else if (kind < 0.82) {
      const r = 0.35 + rng() * 0.25, h = 1.5 + rng() * 1.5;
      const pipe = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 6), roofVentMat);
      pipe.position.set(x, roofY + h / 2, z);
      pipe.castShadow = true;
      parent.add(pipe);
    } else {
      const r = 0.9 + rng() * 0.6, h = 1.8 + rng() * 1.2;
      const tank = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 8), roofAcMat);
      tank.position.set(x, roofY + h / 2, z);
      tank.castShadow = true;
      parent.add(tank);
    }
  }
}

function addRock(parent, o, idx) {
  const rng = mulberry32(hashSeed(String(o.label || idx)));
  const n = Math.max(4, Math.round(o.r / 3.5));
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0, map: tiledClone(TEX.stone, 1, 1) });
  for (let i = 0; i < n; i++) {
    const ang = rng() * Math.PI * 2, rr = rng() * o.r;
    const s = o.r * (0.28 + rng() * 0.24);
    const b = new THREE.Mesh(new THREE.BoxGeometry(s, s * (0.7 + rng() * 0.5), s), mat);
    b.position.set(Math.cos(ang) * rr, s * 0.4, Math.sin(ang) * rr);
    b.rotation.y = rng() * Math.PI;
    b.castShadow = true; b.receiveShadow = true;
    g.add(b);
  }
  const c = toScene(o.cx, o.cy);
  g.position.set(c.x, 0, c.z);
  parent.add(g);
}

function addTank(parent, x, z, r, h, rng) {
  const wallMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.6, map: tiledClone(TEX.metal, 1, 1) });
  const tank = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 8), wallMat);
  tank.position.set(x, h / 2, z);
  tank.castShadow = true; tank.receiveShadow = true;
  parent.add(tank);
  const capMat = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0, color: 0xb8bcc0 });
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(r, r * 0.94, 1.4, 8), capMat);
  cap.position.set(x, h + 0.7, z);
  cap.castShadow = true;
  parent.add(cap);
}

function addTankFarm(parent, o) {
  const rng = mulberry32(hashSeed(String(o.label)));
  // The obstacle's own footprint is sized as a fraction of the current
  // node scatter extent in the 2D scene (see _bg_industrial_v2 in
  // topology_canvas.py), so it can be anywhere from ~150m to ~900m+ wide
  // -- for a compact "vertical world" site, the VISUAL tank farm gets its
  // own small footprint (seeded off the same label, so it's stable) at
  // the obstacle's true centre, independent of how large the real
  // obstacle rect is. The real rect still does the actual RF-obstruction
  // math in sim.obstacles; only what gets drawn here is resized.
  const w = 28 + rng() * 24, d = 20 + rng() * 16;
  const cols = Math.min(5, Math.max(2, Math.round(w / 12)));
  const rows = Math.min(3, Math.max(1, Math.round(d / 12)));
  const cw = w / cols, cd = d / rows;
  const group = new THREE.Group();
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const tr = Math.min(cw, cd) * (0.3 + rng() * 0.1);
      const th = 8 + rng() * 6;
      addTank(group, -w / 2 + cw * (c + 0.5), -d / 2 + cd * (r + 0.5), tr, th, rng);
    }
  }
  const c = toScene((o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2);
  group.position.set(c.x, 0, c.z);
  parent.add(group);
}

// ---- Military Zone: bunkers/checkpoints built from the real obstacles
// (the exact rects that do the RF-obstruction math in sim.obstacles --
// visual_obstacles here, see snapshot.py, so the command bunker still
// renders even on the run where it got dropped from PHY for enclosing
// the AP), plus a full perimeter, corner watchtowers, a flagpole, a
// mortar pit, a helipad, and a patrolling tank + infantry squad -- the
// three.js counterpart to topology_canvas.py's single consolidated
// _bg_military_v1 layout. Plain-colour low-poly materials (no baked
// sandbag/camo texture assets exist yet), matched to that view's own
// hex palette so it reads as the same base.
const MIL_BUNKER_MAT = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, color: 0x8a8570 });
const MIL_BUNKER_DK_MAT = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, color: 0x6f6b5a });
const MIL_SANDBAG_MAT = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, color: 0xa49a72 });
const MIL_DOOR_MAT = new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0.15, color: 0x2b2a22 });
const MIL_NET_MAT = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0x4c5a3a, transparent: true, opacity: 0.88, side: THREE.DoubleSide });
const MIL_NET_MAT2 = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0x6f7a4a, transparent: true, opacity: 0.88, side: THREE.DoubleSide });
const MIL_TOWER_MAT = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.65, color: 0x4c4a3e });
const MIL_TANK_HULL_MATS = [0x586347, 0x4f5a3f].map(c => new THREE.MeshStandardMaterial({ roughness: 0.65, metalness: 0.45, color: c }));
const MIL_TANK_DK_MAT = new THREE.MeshStandardMaterial({ roughness: 0.65, metalness: 0.45, color: 0x454e37 });
const MIL_TANK_TURRET_MAT = new THREE.MeshStandardMaterial({ roughness: 0.65, metalness: 0.45, color: 0x657154 });
// Shared faction-livery palette for every mobile military unit (soldiers,
// planes, trucks, helicopters) -- muted/desaturated ("military type dark
// color") but varied per instance rather than one flat colour for the
// whole army, same idea as MIL_TANK_HULL_MATS' 2-way hull variation just
// extended to 6 tones and reused everywhere so a soldier and a nearby
// truck can plausibly read as the same faction. `main` drives primary
// panels/uniform, `dark` is its coordinated shadow/trim tone (wheels,
// glass, skin, weapons, rotor blades stay their own fixed realistic
// colour regardless of which livery a unit rolls -- only the "paint"
// varies). Picked with rng() so it's deterministic per simulation seed,
// not Math.random().
const MIL_UNIT_PALETTE = [
  { main: 0x4a5a3a, dark: 0x2e3a24 }, // olive drab
  { main: 0x6b5a3a, dark: 0x453a26 }, // desert tan / khaki
  { main: 0x455560, dark: 0x2a343c }, // slate / gunmetal blue-gray
  { main: 0x3d4a58, dark: 0x252e38 }, // navy-gray
  { main: 0x6b4335, dark: 0x422a22 }, // brick / maroon-brown
  { main: 0x484848, dark: 0x2c2c2c }, // charcoal / urban gray
];
// Vehicle builders need BOTH tones of a livery kept in sync (e.g. a
// plane's wings should be the *matching* dark trim for whichever main
// colour its fuselage got, not an independently-rolled one) -- so vehicle
// callers pick one {main, dark} MATERIAL pair per instance from an array
// built with this, same array-of-pre-built-materials-indexed-by-rng
// convention as MIL_TANK_HULL_MATS, just paired instead of flat.
function buildLiveryMatPairs() {
  return MIL_UNIT_PALETTE.map(c => ({
    main: new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0, color: c.main }),
    dark: new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0, color: c.dark }),
  }));
}
function pickLivery(mats, rng) {
  return mats[(rng() * mats.length) | 0];
}
const MIL_SOLDIER_UNIFORM_MATS = MIL_UNIT_PALETTE.map(
  c => new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0, color: c.main }));
const MIL_SOLDIER_SKIN_MAT = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0, color: 0xc69a72 });
const MIL_HELMET_MAT = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.4, color: 0x3d4530 });
const MIL_FLAG_MAT = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0, color: 0xa4392f, side: THREE.DoubleSide });
const MIL_WIRE_MAT = new THREE.LineBasicMaterial({ color: 0x9a9587 });
const MIL_POST_MAT = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0.05, color: 0x554f3d });
const MIL_PAD_MAT = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0x7a715a });
const MIL_PAD_MARK_MAT = new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0, color: 0xe7e2cf });

function addWatchtower(parent, x, z, h) {
  const g = new THREE.Group();
  // Every dimension below scales off `h` (relative to the original
  // baseline height of 13) instead of using fixed numbers -- a "big
  // tower" request means a genuinely bigger structure, not just a
  // taller, spindlier version of the same thin mast: base width, deck,
  // roof and rails all need to grow with it or a 60m tower looks like a
  // toothpick.
  const s = h / 13;
  const legOffset = 2.1 * s;
  const legPos = [[-legOffset, -legOffset], [legOffset, -legOffset],
                  [-legOffset, legOffset], [legOffset, legOffset]];
  const legGeo = new THREE.CylinderGeometry(0.5 * s, 0.5 * s, h, 5);
  for (const [dx, dz] of legPos) {
    const leg = new THREE.Mesh(legGeo, MIL_TOWER_MAT);
    leg.position.set(dx, h / 2, dz);
    leg.castShadow = true;
    g.add(leg);
  }
  // Diagonal cross-bracing between adjacent legs -- reads as an actual
  // timber tower frame instead of four bare poles. One level (not two --
  // with dozens of these towers across the map, halving the brace count
  // is a meaningful chunk of the scene's total draw calls for a barely
  // noticeable visual change).
  function brace(a, b, yFrac) {
    const dx = b[0] - a[0], dz = b[1] - a[1];
    const len = Math.hypot(dx, dz);
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.1 * s, 0.1 * s, len, 4), MIL_TOWER_MAT);
    m.position.set((a[0] + b[0]) / 2, h * yFrac, (a[1] + b[1]) / 2);
    m.rotation.z = Math.PI / 2;
    m.rotation.y = -Math.atan2(dz, dx);
    g.add(m);
  }
  brace(legPos[0], legPos[1], 0.5); brace(legPos[2], legPos[3], 0.5);
  brace(legPos[0], legPos[2], 0.5); brace(legPos[1], legPos[3], 0.5);
  const deck = new THREE.Mesh(new THREE.BoxGeometry(5.4 * s, 0.6 * s, 5.4 * s), MIL_BUNKER_DK_MAT);
  deck.position.y = h;
  deck.castShadow = true;
  g.add(deck);
  // Waist-high railing at the 4 platform corners only (was 8 posts).
  const railGeo = new THREE.BoxGeometry(0.14 * s, 1.0 * s, 0.14 * s);
  for (const [dx, dz] of legPos) {
    const r = new THREE.Mesh(railGeo, MIL_TOWER_MAT);
    r.position.set(dx * 1.24, h + 0.8 * s, dz * 1.24);
    g.add(r);
  }
  // Pyramidal roof canopy + antenna mast.
  const roof = new THREE.Mesh(new THREE.ConeGeometry(4.5 * s, 1.8 * s, 4), MIL_BUNKER_DK_MAT);
  roof.rotation.y = Math.PI / 4;
  roof.position.y = h + 2.2 * s;
  roof.castShadow = true;
  g.add(roof);
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.12 * s, 0.12 * s, 2.6 * s, 5), MIL_TOWER_MAT);
  post.position.y = h + (3.1 + 1.3) * s;
  g.add(post);
  g.position.set(x, 0, z);
  parent.add(g);
}

function addFlagpole(parent, x, z, h) {
  const g = new THREE.Group();
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.12, h, 6), MIL_TOWER_MAT);
  mast.position.y = h / 2;
  mast.castShadow = true;
  g.add(mast);
  const flag = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 1.3), MIL_FLAG_MAT);
  flag.position.set(1.1, h - 0.75, 0);
  g.add(flag);
  g.position.set(x, 0, z);
  parent.add(g);
}

const MIL_TENT_MAT = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0x5c6b4a });
const MIL_TENT_DK_MAT = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0x495638 });
const MIL_CRATE_MAT = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0, color: 0x6b5a3c });

function addTent(parent, x, z, w, h, len) {
  const g = new THREE.Group();
  const panelW = Math.sqrt((w / 2) ** 2 + h ** 2);
  const angle = Math.atan2(h, w / 2);
  for (const side of [-1, 1]) {
    const panel = new THREE.Mesh(new THREE.BoxGeometry(panelW, 0.12, len), MIL_TENT_MAT);
    panel.position.set(side * (w / 4), h / 2, 0);
    panel.rotation.z = side * angle;
    panel.castShadow = true;
    g.add(panel);
  }
  const ridge = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, len, 6), MIL_TENT_DK_MAT);
  ridge.rotation.x = Math.PI / 2;
  ridge.position.y = h;
  g.add(ridge);
  g.position.set(x, 0, z);
  parent.add(g);
}

function addCrateStack(parent, x, z, rng) {
  const g = new THREE.Group();
  for (let i = 0; i < 3; i++) {
    const s = 0.9 + rng() * 0.4;
    const crate = new THREE.Mesh(new THREE.BoxGeometry(s, s, s), MIL_CRATE_MAT);
    crate.position.set((rng() - 0.5) * 1.4, s / 2, (rng() - 0.5) * 1.4 + i * 0.1);
    crate.rotation.y = rng() * 0.6;
    crate.castShadow = true;
    g.add(crate);
  }
  g.position.set(x, 0, z);
  parent.add(g);
}

// Campfire + fuel drums -- the details that make a cluster of tents read
// as an actual lived-in camp instead of just "some tents dropped on the
// ground".
const MIL_FIRE_MAT = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, color: 0xff6a2a, emissive: 0xdd4400, emissiveIntensity: 0.9 });
const MIL_LOG_MAT = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0, color: 0x4a3624 });
const MIL_DRUM_MAT = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.5, color: 0x5c5a3a });
const MIL_DRUM_DK_MAT = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.5, color: 0x3a3826 });
function addCampfire(parent, x, z) {
  const g = new THREE.Group();
  const ringR = 1.1;
  const stoneGeo = new THREE.DodecahedronGeometry(0.22, 0);
  for (let i = 0; i < 8; i++) {
    const a = (2 * Math.PI * i) / 8;
    const stone = new THREE.Mesh(stoneGeo, MIL_ROCK_MAT);
    stone.position.set(Math.cos(a) * ringR, 0.15, Math.sin(a) * ringR);
    g.add(stone);
  }
  for (let i = 0; i < 4; i++) {
    const log = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 1.1, 6), MIL_LOG_MAT);
    log.rotation.z = Math.PI / 2;
    log.rotation.y = (i / 4) * Math.PI;
    log.position.y = 0.12;
    g.add(log);
  }
  const flame = new THREE.Mesh(new THREE.ConeGeometry(0.35, 0.8, 6), MIL_FIRE_MAT);
  flame.position.y = 0.5;
  g.add(flame);
  g.position.set(x, 0, z);
  parent.add(g);
}
function addFuelDrum(parent, x, z, rng) {
  const drum = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 1.0, 10), MIL_DRUM_MAT);
  drum.position.set(x, 0.5, z);
  drum.rotation.y = rng() * Math.PI;
  drum.castShadow = true;
  parent.add(drum);
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.44, 0.44, 0.06, 10), MIL_DRUM_DK_MAT);
  cap.position.set(x, 1.03, z);
  parent.add(cap);
}

// A full camp, not just a few tents: two rows of tents around a central
// campfire, crates and fuel drums along the edges, and its own flagpole --
// this needs to read clearly as an actual military encampment at a
// glance, not blend into the surrounding terrain filler.
function addSupplyDepot(parent, o, idx) {
  const rng = mulberry32(hashSeed(String(o.label || idx)));
  const c = toScene((o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2);
  const tentW = 5.5, tentH = 3.2, tentLen = 7, gapX = 9, gapZ = 11;
  for (let row = 0; row < 2; row++) {
    for (let col = 0; col < 3; col++) {
      addTent(parent, c.x + (col - 1) * gapX, c.z + (row - 0.5) * gapZ, tentW, tentH, tentLen);
    }
  }
  addCampfire(parent, c.x, c.z);
  for (let i = 0; i < 5; i++) {
    addCrateStack(parent, c.x + (i - 2) * gapX * 0.8, c.z + gapZ * 1.15, rng);
  }
  for (let i = 0; i < 4; i++) {
    addFuelDrum(parent, c.x + (i - 1.5) * 2.6, c.z - gapZ * 1.15, rng);
  }
  addFlagpole(parent, c.x - gapX * 1.6, c.z, 8);
  registerFootprint(c.x, c.z, gapX * 3 + tentW, gapZ * 2 + tentLen, tentH);
}

// Smaller standalone camp -- 2 tents + a campfire + crates/drum, not the
// full 6-tent addSupplyDepot -- for scattering extra camps directly at
// scene-space coordinates (no backing obstacle) rather than one tied to
// a real/synthetic bunker obstacle.
function addFieldCamp(parent, cx, cz, rng) {
  const tentW = 4.5, tentH = 2.6, tentLen = 5.5, gap = 6.5;
  addTent(parent, cx - gap / 2, cz, tentW, tentH, tentLen);
  addTent(parent, cx + gap / 2, cz, tentW, tentH, tentLen);
  addCampfire(parent, cx, cz + 4);
  addCrateStack(parent, cx - gap / 2, cz - 4, rng);
  addCrateStack(parent, cx + gap / 2, cz - 4, rng);
  addFuelDrum(parent, cx, cz - 5.5, rng);
}

function addMilitaryStructure(parent, o, idx) {
  // Same "obstacle rect can be huge, visual footprint stays compact and
  // seeded off the label" convention as addBuilding/addTankFarm above.
  const label = String(o.label || idx);
  if (label === 'supply-depot') { addSupplyDepot(parent, o, idx); return; }
  const isCommand = label === 'command-bunker';
  const rng = mulberry32(hashSeed(label));
  const w = isCommand ? 30 + rng() * 8 : 22 + rng() * 20;
  const d = isCommand ? 22 + rng() * 6 : 16 + rng() * 16;
  const h = isCommand ? 4.2 + rng() * 0.8 : 3.2 + rng() * 1.4;
  const group = new THREE.Group();

  const base = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), MIL_BUNKER_MAT);
  base.position.y = h / 2;
  base.castShadow = true;
  base.receiveShadow = true;
  group.add(base);

  // Corner support posts -- breaks up the bare box silhouette.
  for (const [cx, cz] of [[-w / 2, -d / 2], [w / 2, -d / 2], [-w / 2, d / 2], [w / 2, d / 2]]) {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, h + 0.4, 6), MIL_BUNKER_DK_MAT);
    post.position.set(cx, (h + 0.4) / 2, cz);
    post.castShadow = true;
    group.add(post);
  }

  // Entrance door + firing slit on the +z face, so every bunker reads as
  // an actual defended structure, not a plain box.
  const doorW = Math.min(3.2, w * 0.18);
  const door = new THREE.Mesh(new THREE.BoxGeometry(doorW, h * 0.62, 0.3), MIL_DOOR_MAT);
  door.position.set(0, h * 0.31, d / 2 + 0.04);
  group.add(door);
  const slit = new THREE.Mesh(new THREE.BoxGeometry(w * 0.7, 0.4, 0.24), MIL_DOOR_MAT);
  slit.position.set(0, h * 0.8, d / 2 + 0.04);
  group.add(slit);

  // Sandbag courses along the roofline -- the same "sandbag row" read as
  // the 2D canvas's _mil_bunker. One InstancedMesh per bunker instead of
  // one Mesh per bag (up to ~48 on a big bunker) -- with 5+ bunkers on
  // the map this was a real chunk of the scene's total draw calls for
  // small, identical-looking boxes.
  const bag = 1.1;
  const nW = Math.max(3, Math.round(w / (bag * 2.4)));
  const nD = Math.max(2, Math.round(d / (bag * 2.4)));
  const bagMesh = new THREE.InstancedMesh(UNIT_BOX_GEO, MIL_SANDBAG_MAT, nW * 2 + nD * 2);
  bagMesh.castShadow = true;
  let bagIdx = 0;
  for (let i = 0; i < nW; i++) {
    const bx = -w / 2 + (w / nW) * (i + 0.5);
    for (const bz of [-d / 2, d / 2]) {
      dummy.position.set(bx, h + bag / 2, bz);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(bag * 1.8, bag, bag);
      dummy.updateMatrix();
      bagMesh.setMatrixAt(bagIdx++, dummy.matrix);
    }
  }
  for (let i = 0; i < nD; i++) {
    const bz = -d / 2 + (d / nD) * (i + 0.5);
    for (const bx of [-w / 2, w / 2]) {
      dummy.position.set(bx, h + bag / 2, bz);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(bag, bag, bag * 1.8);
      dummy.updateMatrix();
      bagMesh.setMatrixAt(bagIdx++, dummy.matrix);
    }
  }
  bagMesh.instanceMatrix.needsUpdate = true;
  group.add(bagMesh);

  // Camo netting draped over the roof: two overlapping tilted,
  // semi-transparent planes in different tones, same "camo blotch" read
  // as the 2D canvas's ground texture.
  const net1 = new THREE.Mesh(new THREE.PlaneGeometry(w * 0.85, d * 0.85), MIL_NET_MAT);
  net1.rotation.x = -Math.PI / 2 + 0.12;
  net1.position.set(-w * 0.06, h + 0.85, d * 0.05);
  group.add(net1);
  const net2 = new THREE.Mesh(new THREE.PlaneGeometry(w * 0.6, d * 0.6), MIL_NET_MAT2);
  net2.rotation.x = -Math.PI / 2 - 0.08;
  net2.rotation.z = 0.15;
  net2.position.set(w * 0.1, h + 1.05, -d * 0.08);
  group.add(net2);

  const c = toScene((o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2);
  group.position.set(c.x, 0, c.z);
  parent.add(group);

  // Every bunker/checkpoint gets its own watchtower, matching the 2D
  // canvas's fully-deterministic per-bunker tower (not probabilistic).
  addWatchtower(parent, c.x + w * 0.5 + 3.4, c.z, isCommand ? 46 + rng() * 10 : 32 + rng() * 14);
  if (isCommand) {
    addFlagpole(parent, c.x - w * 0.5 - 3, c.z + d * 0.3, 9);
  }
  registerFootprint(c.x, c.z, w, d, h);
}

// Mobile units (tanks, planes, soldiers) are built at realistic
// real-world scale, then scaled up uniformly here -- on a map this wide
// (802.11ah's real long range routinely puts nodes hundreds of metres to
// 1km+ from the AP, see militaryExtent's docstring), a true-to-life 6m
// tank is a handful of pixels next to the AP/STA node markers (fixed at
// ~9 world units regardless of map scale, see entities.js), effectively
// invisible from the default framing. Scaling the whole unit up keeps it
// proportioned correctly (every part grows together, nothing decouples)
// while making it read clearly against the node markers.
const MIL_UNIT_SCALE = 10;
// Soldiers get their own, larger multiplier -- a human silhouette is much
// smaller in real life than a tank (~1.8m vs ~6.4m) and reads as
// disproportionately tiny standing next to a 10x tank even at the same
// scale factor; boosting soldiers further makes them a clearly visible
// presence in their own right rather than an afterthought next to the
// vehicles.
const MIL_SOLDIER_SCALE = 24;

// ---- shared "bright punch" sprite: one soft radial-gradient texture,
// canvas-generated once (same technique entities.js's own text-label
// sprites already use, see makeInfoSprite there -- no external image
// asset needed), reused as the `map` for every muzzle flash / gun flash /
// explosion flash sprite in this file. A flash is a burst of LIGHT, not a
// solid object, so a soft round glow (unlike the deliberately blocky
// primitives everywhere else in this file) is the right register for it;
// solid-matter effects (explosion fireball/smoke, see stepMilitaryPatrol's
// bomb system) stay blocky cubes to match the established aesthetic.
let MIL_GLOW_TEX = null;
function glowTexture() {
  if (MIL_GLOW_TEX) return MIL_GLOW_TEX;
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.35, 'rgba(255,255,255,0.9)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  MIL_GLOW_TEX = new THREE.CanvasTexture(canvas);
  return MIL_GLOW_TEX;
}

// One shared, additive-blended sprite material for every small-arms
// (tank cannon + soldier rifle) muzzle flash -- ~214 units can carry one
// at once (24 tanks + ~190 soldiers, see rebuildMilitaryPatrol's spawn
// counts), and since only a couple are ever visible at the same instant
// (each fires for ~0.1s out of a 7s cycle, see the muzzle-flash blocks in
// stepMilitaryPatrol), sharing ONE material -- exactly like MIL_MUZZLE_MAT
// below already does for the cone mesh -- avoids allocating hundreds of
// one-off Sprite materials that would otherwise need their own disposal
// bookkeeping on every Military Zone rebuild. Guns/explosions (a dozen and
// a handful respectively, not hundreds) get their own per-instance cloned
// materials instead, see makeGunRig and the bomb-effect code further down.
const MIL_MUZZLE_FLASH_MAT = new THREE.SpriteMaterial({
  map: glowTexture(), color: 0xfff2c4, blending: THREE.AdditiveBlending,
  transparent: true, depthWrite: false, opacity: 0.95,
});
// `size` is in the LOCAL unit space of whatever the sprite gets parented
// under (turretGroup for tanks, the soldier model's own local group, or
// `rifle` for the real GLB soldier -- see each call site), so it inherits
// that ancestor's scale (MIL_UNIT_SCALE/MIL_SOLDIER_SCALE/the rifle's own
// bone-scale-cancelling factor) the exact same way the existing muzzle
// cone already does.
function makeMuzzleFlashSprite(size) {
  const spr = new THREE.Sprite(MIL_MUZZLE_FLASH_MAT);
  spr.scale.setScalar(size);
  spr.visible = false;
  return spr;
}

// Per-instance flash sprite for the low-count props that DO need
// independent opacity/scale animation (mortar/artillery guns, bomb
// impacts) -- unlike the shared small-arms material above, these clone
// their own SpriteMaterial since only a dozen guns and a handful of live
// explosions ever exist at once, so the extra material objects are cheap
// and (for guns) get disposed for free by clearMilitaryStatics' generic
// per-descendant traversal, or (for explosions) explicitly on burst-end --
// see stepMilitaryPatrol's bomb-effect code.
function makeFlashSprite(size, color) {
  const mat = new THREE.SpriteMaterial({
    map: glowTexture(), color, blending: THREE.AdditiveBlending,
    transparent: true, depthWrite: false, opacity: 1,
  });
  const spr = new THREE.Sprite(mat);
  spr.scale.setScalar(size);
  spr.userData.baseScale = size;
  spr.visible = false;
  return spr;
}

const MIL_MUZZLE_MAT = new THREE.MeshBasicMaterial({ color: 0xffdd77 });
function buildMilitaryTankMesh(rng) {
  const g = new THREE.Group();
  const hullMat = MIL_TANK_HULL_MATS[(rng() * MIL_TANK_HULL_MATS.length) | 0];

  const hull = new THREE.Mesh(new THREE.BoxGeometry(6.4, 1.5, 3.4), hullMat);
  hull.position.y = 1.1;
  hull.castShadow = true;
  g.add(hull);

  // Sloped glacis plate at the front -- reads as an actual tank silhouette
  // instead of a plain box on tracks.
  const glacis = new THREE.Mesh(new THREE.BoxGeometry(1.7, 1.5, 3.2), hullMat);
  glacis.position.set(3.35, 1.0, 0);
  glacis.rotation.z = -0.5;
  glacis.castShadow = true;
  g.add(glacis);

  const wheelGeo = new THREE.CylinderGeometry(0.42, 0.42, 0.95, 8);
  for (const tz of [-1.75, 1.75]) {
    const track = new THREE.Mesh(new THREE.BoxGeometry(6.8, 0.9, 0.85), MIL_TANK_DK_MAT);
    track.position.set(0, 0.55, tz);
    track.castShadow = true;
    g.add(track);
    for (let i = 0; i < 5; i++) {
      const wheel = new THREE.Mesh(wheelGeo, MIL_TANK_DK_MAT);
      wheel.rotation.x = Math.PI / 2;
      wheel.position.set(-2.6 + i * 1.3, 0.5, tz);
      wheel.castShadow = true;
      g.add(wheel);
    }
  }

  // Turret/hatch/barrel/muzzle live in their own sub-group, pivoting
  // independently of the hull -- see stepMilitaryPatrol, which slowly
  // scans it side to side regardless of travel heading, and flashes the
  // muzzle mesh on a timer. Positions below are unchanged from before,
  // just re-expressed relative to this group's own (-0.4, 0, 0) origin.
  const turretGroup = new THREE.Group();
  turretGroup.position.set(-0.4, 0, 0);
  g.add(turretGroup);

  const turret = new THREE.Mesh(new THREE.CylinderGeometry(1.3, 1.4, 1.1, 8), MIL_TANK_TURRET_MAT);
  turret.position.set(0, 2.4, 0);
  turret.castShadow = true;
  turretGroup.add(turret);
  const hatch = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 0.3, 8), MIL_TANK_DK_MAT);
  hatch.position.set(0, 3.05, 0.4);
  turretGroup.add(hatch);

  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 4.4, 6), MIL_TANK_DK_MAT);
  barrel.rotation.z = Math.PI / 2;
  barrel.position.set(2.6, 2.4, 0);
  barrel.castShadow = true;
  turretGroup.add(barrel);

  const muzzle = new THREE.Mesh(new THREE.ConeGeometry(0.4, 1.0, 6), MIL_MUZZLE_MAT);
  muzzle.rotation.z = -Math.PI / 2;
  muzzle.position.set(4.85, 2.4, 0);
  muzzle.visible = false;
  turretGroup.add(muzzle);

  // Real muzzle flash -- the cone above is kept as-is (still the trigger
  // stepMilitaryPatrol toggles), but a bare 0.4-radius flat-shaded cone is
  // too subtle to read as "firing" at normal viewing distance. This bright
  // additive sprite sits at the same spot, sized clearly against the
  // 64-unit-long scaled hull. Tanks are few enough (24 total, see
  // rebuildMilitaryPatrol) to also afford a real PointLight for punch --
  // soldiers (~190 of them) deliberately do NOT get one, see
  // MIL_MUZZLE_FLASH_MAT's own comment for why.
  const flash = makeMuzzleFlashSprite(2.2);
  flash.position.copy(muzzle.position);
  turretGroup.add(flash);
  const flashLight = new THREE.PointLight(0xffb060, 0, 60, 2);
  flashLight.position.copy(muzzle.position);
  turretGroup.add(flashLight);

  const antenna = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 2.4, 4), MIL_TANK_DK_MAT);
  antenna.position.set(-1.0, 3.6, -0.6);
  antenna.rotation.z = 0.12;
  g.add(antenna);

  const wheels = [];
  for (const ez of [-1.0, 1.0]) {
    const ex = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 0.6, 6), MIL_TANK_DK_MAT);
    ex.rotation.z = Math.PI / 2;
    ex.position.set(-3.1, 1.1, ez);
    g.add(ex);
  }
  for (const c of g.children) {
    // Collect the road wheels (not the drive sprockets/idlers -- there
    // aren't separate ones here) for the rolling animation: anything
    // built from wheelGeo above.
    if (c.geometry === wheelGeo) wheels.push(c);
  }

  g.scale.setScalar(MIL_UNIT_SCALE);
  const dustAnchor = { x: 0, y: 0, z: 0, active: false };
  registerDustEmitter(dustAnchor);
  return { group: g, turretGroup, muzzle, flash, flashLight, wheels, dustAnchor, dustTrailDist: 4.2 };
}

function buildSoldierMesh(rng) {
  const g = new THREE.Group();
  const uniformMat = MIL_SOLDIER_UNIFORM_MATS[(rng() * MIL_SOLDIER_UNIFORM_MATS.length) | 0];
  const legGeo = new THREE.CylinderGeometry(0.09, 0.09, 0.75, 5);
  for (const lx of [-0.13, 0.13]) {
    const leg = new THREE.Mesh(legGeo, uniformMat);
    leg.position.set(lx, 0.38, 0);
    leg.castShadow = true;
    g.add(leg);
  }
  const torso = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.62, 0.26), uniformMat);
  torso.position.y = 1.06;
  torso.castShadow = true;
  g.add(torso);
  const pack = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.4, 0.16), MIL_TANK_DK_MAT);
  pack.position.set(0, 1.08, -0.2);
  g.add(pack);
  const armGeo = new THREE.CylinderGeometry(0.075, 0.075, 0.55, 5);
  for (const ax of [-0.28, 0.28]) {
    const arm = new THREE.Mesh(armGeo, uniformMat);
    arm.position.set(ax, 1.02, 0.08);
    arm.rotation.x = -0.35;
    arm.castShadow = true;
    g.add(arm);
  }
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.19, 6, 6), MIL_SOLDIER_SKIN_MAT);
  head.position.y = 1.52;
  head.castShadow = true;
  g.add(head);
  const helmet = new THREE.Mesh(
    new THREE.SphereGeometry(0.22, 6, 6, 0, Math.PI * 2, 0, Math.PI * 0.6), MIL_HELMET_MAT);
  helmet.position.y = 1.57;
  helmet.castShadow = true;
  g.add(helmet);
  // MIL_RIFLE_MAT (defined below, safe to reference here since this only
  // runs when buildSoldierMesh is actually called, well after module
  // load finishes) -- was MIL_TANK_DK_MAT, near-invisible against the
  // dark uniform/ground at normal viewing distance.
  const rifle = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.1, 0.1), MIL_RIFLE_MAT);
  rifle.position.set(0.2, 1.15, 0.28);
  rifle.rotation.z = 0.15;
  g.add(rifle);
  // Same shared flash mesh/timer convention as the tank turret's --
  // reuses stepMilitaryPatrol's existing `if (t.muzzle)` toggle, no new
  // per-frame code needed.
  const muzzle = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.16, 5), MIL_MUZZLE_MAT);
  muzzle.rotation.z = -Math.PI / 2;
  muzzle.position.set(0.64, 1.28, 0.28);
  muzzle.visible = false;
  g.add(muzzle);
  // Bright flash to go with the cone above -- shares MIL_MUZZLE_FLASH_MAT
  // (see its own comment) rather than a per-soldier material, and
  // deliberately has no PointLight partner: soldiers are numerous enough
  // (~190) that only the tank's cannon gets one.
  const flash = makeMuzzleFlashSprite(0.3);
  flash.position.copy(muzzle.position);
  g.add(flash);
  g.scale.setScalar(MIL_SOLDIER_SCALE);
  // This placeholder is built facing local +Z (backpack sits at -0.2z,
  // i.e. behind; rifle/muzzle sit at +0.28z, i.e. in front) -- same
  // mismatch buildRealSoldier's GLB model had against stepMilitaryPatrol's
  // +X-is-forward heading convention, and the same fix: wrap in an outer
  // group so stepMilitaryPatrol's per-frame rotation.y still drives
  // heading on the WRAPPER, while this fixed correction on the inner
  // group realigns its actual forward axis first. -90 (not +90) per a
  // direct report that the +90 version had it facing opposite its
  // direction of travel.
  g.rotation.y = -Math.PI / 2;
  const wrapper = new THREE.Group();
  wrapper.add(g);
  return { group: wrapper, muzzle, flash, isSoldier: true };
}

// Real modelled soldier, cloned from the loaded Soldier.glb (see the
// import block at the top of this file). Returns null if the model
// hasn't finished loading yet -- callers fall back to buildSoldierMesh
// in that case, and lastMilitaryKey gets reset once loading finishes so
// the next poll swaps the placeholders for the real model automatically.
// Soldier.glb (see the import block at the top of this file) is Mixamo's
// "Vanguard" character -- a fully-rigged body + visor mesh, but its hands
// are genuinely empty, no weapon geometry anywhere in the file. Built and
// parented onto the right-hand bone so it inherits the walk animation's
// arm swing instead of needing its own separate animation.
// Lighter gunmetal gray, not realistic near-black -- at typical camera
// distance the old 0x2b2a22 was indistinguishable from both the dark
// uniform and the ground shadow, so the rifle was technically there but
// never actually visible.
const MIL_RIFLE_MAT = new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.7, color: 0x6b6a60 });
function attachRifle(model) {
  const hand = model.getObjectByName('mixamorigRightHand');
  if (!hand) return null;
  const rifle = new THREE.Group();
  // Dimensions are in WORLD units, sized against the soldier's actual
  // rendered body (~21 wide x 40 tall) -- NOT the same small unit
  // convention the rest of this rig's geometry uses. Reason: this rig's
  // bones carry their own internal scale (~0.24 measured on the hand
  // bone) that's separate from and much smaller than the model wrapper's
  // MIL_SOLDIER_SCALE(24) -- the skinned body mesh renders at full size
  // regardless (skinning uses the mesh's own bind-pose math, not simple
  // parent-child transform inheritance), but a plain object parented
  // directly onto a bone, like this rifle, inherits that tiny bone scale
  // directly. It was correctly attached and correctly sized in its own
  // small-unit terms, just rendering ~100x smaller than intended as a
  // result -- effectively invisible. rifle.scale below cancels that bone
  // scale out so these dimensions land as real, visible world units.
  const barrel = new THREE.Mesh(new THREE.BoxGeometry(20, 2.6, 2.6), MIL_RIFLE_MAT);
  rifle.add(barrel);
  const stock = new THREE.Mesh(new THREE.BoxGeometry(5.2, 3.8, 2.1), MIL_RIFLE_MAT);
  stock.position.set(-11, -0.5, 0);
  rifle.add(stock);
  const mag = new THREE.Mesh(new THREE.BoxGeometry(2.1, 5.8, 2.1), MIL_RIFLE_MAT);
  mag.position.set(2.1, -3.8, 0);
  mag.rotation.z = 0.3;
  rifle.add(mag);
  const sight = new THREE.Mesh(new THREE.BoxGeometry(1.7, 2.1, 1.2), MIL_RIFLE_MAT);
  sight.position.set(3.1, 2.1, 0);
  rifle.add(sight);
  // Muzzle tip -- barrel spans -10..10 along local X (stock/grip side is
  // negative), so the business end is +X regardless of whatever world
  // orientation the hand bone is currently swung to.
  const muzzle = new THREE.Mesh(new THREE.ConeGeometry(1.4, 4.3, 5), MIL_MUZZLE_MAT);
  muzzle.rotation.z = -Math.PI / 2;
  muzzle.position.set(11, 0, 0);
  muzzle.visible = false;
  rifle.add(muzzle);
  // Same shared-material flash sprite as the placeholder soldier's rifle
  // (see MIL_MUZZLE_FLASH_MAT/makeMuzzleFlashSprite) -- sized in the same
  // real-world units as the barrel/muzzle above, since `rifle.scale`
  // (set below) already cancels out the hand bone's tiny internal scale.
  const flash = makeMuzzleFlashSprite(9);
  flash.position.copy(muzzle.position);
  rifle.add(flash);
  rifle.position.set(0.7, -0.5, 0.5);
  const handWorldScale = new THREE.Vector3();
  hand.getWorldScale(handWorldScale);
  rifle.scale.set(1 / handWorldScale.x, 1 / handWorldScale.y, 1 / handWorldScale.z);
  hand.add(rifle);
  return { muzzle, flash };
}

// Lazily built the first time a real soldier is placed (needs an actual
// mesh from the loaded GLB to clone as a template) -- a handful of
// pre-tinted materials shared across every soldier instance, same
// "small fixed array, index with rng()" convention as MIL_TANK_HULL_MATS.
let SOLDIER_UNIFORM_GLB_MATS = null;

function buildRealSoldier(rng, opts) {
  if (!soldierGLTF) return null;
  const model = skeletonClone(soldierGLTF.scene);
  // Earlier attempt tinted the model's own material colour directly here
  // and looked crushed almost to black -- turned out that was mutating
  // the ONE shared material object every soldier clone still points at
  // (skeletonClone() clones the node hierarchy but not materials by
  // reference), stacking a tint on top of whatever the last soldier built
  // that frame had already set, not a single clean multiply. Building a
  // small set of independent tinted clones up front (map/metalness/
  // roughness all carried over from the real base material) and
  // assigning one per soldier fixes both the stacking bug and gives each
  // soldier its own stable colour.
  if (!SOLDIER_UNIFORM_GLB_MATS) {
    let baseMat = null;
    model.traverse((o) => {
      if (o.isMesh && o.name === 'vanguard_Mesh' && !baseMat) baseMat = o.material;
    });
    if (baseMat) {
      SOLDIER_UNIFORM_GLB_MATS = MIL_UNIT_PALETTE.map((c) => {
        const m = baseMat.clone();
        m.color.setHex(c.main);
        return m;
      });
    }
  }
  const liveryMat = SOLDIER_UNIFORM_GLB_MATS
    ? SOLDIER_UNIFORM_GLB_MATS[(rng() * SOLDIER_UNIFORM_GLB_MATS.length) | 0]
    : null;
  model.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
      if (liveryMat && o.name === 'vanguard_Mesh') o.material = liveryMat;
    }
  });
  // Scale applied BEFORE attachRifle (not after, as this used to read) --
  // attachRifle measures the hand bone's actual WORLD scale to size the
  // rifle correctly (see its own comment for why that's necessary), which
  // only reflects this model.scale if it's already been set by the time
  // attachRifle runs.
  model.scale.setScalar(MIL_SOLDIER_SCALE);
  const rifleParts = attachRifle(model);
  const muzzle = rifleParts?.muzzle ?? null;
  const flash = rifleParts?.flash ?? null;
  const mixer = new THREE.AnimationMixer(model);
  // Run, not Walk -- these units travel at 9 m/s (see convoyLane(9) in
  // rebuildMilitaryPatrol), faster than Walk's authored pace. Soldier.glb
  // (see the import block up top) actually ships Idle/Run/TPose/Walk
  // clips; only Walk was ever wired up. Falls back to Walk if Run is
  // somehow missing. timeScale stays close to 1 -- an earlier 1.6x on
  // top of both the faster clip AND fast translation over-corrected into
  // legs cycling far quicker than the body could plausibly be covering
  // ground, reading as a frantic blur instead of an actual run.
  // Stationary/"discussing" huddles (see spawnStandingGroups) pass
  // opts.idle instead -- Idle is a real clip in the same file, just never
  // wired up before since every soldier used to be always on the move.
  const wantIdle = !!(opts && opts.idle);
  const clip = wantIdle
    ? (THREE.AnimationClip.findByName(soldierGLTF.animations, 'Idle')
      || THREE.AnimationClip.findByName(soldierGLTF.animations, 'Walk'))
    : (THREE.AnimationClip.findByName(soldierGLTF.animations, 'Run')
      || THREE.AnimationClip.findByName(soldierGLTF.animations, 'Walk'));
  const action = mixer.clipAction(clip);
  action.play();
  action.timeScale = wantIdle ? 1 : 1.1;
  mixer.update(rng() * clip.duration); // stagger so soldiers aren't in lockstep
  // Vanguard's rest pose actually faces world +Z, but stepMilitaryPatrol
  // sets heading assuming local +X is forward -- the same convention
  // every procedurally-built unit (tanks/trucks/planes) already follows,
  // since those get modelled nose-first along +X on purpose. Fixed by
  // wrapping in an outer group: stepMilitaryPatrol still fully drives the
  // WRAPPER's rotation.y every frame for heading, while this fixed
  // offset on the inner model realigns its actual forward axis first.
  // -90 (not +90) per a direct report that the +90 version had it facing
  // opposite its direction of travel.
  model.rotation.y = -Math.PI / 2;
  const wrapper = new THREE.Group();
  wrapper.add(model);
  return { group: wrapper, mixer, muzzle, flash, isSoldier: true };
}

// User-supplied billboard soldier (soldier_sprite.png, see the loader up
// top) -- a flat, always-camera-facing cutout instead of the rigged 3D
// model. No skeleton/mixer, so no walk-cycle to speed-match and no bones
// to hang a rifle off; the muzzle flash is just a small bright quad
// positioned over roughly where the gun sits in the source art.
// Patrol aircraft -- flies circular loops well above the base (see
// stepMilitaryPatrol's 'air' branch, using circleLoopPos instead of the
// ground units' rectLoopPos), the aerial half of "tanks, soldiers,
// aircraft" a real battle zone needs. Local +X is "forward" (the nose
// direction), same convention buildMilitaryTankMesh uses, so the shared
// atan2(hy,hx) heading code in stepMilitaryPatrol works unchanged.
// Fuselage/nose get the livery's main tone, wings/tail/tanks its
// coordinated dark trim -- cockpit glass stays a fixed dark tactical tint
// regardless of livery (real canopy glass doesn't repaint with the hull).
const MIL_PLANE_LIVERY_MATS = buildLiveryMatPairs();
const MIL_PLANE_COCKPIT_MAT = new THREE.MeshStandardMaterial({ roughness: 0.15, metalness: 0.3, color: 0x2f4550 });
function buildMilitaryPlaneMesh(rng) {
  const liv = pickLivery(MIL_PLANE_LIVERY_MATS, rng);
  const g = new THREE.Group();
  const fuselage = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.3, 7.5, 8), liv.main);
  fuselage.rotation.z = Math.PI / 2;
  fuselage.castShadow = true;
  g.add(fuselage);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.55, 1.6, 8), liv.main);
  nose.rotation.z = -Math.PI / 2;
  nose.position.set(4.3, 0, 0);
  g.add(nose);
  const cockpit = new THREE.Mesh(new THREE.SphereGeometry(0.42, 8, 6), MIL_PLANE_COCKPIT_MAT);
  cockpit.scale.set(1.5, 0.75, 0.75);
  cockpit.position.set(1.9, 0.38, 0);
  g.add(cockpit);
  const wing = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.12, 7.2), liv.dark);
  wing.position.set(-0.4, -0.15, 0);
  wing.castShadow = true;
  g.add(wing);
  const tailWing = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.1, 2.8), liv.dark);
  tailWing.position.set(-3.4, 0, 0);
  g.add(tailWing);
  const fin = new THREE.Mesh(new THREE.BoxGeometry(1.3, 1.4, 0.12), liv.dark);
  fin.position.set(-3.5, 0.75, 0);
  g.add(fin);
  for (const side of [-1, 1]) {
    const tank = new THREE.Mesh(new THREE.CapsuleGeometry(0.14, 1.4, 2, 6), liv.dark);
    tank.rotation.z = Math.PI / 2;
    tank.position.set(-0.2, -0.45, side * 2.6);
    g.add(tank);
  }
  g.scale.setScalar(MIL_UNIT_SCALE);
  return { group: g };
}

// Helicopter -- shares the plane's 'convoyAir' movement kind but flies its
// own lower helipad-to-helipad lane at a fixed cruise altitude (no climb/
// descent -- see stepMilitaryPatrol's convoyAir branch). mainRotor/
// tailRotor are returned by name so stepMilitaryPatrol can spin them
// directly every frame without a tree traversal.
// Body gets the livery's main tone, tail boom/fin/skids/struts/mast its
// dark trim -- cockpit glass and rotor blades stay fixed regardless of
// livery, same reasoning as the plane's canopy.
const MIL_HELI_LIVERY_MATS = buildLiveryMatPairs();
const MIL_HELI_GLASS_MAT = new THREE.MeshStandardMaterial({ roughness: 0.12, metalness: 0.35, color: 0x7fa0b0 });
const MIL_ROTOR_MAT = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.7, color: 0x121212 });
function buildMilitaryHelicopterMesh(rng) {
  const liv = pickLivery(MIL_HELI_LIVERY_MATS, rng);
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.75, 3.2, 4, 8), liv.main);
  body.rotation.z = Math.PI / 2;
  body.castShadow = true;
  g.add(body);
  const cockpit = new THREE.Mesh(new THREE.SphereGeometry(0.7, 8, 6), MIL_HELI_GLASS_MAT);
  cockpit.scale.set(1.25, 1, 1);
  cockpit.position.set(2.15, 0.05, 0);
  g.add(cockpit);
  const tailBoom = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.12, 4.4, 6), liv.dark);
  tailBoom.rotation.z = Math.PI / 2;
  tailBoom.position.set(-3.5, 0.35, 0);
  tailBoom.castShadow = true;
  g.add(tailBoom);
  const tailFin = new THREE.Mesh(new THREE.BoxGeometry(0.7, 1.0, 0.1), liv.dark);
  tailFin.position.set(-5.6, 0.75, 0);
  g.add(tailFin);
  const tailRotor = new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.3, 0.14), MIL_ROTOR_MAT);
  tailRotor.position.set(-5.65, 0.75, 0.22);
  g.add(tailRotor);
  for (const side of [-1, 1]) {
    const skid = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.1, 0.12), liv.dark);
    skid.position.set(-0.3, -1.15, side * 0.9);
    g.add(skid);
    for (const sx of [1.2, -1.7]) {
      const strut = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.9, 6), liv.dark);
      strut.position.set(sx, -0.65, side * 0.85);
      strut.rotation.z = (sx > 0 ? 0.25 : -0.25) * side;
      g.add(strut);
    }
  }
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.6, 6), liv.dark);
  mast.position.set(0.3, 1.15, 0);
  g.add(mast);
  const mainRotor = new THREE.Mesh(new THREE.BoxGeometry(9.0, 0.06, 0.34), MIL_ROTOR_MAT);
  mainRotor.position.set(0.3, 1.5, 0);
  g.add(mainRotor);
  g.scale.setScalar(MIL_UNIT_SCALE);
  return { group: g, mainRotor, tailRotor };
}

// Supply truck -- a ground unit distinct from the tank (cargo bed +
// canvas canopy instead of a turret/barrel), same local +X-forward,
// scale-then-return convention as the other mobile units. Cab/hood get
// the livery's main tone, bed its dark trim; the canvas tarp stays a
// fixed neutral tan regardless of livery (a canopy cover is usually
// separate stores, not painted to match the cab) and wheels stay dark --
// tires being colorful reads as wrong regardless of the rest of the paint.
const MIL_TRUCK_LIVERY_MATS = buildLiveryMatPairs();
const MIL_TRUCK_CANVAS_MAT = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0x6b6048 });
const MIL_TRUCK_WHEEL_MAT = new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0.15, color: 0x232323 });
function buildMilitaryTruckMesh(rng) {
  const liv = pickLivery(MIL_TRUCK_LIVERY_MATS, rng);
  const g = new THREE.Group();
  const cab = new THREE.Mesh(new THREE.BoxGeometry(1.8, 1.6, 2.0), liv.main);
  cab.position.set(2.6, 1.1, 0);
  cab.castShadow = true;
  g.add(cab);
  const hood = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.7, 1.9), liv.main);
  hood.position.set(3.7, 0.75, 0);
  hood.castShadow = true;
  g.add(hood);
  const bed = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.9, 2.1), liv.dark);
  bed.position.set(-0.6, 0.95, 0);
  bed.castShadow = true;
  g.add(bed);
  const tarp = new THREE.Mesh(new THREE.BoxGeometry(4.0, 1.3, 2.0), MIL_TRUCK_CANVAS_MAT);
  tarp.position.set(-0.6, 1.9, 0);
  tarp.castShadow = true;
  g.add(tarp);
  const wheelGeo = new THREE.CylinderGeometry(0.55, 0.55, 0.5, 10);
  const wheels = [];
  for (const wx of [3.0, -0.6, -1.8]) {
    for (const side of [-1, 1]) {
      const wheel = new THREE.Mesh(wheelGeo, MIL_TRUCK_WHEEL_MAT);
      wheel.rotation.x = Math.PI / 2;
      wheel.position.set(wx, 0.55, side * 1.15);
      wheel.castShadow = true;
      g.add(wheel);
      wheels.push(wheel);
    }
  }
  // The truck's own local geometry is somewhat smaller than the tank's
  // even at the same MIL_UNIT_SCALE -- measured via Box3 at 1.7x this
  // came out to 117 units long vs the tank's 78.5 (overshot); 1.14x
  // brings the two to a matching footprint instead of making the truck
  // the bigger of the two.
  g.scale.setScalar(MIL_UNIT_SCALE * 1.14);
  const dustAnchor = { x: 0, y: 0, z: 0, active: false };
  registerDustEmitter(dustAnchor);
  return { group: g, wheels, dustAnchor, dustTrailDist: 3.2 };
}

// Full barbed-wire perimeter (posts + 4 wire strands) around the whole
// base -- the 3D counterpart to the 2D canvas's dashed perimeter
// rectangle, sized off siteRadius the same way the patrol loops are
// (there's no "perimeter" obstacle rect to key off; this is a decorative
// boundary just like the 2D one, no PHY effect either side).
function addMilitaryPerimeter(parent, hw, hd) {
  const corners = [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd], [-hw, -hd]];
  // Posts spaced every 26m around a perimeter that can run well past 2km
  // total on a wide 802.11ah topology used to mean 200-300+ individual
  // Mesh objects for this one fence -- a single InstancedMesh instead.
  const postGeo = new THREE.CylinderGeometry(0.13, 0.13, 1.8, 5);
  const spacing = 26;
  const segCounts = [];
  for (let i = 0; i < 4; i++) {
    const [x0, z0] = corners[i], [x1, z1] = corners[i + 1];
    segCounts.push(Math.max(3, Math.round(Math.hypot(x1 - x0, z1 - z0) / spacing)));
  }
  const totalPosts = segCounts.reduce((a, b) => a + b, 0);
  const postMesh = new THREE.InstancedMesh(postGeo, MIL_POST_MAT, totalPosts);
  postMesh.castShadow = true;
  let postIdx = 0;
  for (let i = 0; i < 4; i++) {
    const [x0, z0] = corners[i], [x1, z1] = corners[i + 1];
    const n = segCounts[i];
    for (let k = 0; k < n; k++) {
      const t = k / n;
      dummy.position.set(x0 + (x1 - x0) * t, 0.9, z0 + (z1 - z0) * t);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      postMesh.setMatrixAt(postIdx++, dummy.matrix);
    }
  }
  postMesh.instanceMatrix.needsUpdate = true;
  parent.add(postMesh);

  for (const hgt of [0.5, 1.0, 1.5, 1.9]) {
    const pts = corners.map(([x, z]) => new THREE.Vector3(x, hgt, z));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    parent.add(new THREE.Line(geo, MIL_WIRE_MAT));
  }
}

// ---- mortar pit / artillery gun firing rig -- these two are static world
// props built once per Military Zone rebuild (not per-frame members of
// militaryMeshes, which only holds mobile patrol/convoy units), so they
// need their own small bookkeeping struct rather than reusing the
// t.muzzle/t.flash fields stepMilitaryPatrol's per-unit loop already
// drives. `militaryGuns` (declared below, near militaryMeshes) holds one
// of these per gun; stepMilitaryPatrol steps it every frame from the same
// call site app.js's animate() already uses, so no new wiring into app.js
// is needed. Real mortars/howitzers fire far less often than a rifle --
// `period` is picked per weapon class (mortar quicker, howitzer slower and
// heavier) to read as "occasional heavy gun", nothing close to the
// small-arms muzzle-flash's own 7s cycle.
// `barrel`'s OWN local rotation is fixed at build time and never changes,
// so the direction its local +Y axis points in ITS PARENT's space (i.e.
// the muzzle-forward direction) can be computed once up front with
// applyEuler and reused every frame for the recoil kick, rather than
// recomputed per frame.
function makeGunRig(barrel, tipLocalY, kickDist, period, seedLabel) {
  const anchor = new THREE.Object3D();
  anchor.position.set(0, tipLocalY, 0);
  barrel.add(anchor);
  const flash = makeFlashSprite(tipLocalY * 2.4, 0xffdd99);
  anchor.add(flash);
  // Only a dozen of these exist total (2 mortars + 10 artillery across
  // both bases, see buildBaseExtras) so a real PointLight per gun -- unlike
  // the 190-soldier small-arms case -- is cheap punch for a "big gun"
  // moment, not a performance risk.
  const light = new THREE.PointLight(0xffaa55, 0, 110, 2);
  anchor.add(light);
  const restPos = barrel.position.clone();
  const recoilDir = new THREE.Vector3(0, 1, 0).applyEuler(barrel.rotation).normalize();
  // Timing doesn't need seeded reproducibility (see this file's own
  // convention for the small-arms muzzle-flash cycle, which is plain
  // simTime-based), but mulberry32(hashSeed(...)) is the cheapest way at
  // hand to give each gun its own stable, well-spread phase offset so a
  // whole battery doesn't fire in lockstep.
  const rng = mulberry32(hashSeed(seedLabel));
  return { barrel, restPos, recoilDir, kickDist, flash, light, period, phase: rng() * period };
}

function addMortarPit(parent, cx, cz, tag) {
  const r = 5.5;
  const rim = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 0.3, 16, 1, true), MIL_PAD_MAT);
  rim.position.set(cx, 0.15, cz);
  parent.add(rim);
  const bagGeo = new THREE.BoxGeometry(0.9, 0.55, 0.55);
  const n = 12;
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    const bag = new THREE.Mesh(bagGeo, MIL_SANDBAG_MAT);
    bag.position.set(cx + Math.cos(a) * r, 0.3, cz + Math.sin(a) * r);
    bag.rotation.y = a;
    bag.castShadow = true;
    parent.add(bag);
  }
  const tube = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 2.6, 6), MIL_TANK_DK_MAT);
  tube.rotation.z = -0.9;
  tube.position.set(cx + 0.6, 1.1, cz);
  parent.add(tube);
  return makeGunRig(tube, 1.3, 0.45, 10, `mortar|${tag}|${cx}|${cz}`);
}

// Towed artillery piece -- split-trail carriage + wheels + gun shield +
// a long barrel angled up for indirect fire, bigger and more substantial
// than the mortar pit above (a real towed howitzer dwarfs an 81mm mortar
// tube). `angle` points the barrel outward, away from the base centre.
function addArtilleryGun(parent, cx, cz, angle, tag) {
  const g = new THREE.Group();
  const wheelGeo = new THREE.CylinderGeometry(1.3, 1.3, 0.4, 12);
  for (const side of [-1, 1]) {
    const wheel = new THREE.Mesh(wheelGeo, MIL_TANK_DK_MAT);
    wheel.rotation.z = Math.PI / 2;
    wheel.position.set(0, 1.3, side * 1.5);
    wheel.castShadow = true;
    g.add(wheel);
  }
  const axle = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 3.0, 8), MIL_TANK_DK_MAT);
  axle.rotation.z = Math.PI / 2;
  axle.position.set(0, 1.3, 0);
  g.add(axle);
  // Split trail legs, splayed out behind for stability when firing.
  for (const side of [-1, 1]) {
    const trail = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.28, 0.28), MIL_TANK_DK_MAT);
    trail.position.set(-2.8, 0.7, side * 0.7);
    trail.rotation.y = side * 0.18;
    trail.castShadow = true;
    g.add(trail);
  }
  const shield = new THREE.Mesh(new THREE.BoxGeometry(0.18, 1.9, 3.2), MIL_TANK_TURRET_MAT);
  shield.position.set(0.8, 1.9, 0);
  shield.castShadow = true;
  g.add(shield);
  const cradle = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.6, 0.6), MIL_TANK_DK_MAT);
  cradle.position.set(0.7, 1.7, 0);
  g.add(cradle);
  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.26, 6.6, 8), MIL_TANK_TURRET_MAT);
  barrel.rotation.z = Math.PI / 2 - 0.32; // angled up ~18 degrees for indirect fire
  barrel.position.set(3.1, 2.7, 0);
  barrel.castShadow = true;
  g.add(barrel);
  const brake = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.6, 8), MIL_TANK_DK_MAT);
  brake.rotation.z = Math.PI / 2 - 0.32;
  brake.position.set(6.1, 3.7, 0);
  g.add(brake);
  // A few nearby ammo crates -- reuses the supply depot's crate look.
  for (let i = 0; i < 3; i++) {
    const crate = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.6, 0.6), MIL_CRATE_MAT);
    crate.position.set(-2.5 - i * 0.5, 0.3, 2.6);
    crate.castShadow = true;
    g.add(crate);
  }
  g.position.set(cx, 0, cz);
  g.rotation.y = angle;
  parent.add(g);
  return makeGunRig(barrel, 3.3, 1.0, 13, `artillery|${tag}|${cx}|${cz}`);
}

// r=9 with 1.6-scale "H" markings was sized before mobile units got their
// 10x scale-up -- a landing helicopter is now ~90 units long, more than
// the old pad's full diameter. Scaled 6x (s) to actually fit one.
function addHelipad(parent, cx, cz) {
  const s = 6;
  const r = 9 * s;
  const pad = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 0.2, 28), MIL_PAD_MAT);
  pad.position.set(cx, 0.1, cz);
  pad.receiveShadow = true;
  parent.add(pad);
  for (const dx of [-1.6 * s, 1.6 * s]) {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.5 * s, 0.05, 4.5 * s), MIL_PAD_MARK_MAT);
    bar.position.set(cx + dx, 0.22, cz);
    parent.add(bar);
  }
  const crossBar = new THREE.Mesh(new THREE.BoxGeometry(3.7 * s, 0.05, 0.5 * s), MIL_PAD_MARK_MAT);
  crossBar.position.set(cx, 0.22, cz);
  parent.add(crossBar);
}

// Radio/comm tower -- a tall slender 4-legged lattice mast (not a
// watchtower's stubby platform) topped with a dish and a blinking
// obstruction light, the tallest single silhouette on the base and a
// distinct landmark from the corner/bunker watchtowers.
const MIL_RADIO_MAT = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.6, color: 0x4a4a42 });
const MIL_DISH_MAT = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.6, color: 0xb8b4a8 });
const MIL_LIGHT_MAT = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.4, color: 0xff4433, emissive: 0xb01f14, emissiveIntensity: 0.7 });
function addRadioTower(parent, x, z, h) {
  const g = new THREE.Group();
  // Base width/leg thickness/dish size all scale off `h` relative to the
  // original baseline height of 32 -- same "a bigger tower needs a
  // bigger footprint, not just more height" fix as addWatchtower.
  const s = h / 32;
  const legOffset = 1.15 * s;
  const legPos = [[-legOffset, -legOffset], [legOffset, -legOffset], [-legOffset, legOffset], [legOffset, legOffset]];
  const legGeo = new THREE.CylinderGeometry(0.11 * s, 0.2 * s, h, 5);
  for (const [dx, dz] of legPos) {
    const leg = new THREE.Mesh(legGeo, MIL_RADIO_MAT);
    leg.position.set(dx * 1.4, h / 2, dz * 1.4);
    leg.rotation.set(dz > 0 ? 0.03 : -0.03, 0, dx > 0 ? -0.03 : 0.03);
    leg.castShadow = true;
    g.add(leg);
  }
  function brace(a, b, yFrac) {
    const y = h * yFrac;
    const dx = (b[0] - a[0]) * 1.4, dz = (b[1] - a[1]) * 1.4;
    const len = Math.hypot(dx, dz);
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.05 * s, 0.05 * s, len, 4), MIL_RADIO_MAT);
    m.position.set((a[0] + b[0]) * 0.7, y, (a[1] + b[1]) * 0.7);
    m.rotation.z = Math.PI / 2;
    m.rotation.y = -Math.atan2(dz, dx);
    g.add(m);
  }
  for (const yFrac of [0.18, 0.38, 0.58, 0.78]) {
    brace(legPos[0], legPos[1], yFrac); brace(legPos[2], legPos[3], yFrac);
    brace(legPos[0], legPos[2], yFrac); brace(legPos[1], legPos[3], yFrac);
  }
  const dish = new THREE.Mesh(new THREE.CylinderGeometry(1.5 * s, 1.5 * s, 0.2 * s, 12, 1, true), MIL_DISH_MAT);
  dish.rotation.x = Math.PI / 2;
  dish.rotation.z = 0.55;
  dish.position.set(0, h * 0.74, 0);
  g.add(dish);
  const whip = new THREE.Mesh(new THREE.CylinderGeometry(0.04 * s, 0.04 * s, h * 0.14, 4), MIL_RADIO_MAT);
  whip.position.y = h + (h * 0.07);
  g.add(whip);
  const light = new THREE.Mesh(new THREE.SphereGeometry(0.28 * s, 6, 6), MIL_LIGHT_MAT);
  light.position.y = h + (h * 0.14);
  g.add(light);
  g.position.set(x, 0, z);
  parent.add(g);
}

// Bridge over a shallow decorative ravine -- a classic battlefield
// chokepoint landmark, purely visual (no obstacle, nothing needs it to
// cross anything for PHY purposes).
const MIL_RAVINE_MAT = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, color: 0x473d2d });
const MIL_BRIDGE_DECK_MAT = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0, color: 0x6b5a3c });
const MIL_BRIDGE_RAIL_MAT = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.5, color: 0x3d3a2e });
function addRavineBridge(parent, cx, cz, angle, span) {
  const g = new THREE.Group();
  const ravine = new THREE.Mesh(new THREE.BoxGeometry(span * 1.15, 1.4, 13), MIL_RAVINE_MAT);
  ravine.position.y = -0.75;
  ravine.receiveShadow = true;
  g.add(ravine);
  const deck = new THREE.Mesh(new THREE.BoxGeometry(span, 0.35, 5.6), MIL_BRIDGE_DECK_MAT);
  deck.position.y = 0.05;
  deck.castShadow = true;
  deck.receiveShadow = true;
  g.add(deck);
  const plankGeo = new THREE.BoxGeometry(0.35, 0.06, 5.7);
  const nPlanks = Math.max(6, Math.round(span / 1.6));
  for (let i = 0; i < nPlanks; i++) {
    const px = -span / 2 + (span / nPlanks) * (i + 0.5);
    const plank = new THREE.Mesh(plankGeo, MIL_BRIDGE_RAIL_MAT);
    plank.position.set(px, 0.24, 0);
    g.add(plank);
  }
  for (const side of [-1, 1]) {
    const rail = new THREE.Mesh(new THREE.BoxGeometry(span, 0.7, 0.15), MIL_BRIDGE_RAIL_MAT);
    rail.position.set(0, 0.55, side * 2.65);
    g.add(rail);
    const nPosts = Math.max(4, Math.round(span / 4.5));
    for (let i = 0; i < nPosts; i++) {
      const px = -span / 2 + (span / (nPosts - 1)) * i;
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.9, 5), MIL_BRIDGE_RAIL_MAT);
      post.position.set(px, 0.5, side * 2.65);
      g.add(post);
    }
  }
  for (const px of [-span * 0.28, 0, span * 0.28]) {
    const pier = new THREE.Mesh(new THREE.BoxGeometry(0.7, 1.5, 5.2), MIL_BRIDGE_DECK_MAT);
    pier.position.set(px, -0.65, 0);
    pier.castShadow = true;
    g.add(pier);
  }
  g.rotation.y = angle;
  g.position.set(cx, 0, cz);
  parent.add(g);
}

// ---- terrain filler: 802.11ah's real long range routinely scatters
// STAs (and so the perimeter, which is sized off them -- see above)
// hundreds of metres out, which left the ground between the base cluster
// and the far nodes looking bare. Everything here is drawn as a small
// number of THREE.InstancedMesh calls (one draw call per prop TYPE,
// regardless of how many hundreds/thousands of instances of it exist) so
// density can go up aggressively without turning into hundreds of
// individual draw calls -- ground patches, rocks, scrub and craters each
// get their own shared unit geometry, scaled/rotated/positioned per
// instance via a transform matrix (and, for ground patches, tinted per
// instance via instanceColor so one mesh still reads as multi-tone camo).
const MIL_CRATER_MAT = new THREE.MeshStandardMaterial({ roughness: 0.88, metalness: 0, color: 0x5c5540 });
const MIL_ROCK_MAT = new THREE.MeshStandardMaterial({ roughness: 0.88, metalness: 0, color: 0x6b6558 });
const MIL_SCRUB_MAT = new THREE.MeshStandardMaterial({ roughness: 0.88, metalness: 0, color: 0x585a3f });
// Opaque, not transparent -- an earlier transparent+depthWrite:false
// version fixed a z-fighting flicker, but blending ~1000 overlapping
// instances is real GPU overdraw cost, and it's unnecessary now the
// patches are large organic blobs rather than small tiles needing soft
// edges to blend together. Opaque quads get early-Z rejection instead.
// polygonOffset -- these patches sit only ~0.03-0.07 units above the
// ground plane, and at this scene's scale (viewed from thousands of
// units out, bases 2km apart) that tiny gap isn't enough depth-buffer
// precision headroom on its own: without this they intermittently
// z-fight/flicker against the ground depending on camera distance,
// reading as "unstable" graphics rather than a steady painted patch.
const MIL_GROUND_PATCH_MAT = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, 
  color: 0xffffff, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
});
// Was a 5-colour "camo" palette spanning dirt/scrub/khaki -- read as a
// busy printed camouflage pattern rather than natural ground, which is
// what prompted swapping the whole look for a plain forest floor: just
// two close, muted mossy-green tones instead of a blotchy multi-colour mix.
const MIL_GROUND_PATCH_COLORS = [0x5c7a3e, 0x6b8f48]
  .map((c) => new THREE.Color(c));
const UNIT_BOX_GEO = new THREE.BoxGeometry(1, 1, 1);
const UNIT_CRATER_GEO = new THREE.CylinderGeometry(1, 0.8, 0.15, 10);
// detail=1 (not 0) -- a plain DodecahedronGeometry(1,0) is only 12 flat
// pentagon faces, which at this non-uniform instance scale read as sharp
// spikes rather than boulders; one subdivision level gives enough extra
// facets to look like an actual rounded rock.
const UNIT_ROCK_GEO = new THREE.DodecahedronGeometry(1, 1);
const UNIT_SCRUB_GEO = new THREE.IcosahedronGeometry(1, 0);
// Simple forest trees -- two InstancedMesh calls (trunk cylinder + cone
// canopy) sharing the same per-instance positions, same low-draw-call
// pattern as the rock/scrub/crater filler below.
const MIL_TRUNK_MAT = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0x4a3826 });
const MIL_CANOPY_MAT = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, color: 0xffffff });
const MIL_CANOPY_COLORS = [0x35492a, 0x3f5531, 0x2e4025]
  .map((c) => new THREE.Color(c));
const UNIT_TRUNK_GEO = new THREE.CylinderGeometry(0.4, 0.55, 1, 6);
const UNIT_CANOPY_GEO = new THREE.ConeGeometry(1, 1, 7);

// Irregular "blob" fan (a jittered-radius polygon, not a plain rectangle)
// for the camo ground patches -- a rotated PlaneGeometry has a hard,
// obviously-rectangular silhouette ("pasted sticker" look) at close range;
// this reads as an organic terrain blotch instead, closer to the 2D
// canvas's own hand-drawn irregular camo polygons. Built once with a
// fixed local seed (shape variety then comes from each instance's own
// non-uniform scale/rotation, same pattern as the rock/scrub geometries).
function makeBlobGeometry(segments, jitter, seed) {
  const rng = mulberry32(seed);
  const positions = [0, 0, 0];
  const uvs = [0.5, 0.5];
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    const r = 1 - jitter + rng() * jitter * 2;
    const x = Math.cos(a) * r, y = Math.sin(a) * r;
    positions.push(x, y, 0);
    uvs.push(x * 0.5 + 0.5, y * 0.5 + 0.5);
  }
  const indices = [];
  for (let i = 1; i <= segments; i++) indices.push(0, i, i + 1);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}
const UNIT_PATCH_GEO = makeBlobGeometry(9, 0.4, 0xca0170);

// True if (x,z) is within `margin` of a rectangle's OUTLINE (not just
// anywhere inside it) -- used to keep solid filler props (rocks etc, NOT
// the flat ground camo -- driving/walking across a painted ground patch
// is fine, clipping through a rock isn't) off the exact path the tank and
// soldiers patrol, see stepMilitaryPatrol's use of rectLoopPos for that
// same rectangle outline.
function nearRectOutline(x, z, hw, hh, margin) {
  const dx = Math.abs(x) - hw;
  const dz = Math.abs(z) - hh;
  if (dx < 0 && dz < 0) return Math.min(-dx, -dz) < margin;
  const ddx = Math.max(dx, 0), ddz = Math.max(dz, 0);
  return Math.hypot(ddx, ddz) < margin;
}

// Distance from (x,z) to the nearest point on segment (x0,y0)-(x1,y1) --
// used to keep filler props off the convoy corridor between the two
// bases (see rebuildMilitaryPatrol's convoyLoops), the straight-line
// counterpart to nearRectOutline's local patrol rings.
function nearLineSegment(x, z, x0, y0, x1, y1, margin) {
  const dx = x1 - x0, dy = y1 - y0;
  const lenSq = dx * dx + dy * dy || 1;
  let t = ((x - x0) * dx + (z - y0) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  const px = x0 + dx * t, py = y0 + dy * t;
  return Math.hypot(x - px, z - py) < margin;
}

function scatterPositions(rng, hw, hd, keepOut, count, avoidLoops, avoidLines) {
  const pts = [];
  for (let i = 0; i < count; i++) {
    const x = (rng() * 2 - 1) * hw * 0.97;
    const z = (rng() * 2 - 1) * hd * 0.97;
    if (Math.hypot(x, z) < keepOut) continue;
    // Margin scales with MIL_UNIT_SCALE -- a tank scaled up 10x is ~64m
    // long, so the old fixed 11m clearance around the patrol path was
    // barely a third of the vehicle's own length and let props clip
    // straight through it.
    if (avoidLoops && avoidLoops.some((l) => nearRectOutline(x, z, l.hw, l.hh, 11 * (MIL_UNIT_SCALE / 3)))) continue;
    if (avoidLines && avoidLines.some((l) => nearLineSegment(x, z, l.x0, l.y0, l.x1, l.y1, l.margin))) continue;
    pts.push([x, z]);
  }
  return pts;
}

const dummy = new THREE.Object3D();

function addInstancedCraters(parent, rng, hw, hd, keepOut, count, avoidLoops, avoidLines) {
  const pts = scatterPositions(rng, hw, hd, keepOut, count, avoidLoops, avoidLines);
  const mesh = new THREE.InstancedMesh(UNIT_CRATER_GEO, MIL_CRATER_MAT, pts.length);
  mesh.frustumCulled = false;
  mesh.receiveShadow = true;
  pts.forEach(([x, z], i) => {
    const r = 2.2 + rng() * 4.5;
    dummy.position.set(x, 0.02, z);
    dummy.rotation.set(0, rng() * Math.PI, 0);
    dummy.scale.set(r, 1, r);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  });
  mesh.instanceMatrix.needsUpdate = true;
  parent.add(mesh);
}

function addInstancedRocks(parent, rng, hw, hd, keepOut, count, avoidLoops, avoidLines) {
  const pts = scatterPositions(rng, hw, hd, keepOut, count, avoidLoops, avoidLines);
  const mesh = new THREE.InstancedMesh(UNIT_ROCK_GEO, MIL_ROCK_MAT, pts.length);
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  pts.forEach(([x, z], i) => {
    const s = 1.1 + rng() * 2.6;
    dummy.position.set(x, s * 0.4, z);
    dummy.rotation.set(rng() * Math.PI, rng() * Math.PI, rng() * Math.PI);
    dummy.scale.set(s, s * (0.7 + rng() * 0.5), s);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  });
  mesh.instanceMatrix.needsUpdate = true;
  parent.add(mesh);
}

function addInstancedScrub(parent, rng, hw, hd, keepOut, count, avoidLoops, avoidLines) {
  const pts = scatterPositions(rng, hw, hd, keepOut, count, avoidLoops, avoidLines);
  const mesh = new THREE.InstancedMesh(UNIT_SCRUB_GEO, MIL_SCRUB_MAT, pts.length);
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  pts.forEach(([x, z], i) => {
    const s = 0.7 + rng() * 1.3;
    dummy.position.set(x, s * 0.55, z);
    dummy.rotation.set(0, rng() * Math.PI, 0);
    dummy.scale.set(s, s * (0.5 + rng() * 0.4), s);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  });
  mesh.instanceMatrix.needsUpdate = true;
  parent.add(mesh);
}

function addInstancedForest(parent, rng, hw, hd, keepOut, count, avoidLoops, avoidLines) {
  const pts = scatterPositions(rng, hw, hd, keepOut, count, avoidLoops, avoidLines);
  const trunks = new THREE.InstancedMesh(UNIT_TRUNK_GEO, MIL_TRUNK_MAT, pts.length);
  const canopies = new THREE.InstancedMesh(UNIT_CANOPY_GEO, MIL_CANOPY_MAT, pts.length);
  trunks.frustumCulled = false;
  canopies.frustumCulled = false;
  trunks.castShadow = true;
  trunks.receiveShadow = true;
  canopies.castShadow = true;
  pts.forEach(([x, z], i) => {
    // Raised from an earlier 12-26 baseline -- next to 26-85-unit-tall
    // towers, those trees read as shrubs. Kept mostly under the corner
    // watchtowers (42) so towers still break the treeline as landmarks.
    const h = 18 + rng() * 20;
    const trunkR = 0.9 + rng() * 0.7;
    dummy.position.set(x, h * 0.5, z);
    dummy.rotation.set(0, rng() * Math.PI, 0);
    dummy.scale.set(trunkR, h, trunkR);
    dummy.updateMatrix();
    trunks.setMatrixAt(i, dummy.matrix);

    const canopyH = h * (0.85 + rng() * 0.35);
    const canopyR = h * (0.28 + rng() * 0.12);
    dummy.position.set(x, h + canopyH * 0.45, z);
    dummy.rotation.set(0, rng() * Math.PI, 0);
    dummy.scale.set(canopyR, canopyH, canopyR);
    dummy.updateMatrix();
    canopies.setMatrixAt(i, dummy.matrix);
    canopies.setColorAt(i, MIL_CANOPY_COLORS[(rng() * MIL_CANOPY_COLORS.length) | 0]);
  });
  trunks.instanceMatrix.needsUpdate = true;
  canopies.instanceMatrix.needsUpdate = true;
  if (canopies.instanceColor) canopies.instanceColor.needsUpdate = true;
  parent.add(trunks);
  parent.add(canopies);
}

// Plain forest-floor tint laid flat on the ground -- unlike small filler
// props, this reads as colour from ANY camera distance, including the
// very-far-out default framing a widely-scattered 802.11ah topology
// forces (small props shrink to sub-pixel out there and stop reading as
// "content" at all; a big tinted patch never does). Two close mossy-green
// tones only, not a multi-colour camo blotch -- kept simple on purpose.
function addMilitaryGroundCamo(parent, rng, hw, hd, count) {
  const mesh = new THREE.InstancedMesh(UNIT_PATCH_GEO, MIL_GROUND_PATCH_MAT, count);
  mesh.frustumCulled = false;
  mesh.receiveShadow = true;
  for (let i = 0; i < count; i++) {
    const x = (rng() * 2 - 1) * hw * 0.98;
    const z = (rng() * 2 - 1) * hd * 0.98;
    // Large, slow-varying terrain zones (tens to well over a hundred
    // metres) rather than small camo-net-scale speckles -- this reads as
    // natural ground cover variation (dirt/scrub/dry grass) at the scale
    // this map is actually viewed from, not a printed camouflage pattern.
    const sx = 55 + rng() * 140;
    const sz = sx * (0.55 + rng() * 0.75);
    dummy.position.set(x, 0.03 + rng() * 0.04, z);
    dummy.rotation.set(-Math.PI / 2, 0, rng() * Math.PI);
    dummy.scale.set(sx, sz, 1);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    mesh.setColorAt(i, MIL_GROUND_PATCH_COLORS[(rng() * MIL_GROUND_PATCH_COLORS.length) | 0]);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  parent.add(mesh);
}

// Charred, flat decals -- same organic blob shape and lay-flat convention
// as the ground camo patches (including polygonOffset, for the same
// z-fighting-at-distance reason), just near-black and semi-transparent so
// they read as scorched ground instead of a solid black sticker.
const MIL_SCORCH_MAT = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, 
  color: 0x161410, transparent: true, opacity: 0.6,
  polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
});
function addInstancedScorchMarks(parent, rng, hw, hd, keepOut, count, avoidLoops, avoidLines) {
  const pts = scatterPositions(rng, hw, hd, keepOut, count, avoidLoops, avoidLines);
  const mesh = new THREE.InstancedMesh(UNIT_PATCH_GEO, MIL_SCORCH_MAT, pts.length);
  mesh.frustumCulled = false;
  pts.forEach(([x, z], i) => {
    const r = 3 + rng() * 5;
    dummy.position.set(x, 0.05, z);
    dummy.rotation.set(-Math.PI / 2, 0, rng() * Math.PI);
    dummy.scale.set(r, r * (0.6 + rng() * 0.5), 1);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  });
  mesh.instanceMatrix.needsUpdate = true;
  parent.add(mesh);
}

// Small ponds/shell-crater pools -- same organic blob + lay-flat-with-
// polygonOffset convention as the ground camo/scorch decals above, just a
// muddy rim ring under a flat water-blue fill so a battlefield with
// natural low ground reads as terrain, not one uniform dirt field.
// MeshBasicMaterial (not Lambert) on purpose -- ignores scene lighting
// entirely, same choice the Smart City park pond makes, so the water stays
// a clean, saturated blue instead of getting muddied by the Zone's warm
// sun tint like everything else here.
const MIL_WATER_MAT = new THREE.MeshBasicMaterial({
  color: 0x3f7fa6,
  polygonOffset: true, polygonOffsetFactor: -6, polygonOffsetUnits: -6,
});
const MIL_MUD_RIM_MAT = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, 
  color: 0x4a3f28,
  polygonOffset: true, polygonOffsetFactor: -5, polygonOffsetUnits: -5,
});
function addMilitaryWaterBody(parent, x, z, rx, rz, rng) {
  const rim = new THREE.Mesh(UNIT_PATCH_GEO, MIL_MUD_RIM_MAT);
  rim.rotation.set(-Math.PI / 2, 0, rng() * Math.PI);
  rim.position.set(x, 0.05, z);
  rim.scale.set(rx * 1.2, rz * 1.2, 1);
  rim.receiveShadow = true;
  parent.add(rim);

  const water = new THREE.Mesh(UNIT_PATCH_GEO, MIL_WATER_MAT);
  water.rotation.set(-Math.PI / 2, 0, rng() * Math.PI);
  water.position.set(x, 0.09, z);
  water.scale.set(rx, rz, 1);
  parent.add(water);
}

// Small scattered battle debris -- crates, tipped oil drums, sheared
// scrap plate -- distinct from the rocks/scrub/craters above (those read
// as terrain, these read as things someone left behind or blew apart).
// Same InstancedMesh-per-type approach as the rock/scrub/crater filler:
// one shared geometry+material per debris flavour, transform-only per
// instance. Pool sized to `count` up front and trimmed via `.count` after
// scattering, since each instance is randomly assigned one of the three
// flavours rather than split into separate scatterPositions calls.
// Crate/drum materials reused from the field-camp props above (same
// wood-crate/fuel-drum look, just loose on the ground instead of stacked
// at a camp) -- only scrap plate needs a new material.
const MIL_SCRAP_MAT = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.55, color: 0x46484a });
const UNIT_DEBRIS_DRUM_GEO = new THREE.CylinderGeometry(1, 1, 1.6, 10);
function addInstancedDebris(parent, rng, hw, hd, keepOut, count, avoidLoops, avoidLines) {
  const pts = scatterPositions(rng, hw, hd, keepOut, count, avoidLoops, avoidLines);
  const crates = new THREE.InstancedMesh(UNIT_BOX_GEO, MIL_CRATE_MAT, pts.length);
  const drums = new THREE.InstancedMesh(UNIT_DEBRIS_DRUM_GEO, MIL_DRUM_MAT, pts.length);
  const scraps = new THREE.InstancedMesh(UNIT_BOX_GEO, MIL_SCRAP_MAT, pts.length);
  for (const m of [crates, drums, scraps]) { m.frustumCulled = false; m.castShadow = true; m.receiveShadow = true; }
  let nCrate = 0, nDrum = 0, nScrap = 0;
  pts.forEach(([x, z]) => {
    const pick = rng();
    if (pick < 0.38) {
      const s = 1.0 + rng() * 0.8;
      dummy.position.set(x, s * 0.5, z);
      dummy.rotation.set(0, rng() * Math.PI, (rng() - 0.5) * 0.5);
      dummy.scale.set(s, s * (0.7 + rng() * 0.4), s * (0.8 + rng() * 0.4));
      dummy.updateMatrix();
      crates.setMatrixAt(nCrate++, dummy.matrix);
    } else if (pick < 0.7) {
      const s = 0.85 + rng() * 0.45;
      const tipped = rng() < 0.5;
      dummy.position.set(x, tipped ? s * 0.5 : s * 0.8, z);
      dummy.rotation.set(tipped ? Math.PI / 2 : 0, rng() * Math.PI, tipped ? rng() * Math.PI : 0);
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      drums.setMatrixAt(nDrum++, dummy.matrix);
    } else {
      const sx = 0.7 + rng() * 1.4, sy = 0.06 + rng() * 0.08, sz = 0.5 + rng() * 1.0;
      dummy.position.set(x, sy * 2, z);
      dummy.rotation.set((rng() - 0.5) * 0.6, rng() * Math.PI, (rng() - 0.5) * 0.6);
      dummy.scale.set(sx, sy, sz);
      dummy.updateMatrix();
      scraps.setMatrixAt(nScrap++, dummy.matrix);
    }
  });
  crates.count = nCrate; drums.count = nDrum; scraps.count = nScrap;
  crates.instanceMatrix.needsUpdate = true;
  drums.instanceMatrix.needsUpdate = true;
  scraps.instanceMatrix.needsUpdate = true;
  parent.add(crates, drums, scraps);
}

// A knocked-out tank/truck -- reuses the live vehicle's own builder (same
// silhouette/scale as the moving ones) but recolours every mesh charred
// black, tilts it like it's dug into a shell crater, and gives it a slow
// rising smoke wisp via the same emitter pool the industrial smokestacks
// use, for a "still smouldering" read instead of pristine wreckage.
const MIL_WRECK_MAT = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.5, color: 0x2a2621 });
function addVehicleWreck(parent, x, z, rng, isTank) {
  const built = isTank ? buildMilitaryTankMesh(rng) : buildMilitaryTruckMesh(rng);
  const g = built.group;
  g.traverse((o) => { if (o.isMesh) o.material = MIL_WRECK_MAT; });
  g.rotation.y = rng() * Math.PI * 2;
  g.rotation.z = (rng() - 0.5) * 0.4;
  g.position.set(x, 0, z);
  parent.add(g);
  registerSmokestack({ x, y: 12, z });
}

function addFillerOutpost(parent, x, z, rng) {
  const r = 2.0 + rng() * 0.6;
  const n = 6;
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    const bag = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.5, 0.5), MIL_SANDBAG_MAT);
    bag.position.set(x + Math.cos(a) * r, 0.28, z + Math.sin(a) * r);
    bag.rotation.y = a;
    bag.castShadow = true;
    parent.add(bag);
  }
}

function addMilitaryFiller(parent, rng, hw, hd, loops, avoidLines) {
  const area = hw * hd;
  const keepOut = 68; // radius around the origin where the command bunker cluster already sits
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, Math.round(v)));

  // Ground camo intentionally ignores `loops`/`avoidLines` -- it's flat
  // paint on the terrain, patrol units driving/walking over it is fine
  // and expected. Everything below is a solid prop, which would
  // otherwise sometimes spawn a rock or bush directly on the exact line
  // a tank/soldier/convoy walks.
  // Counts trimmed from earlier, denser passes -- the mobile units (tanks/
  // planes/soldiers) are now scaled 10x and are the actual visual focus,
  // so terrain filler doesn't need to carry the whole "does this look
  // full" job by itself any more, and a real browser (not the headless/
  // software-rendered profile used to check this visually) was measured
  // averaging ~9 FPS with the previous, much higher, counts -- the janky,
  // stuttering motion that produces reads as "unstable"/broken animation
  // even though the animation code itself is fine.
  // Eased back down from a "doubled" pass -- per explicit request for a
  // sparser, less cluttered feel now that the mobile units themselves
  // spread across the whole area (see groundSpread/airSpread), the
  // terrain filler doesn't need to be this dense to avoid looking empty.
  // The base ground plane itself now carries a tinted grass texture (see
  // scene.js's ENV_GROUND_TINT), so these patches only need to add subtle
  // tonal variation on top rather than provide all the ground's colour --
  // an InstancedMesh, so the count itself is nearly free either way.
  addMilitaryGroundCamo(parent, rng, hw, hd, clamp(area / 11000, 70, 280));
  addInstancedCraters(parent, rng, hw, hd, keepOut, clamp(area / 9000, 30, 220), loops, avoidLines);
  addInstancedRocks(parent, rng, hw, hd, keepOut, clamp(area / 7000, 40, 300), loops, avoidLines);
  addInstancedScrub(parent, rng, hw, hd, keepOut, clamp(area / 5500, 50, 340), loops, avoidLines);
  addInstancedForest(parent, rng, hw, hd, keepOut, clamp(area / 4500, 70, 420), loops, avoidLines);

  // Ponds/shell-crater pools -- small in number (each is 2 individual,
  // non-instanced meshes), scattered like the outposts/wrecks below rather
  // than instanced, so a wide, sparse field reads as having real low
  // ground instead of being uniformly dry.
  const nPonds = clamp(area / 220000, 2, 5);
  for (const [x, z] of scatterPositions(rng, hw, hd, keepOut, nPonds, loops, avoidLines)) {
    const rx = 16 + rng() * 24;
    const rz = rx * (0.6 + rng() * 0.6);
    addMilitaryWaterBody(parent, x, z, rx, rz, rng);
  }

  // Sparse on purpose -- real battle damage is localised, not blanket
  // coverage; too many of these reads as "the ground is dirty" rather
  // than "fighting happened here".
  addInstancedScorchMarks(parent, rng, hw, hd, keepOut, clamp(area / 45000, 10, 40), loops, avoidLines);
  addInstancedDebris(parent, rng, hw, hd, keepOut, clamp(area / 26000, 20, 90), loops, avoidLines);

  // Outposts aren't instanced (6 individual sandbag boxes each), so their
  // count matters a lot more for draw calls than the instanced props above.
  const nOutposts = clamp(area / 150000, 4, 14);
  for (const [x, z] of scatterPositions(rng, hw, hd, keepOut, nOutposts, loops, avoidLines)) {
    addFillerOutpost(parent, x, z, rng);
  }

  // A handful of knocked-out vehicles -- capped low, each is a full
  // ~20-mesh non-instanced group same as an active tank/truck.
  const nWrecks = clamp(area / 350000, 2, 5);
  for (const [x, z] of scatterPositions(rng, hw, hd, keepOut, nWrecks, loops, avoidLines)) {
    addVehicleWreck(parent, x, z, rng, rng() < 0.6);
  }

  // Extra standalone lookout towers scattered across the wider terrain --
  // each is a full non-instanced ~23-mesh Group (legs/braces/deck/rails),
  // so this was the single biggest draw-call cost in the whole scene at
  // the old cap of up to 34 (~780 draw calls just for towers). The base
  // still has its corner/bunker/radio towers; this only adds more spread
  // further out.
  // Each tower is 15 individual (non-instanced) meshes, so this count
  // matters a lot for draw calls -- raised from the old 3-9 cap for
  // "add towers", but capped well short of the 22 that made the scene
  // stutter (~800+ extra draw calls across both bases at that count).
  const nExtraTowers = clamp(area / 130000, 6, 14);
  for (const [x, z] of scatterPositions(rng, hw, hd, keepOut, nExtraTowers, loops, avoidLines)) {
    addWatchtower(parent, x, z, 26 + rng() * 12);
  }
}

let militaryMeshes = [];
let militaryLoops = [];
let militaryAirLoops = [];
let militaryConvoyLoops = [];
let militaryAirConvoyLoops = [];
let militaryStatics = [];
// Mortar pit / artillery gun firing rigs (see makeGunRig) -- these are
// static props, not militaryMeshes members, but their flash sprite/light/
// barrel mesh are all descendants of a militaryStatics-tracked group, so
// the generic traverse+dispose loop below already frees them; this array
// just needs to be dropped in step so stepMilitaryPatrol stops iterating
// stale (removed) gun references.
let militaryGuns = [];
let lastMilitaryKey = '';
// Current base extent + whether Military Zone is the active environment --
// set every poll from rebuildMilitaryPatrol (both branches), read by the
// ambient bomb-effect spawner (see stepMilitaryPatrol) to know where it's
// allowed to scatter impacts and whether it should be spawning new ones at
// all. MIL_BASE_OFFSET_X/Y mirror the OFF_X/OFF_Y rebuildMilitaryPatrol
// itself uses to place the second base -- hoisted to module scope (instead
// of staying a rebuildMilitaryPatrol-local const) purely so this same
// value doesn't need duplicating here. `let`, not `const` -- 2000 is only
// the default for a "normal"-range PHY config; rebuildMilitaryPatrol grows
// it dynamically (see its own comment there) once a base's own hw/hd
// footprint (scales with the AP's real coverage radius, see
// militaryExtent) gets close enough to that fixed 2000 gap that the two
// bases would start to interpenetrate for a longer-range PHY config.
let MIL_BASE_OFFSET_X = 2000, MIL_BASE_OFFSET_Y = 0;
let militaryFieldHw = 0, militaryFieldHd = 0;
let militaryZoneActive = false;
function clearMilitaryStatics() {
  for (const m of militaryStatics) {
    buildingsGroup.remove(m);
    m.traverse((o) => {
      o.geometry?.dispose?.();
      const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
      for (const mm of mats) mm.dispose?.();
    });
  }
  militaryStatics = [];
  militaryGuns = [];
}
// Spawns `count` units across several loops (indices into `loopSet`, e.g.
// militaryLoops or militaryAirLoops), round-robining so each loop gets an
// even share instead of piling every unit onto one ring -- a real
// battlefield has units spread across the whole area, not one conga line.
// `lateralSpread`, if given, assigns each unit its own random offset
// perpendicular to its direction of travel (applied in stepMilitaryPatrol
// via the shared squadOffset field) so a whole group fans out across the
// width of the area instead of riding single-file along one line.
// Deterministic EVENLY-SPACED slots (plus small jitter), not independent
// random offsets -- independent randomness can by chance put two units
// close enough to overlap; evenly spacing them across the band guarantees
// a minimum gap between every pair in the group. `laneCenter` shifts the
// whole band so different unit TYPES (trucks/tanks/soldiers/planes) get
// non-overlapping bands instead of all sharing the same centred range.
function spawnPatrolGroup(rng, loopIndices, count, build, lateralSpread, laneCenter) {
  const perLoop = Math.ceil(count / loopIndices.length);
  const counters = loopIndices.map(() => 0);
  const out = [];
  const jitter = lateralSpread ? (lateralSpread / Math.max(1, count)) * 0.4 : 0;
  for (let i = 0; i < count; i++) {
    const li = i % loopIndices.length;
    const built = build(rng, i);
    vehiclesGroup.add(built.group);
    const slot = count > 1 ? (i / (count - 1)) * 2 - 1 : 0;
    const lateral = (laneCenter || 0) + (lateralSpread ? slot * lateralSpread + (rng() * 2 - 1) * jitter : 0);
    out.push({
      ...built, loopIdx: loopIndices[li],
      dir: counters[li] % 2 ? -1 : 1, phase: counters[li] / perLoop,
      squadOffset: lateral,
    });
    counters[li]++;
  }
  return out;
}
// Per-member (lateral, forward) offsets for a squad's formation shape, in
// the squad's own travel-relative frame (lateral = perpendicular to
// heading, forward = along it) -- consumed by spawnSquads below via the
// existing squadOffset/squadForwardOffset fields stepMilitaryPatrol
// already applies, so no new per-frame code is needed to support these:
// only how the offsets are computed at spawn time changes.
function squadLineOffsets(size, gap) {
  const offs = [];
  for (let i = 0; i < size; i++) offs.push([0, (i - (size - 1) / 2) * gap]);
  return offs;
}
// Infantry wedge -- a single point soldier out front, each row behind one
// wider, the classic "V" a real squad forms while advancing together.
// Rows fill left-to-right so the shape stays a clean triangle even when
// `size` doesn't end exactly on a row boundary.
function squadTriangleOffsets(size, gap) {
  const offs = [];
  let row = 0;
  while (offs.length < size) {
    const rowSize = Math.min(row + 1, size - offs.length);
    for (let c = 0; c < rowSize; c++) {
      offs.push([(c - (rowSize - 1) / 2) * gap, -row * gap * 0.9]);
    }
    row++;
  }
  return offs;
}
// Roughly square block/grid -- cols~=sqrt(size), reads as a formed-up unit
// marching in ranks rather than a loose crowd.
function squadSquareOffsets(size, gap) {
  const cols = Math.max(1, Math.round(Math.sqrt(size)));
  const rows = Math.ceil(size / cols);
  const offs = [];
  for (let i = 0; i < size; i++) {
    const row = Math.floor(i / cols), col = i % cols;
    offs.push([(col - (cols - 1) / 2) * gap, (row - (rows - 1) / 2) * gap]);
  }
  return offs;
}
// Evenly spaced ring -- used both for squads marching in a defensive
// perimeter shape and (via a tighter gap, see spawnStandingGroups) for
// stationary huddles facing inward.
function squadCircleOffsets(size, gap) {
  const r = Math.max(gap * 0.5, (gap * size) / (2 * Math.PI));
  const offs = [];
  for (let i = 0; i < size; i++) {
    const a = (i / size) * Math.PI * 2;
    offs.push([Math.cos(a) * r, Math.sin(a) * r]);
  }
  return offs;
}
function squadFormationOffsets(formation, size, gap) {
  if (formation === 'triangle') return squadTriangleOffsets(size, gap);
  if (formation === 'square') return squadSquareOffsets(size, gap);
  if (formation === 'circle') return squadCircleOffsets(size, gap);
  return squadLineOffsets(size, gap);
}

// Same idea as spawnPatrolGroup, but spawns SQUADS: squadSize units that
// share one loopIdx/dir/phase (so they move together as a unit, not
// independently scattered around the ring) plus a `squadOffset` -- a
// lateral distance from the squad's centre line, applied perpendicular to
// the direction of travel in stepMilitaryPatrol -- so a squad reads as an
// organized formation moving together, not a loose mob. `formations`, if
// given, is a list of shape names (see squadFormationOffsets) cycled
// across squads round-robin -- e.g. ['line','triangle','circle','square']
// gives an even mix rather than every squad using the same shape.
function spawnSquads(rng, loopIndices, squadCount, squadSize, build, lateralSpread, laneCenter, formations) {
  const perLoop = Math.ceil(squadCount / loopIndices.length);
  const counters = loopIndices.map(() => 0);
  const out = [];
  // fileGap doubles as the general member-spacing unit for every
  // formation shape below, not just the single-file line -- small lateral
  // wobble on top (well under one gap, so nobody can end up overlapping a
  // squad-mate) keeps every shape from reading as a perfectly rigid rail.
  const fileGap = MIL_SOLDIER_SCALE * 2.2;
  const fileWobble = MIL_SOLDIER_SCALE * 0.35;
  // Each squad's OWN centreline is an evenly-spaced slot across the band
  // (same reasoning as spawnPatrolGroup) so no two squads' formation
  // blocks can ever be assigned overlapping territory.
  const squadJitter = lateralSpread ? (lateralSpread / Math.max(1, squadCount)) * 0.3 : 0;
  for (let s = 0; s < squadCount; s++) {
    const li = s % loopIndices.length;
    const loopIdx = loopIndices[li];
    const dir = counters[li] % 2 ? -1 : 1;
    const phase = counters[li] / perLoop;
    counters[li]++;
    const slot = squadCount > 1 ? (s / (squadCount - 1)) * 2 - 1 : 0;
    const squadLateral = (laneCenter || 0)
      + (lateralSpread ? slot * lateralSpread + (rng() * 2 - 1) * squadJitter : 0);
    const formation = formations && formations.length ? formations[s % formations.length] : 'line';
    const offs = squadFormationOffsets(formation, squadSize, fileGap);
    for (let m = 0; m < squadSize; m++) {
      const built = build(rng, s * squadSize + m);
      vehiclesGroup.add(built.group);
      const [fdx, fdz] = offs[m];
      const wobble = (rng() * 2 - 1) * fileWobble * 0.4;
      out.push({
        ...built, loopIdx, dir, phase,
        squadOffset: squadLateral + fdx + wobble,
        squadForwardOffset: fdz,
      });
    }
  }
  return out;
}

// Small stationary huddles -- soldiers standing in a loose ring facing
// inward (an idle animation, not marching), the "gathered/discussing"
// counterpart to the moving formation squads above. Positioned via the
// same scatterPositions rejection-sampling every other terrain-filler
// prop uses, so a huddle can never spawn on the patrol/convoy corridors.
// `kind: 'stationary'` is handled by its own early-return branch in
// stepMilitaryPatrol -- no path/loop following, just a fixed world
// position and heading set once here.
function spawnStandingGroups(rng, hw, hd, keepOut, count, build, avoidLoops, avoidLines, baseOffsetX, baseOffsetY) {
  const pts = scatterPositions(rng, hw, hd, keepOut, count, avoidLoops, avoidLines);
  const out = [];
  const gap = MIL_SOLDIER_SCALE * 1.6;
  for (const [x, z] of pts) {
    const groupSize = 3 + ((rng() * 3) | 0); // 3-5 soldiers per huddle
    const offs = squadCircleOffsets(groupSize, gap);
    for (let m = 0; m < groupSize; m++) {
      const built = build(rng, m);
      vehiclesGroup.add(built.group);
      const [dx, dz] = offs[m];
      // Face the huddle's own centre, not a shared heading -- reads as
      // soldiers actually facing each other rather than all facing north.
      const heading = Math.atan2(-dz, -dx);
      out.push({
        ...built, kind: 'stationary',
        worldX: baseOffsetX + x + dx, worldZ: baseOffsetY + z + dz,
        heading,
      });
    }
  }
  return out;
}

// ---- paratroopers ---------------------------------------------------
// A soldier descending under an open canopy, drifting down onto a fixed
// drop point and standing there for a while before the whole thing loops
// -- same "always something to see, not a one-shot event a viewer could
// easily miss" philosophy as the gun firing/explosions/patrol loops
// elsewhere in this file, rather than a single drop that only plays once.
const MIL_CHUTE_CANOPY_MAT = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0, color: 0xcabb8a, side: THREE.DoubleSide });
const MIL_CHUTE_STRIPE_MAT = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0, color: 0x8a5a3a, side: THREE.DoubleSide });
const MIL_CHUTE_CORD_MAT = new THREE.MeshBasicMaterial({ color: 0x1c1a16 });
// Orients a unit-length-agnostic cylinder mesh to run exactly between two
// points (position at the midpoint, default +Y axis rotated onto the
// segment's direction) -- used for the shroud cords, which are genuine 3D
// diagonals (canopy rim down to a central harness point), unlike the
// tower cross-bracing elsewhere in this file which only ever needs a
// horizontal rotation.
function orientCordBetween(mesh, from, to) {
  const dir = new THREE.Vector3().subVectors(to, from);
  const len = dir.length();
  mesh.position.copy(from).addScaledVector(dir, 0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
  return len;
}
// Canopy radius relative to the soldier hanging below it -- a real T-10
// round canopy is roughly 5-6x a person's height across; this is a bit
// more modest than that (a fully realistic size read as absurdly huge
// next to everything else at this scene's scale) but still clearly a
// large canopy DOMINATING the silhouette, not a token accessory barely
// bigger than the soldier's own shoulders (the original r=8 was actually
// smaller than the ~21-wide soldier body underneath it).
const MIL_CHUTE_RADIUS = 22;
// Both the base canopy dome and each gore stripe below get their vertex
// positions perturbed every frame (see stepChuteWave) to read as fabric
// rippling in the wind rather than a rigid moulded dome -- captures each
// mesh's ORIGINAL (pre-ripple) vertex positions here once at build time
// so the per-frame update always displaces from the true rest shape
// instead of compounding drift onto an already-displaced one.
function makeRippleSurface(geo, mat) {
  const mesh = new THREE.Mesh(geo, mat);
  return { mesh, geo, basePos: geo.attributes.position.array.slice() };
}
function buildParachuteCanopy(rng) {
  const g = new THREE.Group();
  const r = MIL_CHUTE_RADIUS;
  // Segment counts raised from the original 10x6 -- a coarse sphere has
  // too few vertices for the per-vertex ripple below to read as fabric
  // undulating rather than a faceted shape just wobbling as a whole.
  const canopyPart = makeRippleSurface(
    new THREE.SphereGeometry(r, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2),
    MIL_CHUTE_CANOPY_MAT,
  );
  canopyPart.mesh.scale.y = 0.55;
  canopyPart.mesh.castShadow = true;
  g.add(canopyPart.mesh);
  // 4 evenly-spaced gore stripes -- thin wedge slices of the same
  // hemisphere, just re-coloured and pushed out a hair to avoid
  // z-fighting with the base canopy -- enough to read as a panelled
  // parachute canopy instead of a plain dome. Rippled with the exact same
  // per-vertex formula (see stepChuteWave) so they stay visually glued to
  // the base dome's surface as both deform, instead of tearing away from
  // it as separate rigid pieces.
  const stripeParts = [];
  for (let i = 0; i < 4; i++) {
    const stripePart = makeRippleSurface(
      new THREE.SphereGeometry(r * 1.01, 16, 10, (i / 4) * Math.PI * 2, Math.PI / 4, 0, Math.PI / 2),
      MIL_CHUTE_STRIPE_MAT,
    );
    stripePart.mesh.scale.y = 0.55;
    g.add(stripePart.mesh);
    stripeParts.push(stripePart);
  }
  // Shroud lines: rim (spread around the canopy's now-elliptical base) down
  // to one shared harness point below -- a real canopy's dozens of cords
  // read fine as this many at the distance/scale these units are viewed.
  // Cord thickness scaled up from the original fixed 0.05 to stay in
  // proportion now the canopy itself is much bigger.
  const cordN = 6;
  const cordR = r * 0.006;
  const harness = new THREE.Vector3(0, -r * 2.1, 0);
  const rimY = r * 0.55 * 0.05; // canopy's own scale.y flattens its rim height
  for (let i = 0; i < cordN; i++) {
    const a = (i / cordN) * Math.PI * 2;
    const rim = new THREE.Vector3(Math.cos(a) * r * 0.97, rimY, Math.sin(a) * r * 0.97);
    const cord = new THREE.Mesh(new THREE.CylinderGeometry(cordR, cordR, 1, 4), MIL_CHUTE_CORD_MAT);
    const len = orientCordBetween(cord, rim, harness);
    cord.scale.y = len;
    g.add(cord);
  }
  g.userData.harnessY = harness.y;
  g.userData.canopyR = r;
  g.userData.rippleParts = [canopyPart, ...stripeParts];
  g.userData.wavePhase = rng() * Math.PI * 2;
  return g;
}
// Displaces each ripple-enabled surface's vertices radially by a small
// sine wave in azimuth + time, faded out toward the crown (height
// fraction 1) and strongest at the rim (height fraction 0) -- real
// canopy fabric flutters most at its free edge and is comparatively
// still where the vent/crown holds its shape. Driven by the SAME
// (angle, height, time, per-unit phase) formula for both the base dome
// and every gore stripe (see buildParachuteCanopy), so they displace
// identically at any shared point and never visibly separate.
function stepChuteWave(chuteGroup, simTime) {
  const { rippleParts, canopyR, wavePhase } = chuteGroup.userData;
  for (const part of rippleParts) {
    const pos = part.geo.attributes.position;
    const base = part.basePos;
    for (let i = 0; i < pos.count; i++) {
      const bx = base[i * 3], by = base[i * 3 + 1], bz = base[i * 3 + 2];
      const heightFrac = Math.min(1, Math.max(0, by / canopyR));
      const angle = Math.atan2(bz, bx);
      const ripple = 1 + Math.sin(angle * 5 + simTime * 4 + wavePhase) * 0.05 * (1 - heightFrac);
      pos.setXYZ(i, bx * ripple, by, bz * ripple);
    }
    pos.needsUpdate = true;
  }
}
// Wraps a soldier (real GLB model, falling back to the placeholder) with a
// canopy overhead -- the soldier hangs at the harness point, in whatever
// idle pose buildRealSoldier's Idle clip gives it (close enough to a
// harnessed hang at this low-poly scale/distance). Returns the assembled
// unit plus `chuteGroup` separately so stepMilitaryPatrol can hide just
// the canopy on landing without touching the soldier underneath.
function buildParatrooper(rng) {
  const wrapper = new THREE.Group();
  const chuteGroup = buildParachuteCanopy(rng);
  wrapper.add(chuteGroup);
  const soldier = (buildRealSoldier(rng, { idle: true }) || buildSoldierMesh(rng));
  // Both buildRealSoldier and buildSoldierMesh root their returned group at
  // the soldier's FEET (ground contact point), not the shoulders -- placing
  // that origin directly at the harness point (as an earlier version of
  // this did) put the whole ~40-unit-tall body ABOVE the harness, poking
  // up through the canopy instead of hanging below it. Risers/harness
  // attach around the shoulders in reality, roughly 85% of the way up the
  // body, so shift the feet down by that much instead.
  const shoulderDrop = MIL_SOLDIER_SCALE * 1.7 * 0.85;
  soldier.group.position.y = chuteGroup.userData.harnessY - shoulderDrop;
  wrapper.add(soldier.group);
  return { group: wrapper, mixer: soldier.mixer, chuteGroup, isSoldier: false };
}
// One (or a small handful) per base -- a fixed drop point chosen once at
// spawn time via the same scatterPositions rejection sampling as every
// other placed prop, so it can't land on the patrol rings/convoy
// corridor. `kind: 'paradrop'` gets its own stepMilitaryPatrol branch:
// descend, stand a while at the drop point (canopy hidden, chute
// "gathered"), then a short fully-hidden gap before redeploying from
// altitude again -- see that branch for the phase timing.
function spawnParatroopers(rng, hw, hd, keepOut, count, avoidLoops, avoidLines, baseOffsetX, baseOffsetY) {
  const pts = scatterPositions(rng, hw, hd, keepOut, count, avoidLoops, avoidLines);
  const out = [];
  pts.forEach(([x, z], i) => {
    const built = buildParatrooper(rng);
    vehiclesGroup.add(built.group);
    out.push({
      ...built, kind: 'paradrop',
      dropX: baseOffsetX + x, dropZ: baseOffsetY + z,
      // Well clear of the helicopter's 300-unit convoy cruise altitude
      // (see heliSpeedMps/airConvoyLoops below) -- jump altitude must
      // read as higher than the helicopter, not just sometimes higher by
      // chance the way the old 260-350 range (dipping below 300 half the
      // time) did.
      startY: 380 + rng() * 90,
      fallDuration: 22 + rng() * 6,
      // Impact: a short physics-driven stumble between touchdown and
      // standing upright (see the 'paradrop' branch in stepMilitaryPatrol)
      // -- landing dead upright and motionless read as a puppet snapping
      // into a pose, not a person hitting the ground under momentum.
      impactDuration: 1.9,
      // Random per-unit tumble direction/magnitude (radians) -- pitch
      // (forward/back) and roll (side/side) mixed independently so units
      // don't all keel over the same way. +-1.1 rad (~63 deg) is a hard
      // stumble/knockdown, not a subtle wobble.
      tumblePitch: (rng() * 2 - 1) * 1.1,
      tumbleRoll: (rng() * 2 - 1) * 0.75,
      // Momentum skid across the ground as they go down, decaying to a
      // stop -- same drag-style exponential approach every other decaying
      // motion in this file uses (see e.g. the fall's sway radius).
      skidAngle: rng() * Math.PI * 2,
      skidDist: 6 + rng() * 10,
      standDuration: 16 + rng() * 8,
      gapDuration: 3,
      phase: pts.length > 1 ? i / pts.length : rng(),
      swaySeed: rng() * Math.PI * 2,
    });
  });
  return out;
}
export function rebuildMilitaryPatrol(obstacles, environment) {
  if (environment !== 'Military Zone') {
    if (militaryMeshes.length) {
      for (const t of militaryMeshes) vehiclesGroup.remove(t.group);
      militaryMeshes = [];
      militaryLoops = [];
      militaryAirLoops = [];
      militaryConvoyLoops = [];
      militaryAirConvoyLoops = [];
    }
    clearMilitaryStatics();
    clearDustEmitters();
    clearMilitaryExplosions();
    militaryZoneActive = false;
    lastMilitaryKey = '';
    // Back to the default (much shorter) band -- unless Paddy Field is
    // active: its mist band is set in rebuildProps, which runs BEFORE
    // this per-poll reset, so a bare setFogRange() here would stomp it
    // on the very same poll.
    setFogRange(paddyFogBand?.[0], paddyFogBand?.[1]);
    setBattleAtmosphere(false);
    return;
  }
  const { hw, hd } = militaryExtent(obstacles);
  militaryFieldHw = hw; militaryFieldHd = hd;
  militaryZoneActive = true;
  // A second, fully independent replica base ~2km away (world units == 1
  // metre, matching the rest of the sim) -- own bunkers, camp, perimeter,
  // towers and patrol units, not just decoration copy-pasted near the
  // first one. 2000 is only the default gap -- a base's own hw/hd grows
  // with the AP's real coverage radius (see militaryExtent), and a fixed
  // 2000 gap stops being enough separation once hw alone approaches half
  // that (each base reaches roughly hw past its own centre in every
  // direction, so two bases first touch at gap == 2*hw). Grown here,
  // rather than left fixed, so a longer-range PHY config can't ever leave
  // the two bases overlapping/interpenetrating.
  MIL_BASE_OFFSET_X = Math.max(2000, hw * 2.3);
  const OFF_X = MIL_BASE_OFFSET_X, OFF_Y = MIL_BASE_OFFSET_Y;
  // Push the haze band out past the real scene extent (including the
  // second base) -- see setFogRange's docstring -- so it only reaches
  // full sky-colour well beyond both perimeters, not over the battlefield.
  setFogRange(Math.max(hw, hd) * 1.7, OFF_X + Math.max(hw, hd) * 2.2);
  setBattleAtmosphere(true);
  // Ring layout, x2 (one set per base): tanks (outer two), soldiers
  // (inner two), trucks (near-base ring). Each mobile-unit type gets its
  // OWN dedicated ring per base -- sharing a ring between two unit types
  // sounded fine in theory (different phase offsets) but each type's
  // phases are computed independently starting from 0, so two types on
  // the same ring could easily land on the same phase and visibly
  // overlap/collide. One ring per type rules that out entirely.
  // Last two fractions (0.95/0.87) are the border-guard patrol lines --
  // close to the actual perimeter fence (at 1.0), not the inner patrol
  // rings above. Kept as their own dedicated rings rather than reusing
  // 0.78, same reasoning as the tank/soldier/truck split below: sharing a
  // ring means sharing a phase-collision risk.
  const ringFracs = [0.78, 0.55, 0.34, 0.18, 0.50, 0.95, 0.87];
  const loops = [];
  for (const [cx, cy] of [[0, 0], [OFF_X, OFF_Y]]) {
    for (const f of ringFracs) loops.push({ cx, cy, hw: hw * f, hh: hd * f });
  }
  // Period scales with loop perimeter so patrol speed stays realistic
  // (~7 m/s tank/truck, ~1.6 m/s walking soldier) regardless of how large
  // the base actually is -- a fixed period made vehicles on a 900m+ site
  // (802.11ah's real long range can scatter checkpoints that far out)
  // whip around the loop far faster than a real tank or soldier could.
  for (const l of loops) {
    const perim = 2 * (2 * l.hw + 2 * l.hh);
    l.period = Math.max(20, perim / 7);
  }
  // Within each base's 7 rings (indices 0-6 for base 1, 7-13 for base 2):
  // slots 2,3 are soldier rings (need walking pace, not tank pace -- a
  // previous version only fixed one of the two and left soldiers on the
  // other one sprinting at full tank speed), slot 4 is the truck ring
  // (faster than a tank's cross-country pace), and slots 5,6 are the
  // border-guard lines -- running pace, faster than the inner patrol
  // soldiers' walk but not as fast as the inter-base convoy's 9 m/s.
  for (const base of [0, 7]) {
    for (const idx of [base + 2, base + 3]) {
      loops[idx].period = Math.max(16, (2 * (2 * loops[idx].hw + 2 * loops[idx].hh)) / 1.6);
    }
    loops[base + 4].period = Math.max(12, (2 * (2 * loops[base + 4].hw + 2 * loops[base + 4].hh)) / 16);
    for (const idx of [base + 5, base + 6]) {
      loops[idx].period = Math.max(14, (2 * (2 * loops[idx].hw + 2 * loops[idx].hh)) / 7);
    }
  }
  const airLoops = [];
  for (const [cx, cy] of [[0, 0], [OFF_X, OFF_Y]]) {
    airLoops.push({ cx, cy, r: Math.max(hw, hd) * 0.92, alt: 95 });
    airLoops.push({ cx, cy, r: Math.max(hw, hd) * 0.62, alt: 135 });
  }
  const planeSpeedMps = 150; // fast enough to be unmistakably in motion, not just realistic
  for (const l of airLoops) l.period = Math.max(20, (2 * Math.PI * l.r) / planeSpeedMps);

  // Convoy routes -- straight lines between the two bases' centres, not
  // loops around either one, so units genuinely travel from one end to
  // the other (and back) instead of just guarding their own base. Speeds
  // are well past "realistic" -- at this scene's scale (bases can be
  // 1000+ units across, 2km apart), a real-world 7 m/s tank covers such
  // a tiny fraction of the visible frame each second that it read as
  // "nothing is moving" even though it genuinely was; motion needs to be
  // unmistakable at a glance, not just technically present.
  const baseDist = Math.hypot(OFF_X, OFF_Y);
  // reversed swaps which endpoint a unit starts toward -- since heading
  // is always derived from the path's own tangent (see stepMilitaryPatrol),
  // this flips BOTH the direction of travel AND which way units face
  // together, so it can never reintroduce a facing/motion mismatch the
  // way naively negating just one of the two would.
  function convoyLane(speedMps, reversed = false) {
    const period = Math.max(15, (baseDist * 2) / speedMps);
    return reversed
      ? { x0: OFF_X, y0: OFF_Y, x1: 0, y1: 0, period }
      : { x0: 0, y0: 0, x1: OFF_X, y1: OFF_Y, period };
  }
  const convoyLoops = [
    convoyLane(60),   // 0: trucks
    convoyLane(35),   // 1: tanks
    // Tried 154 (a literal +1000%) per an explicit request for a dramatic
    // speed boost, but at that speed the translation vastly outpaces the
    // run clip's own stride length -- legs freeze mid-stride while the
    // body glides across the ground, reading as a floating/leaping pose
    // instead of running (confirmed by screenshotting it: every soldier
    // hangs in one long static-looking stride). 30 is still more than
    // double the previous 14 and clearly faster than a plain jog, without
    // outrunning what the animation can sell as an actual running motion.
    // reversed=false per a follow-up request flipping this back --
    // soldiers now start toward base 2 in lockstep with trucks/tanks
    // again, instead of launching toward base 1 on their own.
    convoyLane(45, false),   // 2: marching squads/solo troops
  ];
  // Well above anything on the ground (radio towers top out around 85) --
  // the old 110/145 flew planes low enough to visually skim the terrain.
  // Helicopters (lane 2) fly their own, much lower helipad-to-helipad
  // route -- same (hw*0.44, -hd*0.46) local point addHelipad places its
  // pad at, offset into base 2's world position for the far end -- so
  // they visibly depart/arrive at an actual helipad, not just mid-air.
  const heliSpeedMps = 45;
  const heliX0 = hw * 0.44, heliY0 = -hd * 0.46;
  const heliX1 = OFF_X + hw * 0.44, heliY1 = OFF_Y - hd * 0.46;
  const heliDist = Math.hypot(heliX1 - heliX0, heliY1 - heliY0);
  const airConvoyLoops = [
    { x0: 0, y0: 0, x1: OFF_X, y1: OFF_Y, alt: 320, period: Math.max(20, (baseDist * 2) / planeSpeedMps) },
    { x0: 0, y0: 0, x1: OFF_X, y1: OFF_Y, alt: 420, period: Math.max(20, (baseDist * 2) / planeSpeedMps) },
    // Was 95, then 170, then 240 -- kept short of the planes' low lane
    // (320) to avoid the two crossing at the same altitude, but pushed as
    // close to it as still leaves clear separation.
    { x0: heliX0, y0: heliY0, x1: heliX1, y1: heliY1, alt: 300, period: Math.max(15, (heliDist * 2) / heliSpeedMps) },
  ];
  // How far units spread from the centre line -- generous relative to the
  // base's own footprint so the whole corridor width is used. Split into
  // 4 non-overlapping BANDS (one per unit type) so trucks/tanks/squads/
  // solo-troops can never cross into each other's territory -- combined
  // with spawnPatrolGroup/spawnSquads' evenly-spaced slots (not
  // independent random offsets) within each band, no two units, of any
  // type, can ever end up on the same lateral track: guaranteed no
  // overlap, and if the auto-rebuild ever changes counts, units simply
  // get reassigned to new slots rather than needing to "reroute".
  const groundSpread = Math.max(hw, hd) * 0.85;
  const bandW = groundSpread * 0.42;
  const truckBand = -groundSpread * 0.78, tankBand = -groundSpread * 0.26;
  const squadBand = groundSpread * 0.26, soloBand = groundSpread * 0.78;
  const airSpread = Math.max(hw, hd) * 0.65;

  const key = JSON.stringify(loops) + '|' + JSON.stringify(airLoops) + '|'
    + JSON.stringify(convoyLoops) + '|' + JSON.stringify(airConvoyLoops) + '|' + obstacles.length;
  if (key === lastMilitaryKey) return;
  lastMilitaryKey = key;
  militaryLoops = loops;
  militaryAirLoops = airLoops;
  militaryConvoyLoops = convoyLoops;
  militaryAirConvoyLoops = airConvoyLoops;

  for (const t of militaryMeshes) vehiclesGroup.remove(t.group);
  militaryMeshes = [];
  clearMilitaryStatics();
  clearDustEmitters();

  // Every mobile unit travels the convoy route between the two bases
  // instead of patrolling locally -- tanks/trucks/planes each get their
  // own lane pair (see convoyLoops/airConvoyLoops above), one marching
  // squad lane for soldiers. Counts unchanged from the previous (doubled)
  // pass, just re-tasked from local patrol to the inter-base route.
  const rng = mulberry32(hashSeed('military|' + key));
  militaryMeshes.push(...spawnPatrolGroup(rng, [1], 24, buildMilitaryTankMesh, bandW, tankBand)
    .map((m) => ({ ...m, kind: 'convoyGround' })));
  // Formation mix cycles line/triangle/circle/square across the 12 squads
  // (3 of each) -- per an explicit request for visible variety instead of
  // every squad marching single-file.
  militaryMeshes.push(...spawnSquads(rng, [2], 12, 12,
    (r) => buildRealSoldier(r) || buildSoldierMesh(r), bandW, squadBand,
    ['line', 'triangle', 'circle', 'square'])
    .map((m) => ({ ...m, kind: 'convoyGround' })));
  // Lone troops, not squad members -- spawnPatrolGroup (not spawnSquads)
  // so each one gets its own independent lateral position and phase,
  // rather than moving as part of a formation block.
  militaryMeshes.push(...spawnPatrolGroup(rng, [2], 14,
    (r) => buildRealSoldier(r) || buildSoldierMesh(r), bandW, soloBand)
    .map((m) => ({ ...m, kind: 'convoyGround' })));
  militaryMeshes.push(...spawnPatrolGroup(rng, [0], 15, buildMilitaryTruckMesh, bandW, truckBand)
    .map((m) => ({ ...m, kind: 'convoyGround' })));
  // Border guards -- two lines of running soldiers circling each base's
  // actual perimeter fence (rings at 0.95/0.87 of hw,hd, see ringFracs
  // above), rather than the inter-base convoy route. `kind` is left
  // unset so these fall through to stepMilitaryPatrol's rectLoopPos
  // branch instead of the straight-line convoy one. Loop indices 5/6 are
  // base 1's border rings, 12/13 are base 2's (7 rings per base).
  militaryMeshes.push(...spawnPatrolGroup(rng, [5, 12], 16,
    (r) => buildRealSoldier(r) || buildSoldierMesh(r), 5));
  militaryMeshes.push(...spawnPatrolGroup(rng, [6, 13], 16,
    (r) => buildRealSoldier(r) || buildSoldierMesh(r), 5));
  // Cut from 12/6 -- a less crowded sky. Both fly a fixed cruise altitude
  // for the whole trip now (see stepMilitaryPatrol's convoyAir branch),
  // no climb-out/descent, so there's no more "bigger ramp for helicopters
  // than planes" distinction to preserve here.
  militaryMeshes.push(...spawnPatrolGroup(rng, [0, 1], 4, buildMilitaryPlaneMesh, airSpread)
    .map((m) => ({ ...m, kind: 'convoyAir' })));
  militaryMeshes.push(...spawnPatrolGroup(rng, [2], 2, buildMilitaryHelicopterMesh, Math.max(hw, hd) * 0.12)
    .map((m) => ({ ...m, kind: 'convoyAir' })));
  if (!soldierGLTF) {
    // Real model not loaded yet -- force one more rebuild once it is, so
    // the placeholder capsule figures get swapped out automatically
    // rather than staying stuck until something else changes the scene.
    loadSoldierModel().then((gltf) => { if (gltf) lastMilitaryKey = ''; });
  }

  // Static extras, base 1: no obstacle rect of their own (full perimeter +
  // 4 corner watchtowers, forward mortar pit, helipad) -- mirrors
  // topology_canvas.py's _bg_military_v1, which draws these
  // independently of any obstacle registration. Tracked via a
  // before/after diff of buildingsGroup's children so every helper
  // (some add one mesh, some add a dozen) gets cleaned up uniformly.
  const before = new Set(buildingsGroup.children);
  // Convoy-corridor avoidance for filler props, in each base's own LOCAL
  // space (base 1's local space is world space; base 2's local space is
  // shifted by -OFF_X,-OFF_Y since its wrapper group sits at the offset).
  // Margin covers the widest lane (tank lanes at +-40) plus vehicle width.
  const convoyMargin = 60;
  const base1ConvoyLines = [{ x0: 0, y0: 0, x1: OFF_X, y1: OFF_Y, margin: convoyMargin }];
  const base2ConvoyLines = [{ x0: -OFF_X, y0: -OFF_Y, x1: 0, y1: 0, margin: convoyMargin }];
  const extras1 = buildBaseExtras(buildingsGroup, hw, hd, key, 'base1', loops.slice(0, 5), base1ConvoyLines, 0, 0);
  militaryGuns.push(...extras1.guns);
  militaryMeshes.push(...extras1.standing);

  // Base 2: the replica, 2km away. Its command bunker/checkpoints/camp
  // have no real obstacle behind them (base 1's come from the actual
  // PHY-relevant obstacles the Python side registers) -- they're built
  // straight from addMilitaryStructure/addSupplyDepot with synthetic
  // obstacle-shaped objects, visual only, same as everything else here.
  const base2 = new THREE.Group();
  base2.position.set(OFF_X, 0, OFF_Y);
  buildSyntheticBunkers(base2, hw, hd);
  // Base 2's own filler (loops/lines are base-1-relative shapes, see
  // buildBaseExtras' own comment) sits inside the base2 wrapper group, so
  // its static props get the +OFF_X,+OFF_Y offset for free via the scene
  // graph -- but standing-group soldiers are plain militaryMeshes entries
  // driven by world-space worldX/worldZ (see stepMilitaryPatrol's
  // 'stationary' branch), so spawnStandingGroups needs the offset passed
  // in explicitly to place them at base 2's actual world position.
  const extras2 = buildBaseExtras(base2, hw, hd, key, 'base2', loops.slice(0, 5), base2ConvoyLines, OFF_X, OFF_Y);
  militaryGuns.push(...extras2.guns);
  militaryMeshes.push(...extras2.standing);
  buildingsGroup.add(base2);

  for (const child of buildingsGroup.children) {
    if (!before.has(child)) militaryStatics.push(child);
  }
}

// Perimeter + corner watchtowers + mortar pit + helipad + radio towers +
// bridge + ground camo + terrain filler -- everything a base needs
// *besides* its bunkers (base 1 gets those from the real obstacle list in
// rebuildBuildings; base 2 gets synthetic ones from buildSyntheticBunkers).
// `avoidLoops` only needs hw/hh (not cx/cy, see nearRectOutline), so the
// same base-1-relative loop shapes work unchanged for base 2's own filler.
function buildBaseExtras(parent, hw, hd, key, tag, avoidLoops, avoidLines, offsetX, offsetY) {
  addMilitaryPerimeter(parent, hw, hd);
  addWatchtower(parent, -hw, -hd, 42);
  addWatchtower(parent, hw, -hd, 42);
  addWatchtower(parent, -hw, hd, 42);
  addWatchtower(parent, hw, hd, 42);
  // Positioned in the gaps *between* the patrol rings (0.78/0.55/0.34/0.18
  // of hw,hd), not at fractions that could sit right on a ring's
  // straight-edge outline: a fixed structure that close to a path a 64m
  // -scaled tank or 43m-scaled soldier actually walks looked like it was
  // clipping straight through them.
  // Both mortar pit and artillery guns now return a firing rig (see
  // makeGunRig) -- collected into `guns` and returned below so
  // rebuildMilitaryPatrol can add them to the module-level militaryGuns
  // array stepMilitaryPatrol drives every frame.
  const guns = [addMortarPit(parent, hw * 0.44, hd * 0.44, tag)];
  addHelipad(parent, hw * 0.44, -hd * 0.46);
  addRadioTower(parent, -hw * 0.66, hd * 0.12, 85);
  addRadioTower(parent, hw * 0.68, hd * 0.64, 70);
  addRavineBridge(parent, hw * 0.46, -hd * 0.22, Math.PI / 2, 30);
  // A 5-gun artillery battery along the base's south-west edge -- clear of
  // every other fixed structure above (mortar/helipad sit east, radio
  // towers north, bridge south-east), spaced out in a row facing outward
  // through the perimeter.
  for (let i = 0; i < 5; i++) {
    guns.push(addArtilleryGun(parent, -hw * 0.8, hd * 0.55 - i * 16, Math.PI, tag));
  }
  // 5 more watchtowers along the north edge -- bigger than the scattered
  // filler towers (see addMilitaryFiller's nExtraTowers, 26-38 tall) so
  // this reads as a deliberate line, not just more of the same random
  // background ones.
  for (let i = 0; i < 5; i++) {
    addWatchtower(parent, -hw * 0.5 + i * (hw * 0.25), hd * 0.82, 55);
  }
  // 5 more field camps down the east edge, clear of the gun battery
  // (west), radio towers (north) and bridge (south-east).
  const campRng = mulberry32(hashSeed(`military-camps|${tag}|${key}`));
  for (let i = 0; i < 5; i++) {
    addFieldCamp(parent, hw * 0.78, -hd * 0.65 + i * 16, campRng);
  }
  addMilitaryFiller(parent, mulberry32(hashSeed(`military-filler|${tag}|${key}`)), hw, hd, avoidLoops, avoidLines);
  // A few small huddles of soldiers standing around talking -- the "at
  // ease" counterpart to every other unit here being on patrol/convoy/
  // guard duty. Scattered across the base like the terrain filler above,
  // clear of the same patrol rings/convoy corridor. Count scales gently
  // with base size, capped low: these are a garnish, not the main troop
  // presence (that's the marching squads/border guards/solo troops).
  const standRng = mulberry32(hashSeed(`military-standing|${tag}|${key}`));
  const nHuddles = Math.max(2, Math.min(6, Math.round((hw * hd) / 200000)));
  const standing = spawnStandingGroups(standRng, hw, hd, 80, nHuddles,
    (r) => buildRealSoldier(r, { idle: true }) || buildSoldierMesh(r),
    avoidLoops, avoidLines, offsetX || 0, offsetY || 0);
  // One paratrooper drop per base (2 total, see spawnParatroopers) -- a
  // fixed drop point, same keepOut/avoidance as everything else here.
  const chuteRng = mulberry32(hashSeed(`military-chute|${tag}|${key}`));
  const paratroopers = spawnParatroopers(chuteRng, hw, hd, 80, 1,
    avoidLoops, avoidLines, offsetX || 0, offsetY || 0);
  return { guns, standing: standing.concat(paratroopers) };
}

// Base 2's command bunker/checkpoints/camp -- same builders (and same
// labels, so they come out visually identical) as base 1 uses, just fed
// synthetic obstacle-shaped objects instead of real ones from the Python
// obstacle list, since base 2 has no PHY presence of its own.
function buildSyntheticBunkers(parent, hw, hd) {
  addMilitaryStructure(parent, { label: 'command-bunker', x0: -5, y0: -5, x1: 5, y1: 5 }, 'command-bunker');
  const checkpoints = [
    [0, -hd * 0.36, 'checkpoint-north'],
    [0, hd * 0.36, 'checkpoint-south'],
    [hw * 0.36, 0, 'checkpoint-east'],
    [-hw * 0.36, 0, 'checkpoint-west'],
  ];
  for (const [cx, cz, label] of checkpoints) {
    addMilitaryStructure(parent, { label, x0: cx - 4, y0: cz - 4, x1: cx + 4, y1: cz + 4 }, label);
  }
  const dx = hw * 0.20, dz = -hd * 0.10;
  addSupplyDepot(parent, { label: 'supply-depot', x0: dx - 3, y0: dz - 3, x1: dx + 3, y1: dz + 3 }, 'supply-depot');
}

// ---- ambient bomb/explosion effects -----------------------------------
// Periodic impact bursts scattered across the field -- flash, fireball/
// smoke, a fast ground shockwave ring, and a lingering scorch decal.
// Purely ambient/decorative, exactly like the smoke/dust puffs in
// smoke.js: NOT tied to any real sim event (packet tx, association
// state, etc.) -- this is an RF simulator, not a combat sim, so timing is
// plain pseudo-random the same way the small-arms muzzle-flash cycle
// already is, not driven by anything in /api/state. Parented under
// smokeGroup (imported from scene.js) rather than propsGroup/
// buildingsGroup -- both of those get wholesale-cleared by unrelated
// rebuilds (rebuildProps/rebuildBuildings key changes) whenever the scene
// changes, which would silently kill a burst mid-animation; smokeGroup is
// the one group in the scene ONLY ever touched by per-frame pooled-effect
// code like this, same reason smoke.js's own puffs live there.
//
// Same hard-cap discipline MAX_SMOKE_PUFFS/MAX_DUST_PUFFS in smoke.js
// insist on (see that file's own comments for the exact past incident:
// dozens of uncapped emitters steadied out around 350-400+ extra draw
// calls, read by a user as the whole scene being "stuck"). A burst is a
// heavier composite effect than a single puff -- 1 flash sprite + 3
// fireball/smoke cubes + 1 shockwave ring, ~5 objects -- so the concurrent
// cap is deliberately much lower than the puff pools' own 48-60.
const MAX_EXPLOSIONS = 6;
// Scorch decals outlive the burst itself (a lasting mark, not part of the
// flash/smoke), but still need their OWN independent cap for exactly the
// same "no unbounded growth over a long session" reason -- left uncapped,
// a long-running session would accumulate one more permanently-live decal
// per burst forever.
const MAX_DYNAMIC_SCORCH = 20;
let militaryExplosions = [];
let militaryScorchDecals = [];
let explosionSpawnTimer = 0;
const MIL_FIREBALL_COLOR = new THREE.Color(0xff7a33);
const MIL_EXPLOSION_SMOKE_COLOR = new THREE.Color(0x39352c);

function clearMilitaryExplosions() {
  for (const e of militaryExplosions) {
    smokeGroup.remove(e.flash); e.flash.material.dispose();
    for (const d of e.debris) { smokeGroup.remove(d.mesh); d.mesh.material.dispose(); }
    smokeGroup.remove(e.ring); e.ring.geometry.dispose(); e.ring.material.dispose();
  }
  militaryExplosions = [];
  for (const s of militaryScorchDecals) { smokeGroup.remove(s.mesh); s.mesh.material.dispose(); }
  militaryScorchDecals = [];
}

// Picks a point spread across whichever base's footprint is currently
// built (see militaryFieldHw/Hd, set every poll by rebuildMilitaryPatrol),
// biased away from the command-bunker cluster at each base's own centre
// (same 68-90-unit keepOut idea addMilitaryFiller already uses) so a burst
// doesn't visually erupt from inside a static building mesh.
function randomExplosionPoint() {
  if (militaryFieldHw <= 0 || militaryFieldHd <= 0) return null;
  const bases = [[0, 0], [MIL_BASE_OFFSET_X, MIL_BASE_OFFSET_Y]];
  const [bx, by] = bases[(Math.random() * bases.length) | 0];
  for (let attempt = 0; attempt < 4; attempt++) {
    const x = bx + (Math.random() * 2 - 1) * militaryFieldHw * 0.92;
    const z = by + (Math.random() * 2 - 1) * militaryFieldHd * 0.92;
    if (Math.hypot(x - bx, z - by) > 90) return [x, z];
  }
  return null;
}

function spawnScorchDecal(x, z) {
  if (militaryScorchDecals.length >= MAX_DYNAMIC_SCORCH) {
    // Recycle the oldest mark instead of silently dropping this burst's
    // one -- same "cap, don't grow" rule as everything else here, just
    // reusing the slot rather than skipping the spawn outright.
    const oldest = militaryScorchDecals.shift();
    smokeGroup.remove(oldest.mesh);
    oldest.mesh.material.dispose();
  }
  // MIL_SCORCH_MAT's own look (near-black, semi-transparent, polygon-
  // offset against the ground -- see its own comment for why that offset
  // matters at this scene's scale) is exactly right here too, but it's a
  // clone, not the shared instance: MIL_SCORCH_MAT backs an InstancedMesh
  // of pre-placed static scorch marks with one fixed opacity for all of
  // them, and this decal needs its own independent fade-in/out timeline,
  // same reasoning as smoke.js cloning a fresh material per puff instead
  // of sharing one.
  const mat = MIL_SCORCH_MAT.clone();
  mat.opacity = 0;
  const mesh = new THREE.Mesh(UNIT_PATCH_GEO, mat);
  const r = 5 + Math.random() * 6;
  mesh.position.set(x, 0.06, z);
  mesh.rotation.set(-Math.PI / 2, 0, Math.random() * Math.PI);
  mesh.scale.set(r, r * (0.6 + Math.random() * 0.5), 1);
  smokeGroup.add(mesh);
  militaryScorchDecals.push({ mesh, age: 0, life: 45 + Math.random() * 20 });
}

function spawnExplosion(x, z) {
  if (militaryExplosions.length >= MAX_EXPLOSIONS) return;
  // Flash -- same soft additive glow sprite as muzzle/gun flashes (see
  // makeFlashSprite), just much bigger and brighter, gone within ~0.15s.
  const flash = makeFlashSprite(16, 0xfff2c8);
  flash.position.set(x, 4, z);
  flash.visible = true;
  smokeGroup.add(flash);

  // Fireball -> smoke: blocky cubes (BoxGeometry, MeshStandardMaterial),
  // matching the deliberately low-poly primitive look every other
  // particle-ish effect in this file uses (see smoke.js's own puffs) --
  // NOT another soft gradient sprite, which would read as "matter" made
  // of light rather than an actual solid cloud.
  const debris = [];
  for (let i = 0; i < 3; i++) {
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, color: MIL_FIREBALL_COLOR, transparent: true, opacity: 0.95 });
    const mesh = new THREE.Mesh(UNIT_BOX_GEO, mat);
    mesh.position.set(x + (Math.random() - 0.5) * 4, 1 + Math.random() * 2, z + (Math.random() - 0.5) * 4);
    mesh.rotation.y = Math.random() * Math.PI;
    mesh.scale.setScalar(1.4 + Math.random() * 1.2);
    smokeGroup.add(mesh);
    debris.push({ mesh, drift: [(Math.random() - 0.5) * 3, (Math.random() - 0.5) * 3] });
  }

  // Ground shockwave -- same technique entities.js's packet broadcast-
  // pulse ring uses (a flat RingGeometry that scales up while fading),
  // just built inline here rather than imported, to avoid a world.js <->
  // entities.js circular import (entities.js already imports FROM this
  // file for roofHeightAt).
  const ringGeo = new THREE.RingGeometry(0.85, 1, 20);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0xffd9a0, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false,
  });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(x, 1.0, z);
  smokeGroup.add(ring);

  militaryExplosions.push({ age: 0, life: 2.6, flash, debris, ring });
  spawnScorchDecal(x, z);
}

function stepMilitaryExplosions(simTime, dt) {
  if (militaryZoneActive) {
    explosionSpawnTimer -= dt;
    if (explosionSpawnTimer <= 0) {
      // Next burst in ~2.5-6s -- occasional impacts across a large,
      // two-base field, not a barrage.
      explosionSpawnTimer = 2.5 + Math.random() * 3.5;
      const pt = randomExplosionPoint();
      if (pt) spawnExplosion(pt[0], pt[1]);
    }
  }
  for (let i = militaryExplosions.length - 1; i >= 0; i--) {
    const e = militaryExplosions[i];
    e.age += dt;
    const t = e.age / e.life;
    if (t >= 1) {
      smokeGroup.remove(e.flash); e.flash.material.dispose();
      for (const d of e.debris) { smokeGroup.remove(d.mesh); d.mesh.material.dispose(); }
      smokeGroup.remove(e.ring); e.ring.geometry.dispose(); e.ring.material.dispose();
      militaryExplosions.splice(i, 1);
      continue;
    }
    // Flash: near-instant, fully gone by ~0.15s in.
    const flashT = e.age / 0.15;
    e.flash.visible = flashT < 1;
    if (flashT < 1) {
      e.flash.material.opacity = 1 - flashT;
      e.flash.scale.setScalar(e.flash.userData.baseScale * (0.7 + flashT * 1.6));
    }
    // Fireball -> smoke: colour drifts from bright orange to dark smoke
    // grey while the cubes grow, rise and fade across the burst's full
    // lifetime -- a compressed one-shot version of spawnSmoke's own
    // per-puff animation math in smoke.js (rise/grow/fade), just
    // triggered as a batch instead of a continuous per-frame emitter,
    // since a bomb impact is a single event, not an ongoing plume.
    for (const d of e.debris) {
      d.mesh.position.x += d.drift[0] * dt;
      d.mesh.position.z += d.drift[1] * dt;
      d.mesh.position.y += dt * (1.2 + t * 2.0);
      d.mesh.scale.setScalar(1.4 + t * 3.2);
      d.mesh.material.color.lerpColors(MIL_FIREBALL_COLOR, MIL_EXPLOSION_SMOKE_COLOR, Math.min(1, t * 1.6));
      d.mesh.material.opacity = 0.95 * (1 - t);
    }
    // Shockwave -- fast expand, fast fade, done well before the smoke
    // clears (~0.7s of the full ~2.6s burst life).
    const ringT = Math.min(1, e.age / 0.7);
    e.ring.visible = ringT < 1;
    if (ringT < 1) {
      e.ring.scale.setScalar(1 + ringT * 22);
      e.ring.material.opacity = 0.9 * (1 - ringT);
    }
  }
  // Dynamic scorch decals fade out slowly over a much longer lifetime than
  // the burst itself -- capped independently at MAX_DYNAMIC_SCORCH above.
  for (let i = militaryScorchDecals.length - 1; i >= 0; i--) {
    const s = militaryScorchDecals[i];
    s.age += dt;
    const t = s.age / s.life;
    if (t >= 1) {
      smokeGroup.remove(s.mesh); s.mesh.material.dispose();
      militaryScorchDecals.splice(i, 1);
      continue;
    }
    const fadeIn = Math.min(1, t / 0.05);
    const fadeOut = t > 0.8 ? 1 - (t - 0.8) / 0.2 : 1;
    s.mesh.material.opacity = 0.55 * fadeIn * fadeOut;
  }
}

// Mortar pit / artillery gun firing -- see makeGunRig and militaryGuns.
// FLASH/RECOIL windows mirror the small-arms muzzle-flash's own short
// cyc<window convention (see the militaryMeshes loop in stepMilitaryPatrol
// below), just on each gun's own (much longer, per-gun-phased) period.
const GUN_FLASH_WINDOW = 0.22;
const GUN_RECOIL_WINDOW = 0.55;
function stepMilitaryGuns(simTime) {
  for (const gun of militaryGuns) {
    const cyc = (simTime + gun.phase) % gun.period;
    const firing = cyc < GUN_FLASH_WINDOW;
    gun.flash.visible = firing;
    gun.light.visible = firing;
    if (firing) {
      const w = cyc / GUN_FLASH_WINDOW;
      const k = Math.max(0, 1 - w * 1.5);
      gun.flash.material.opacity = k;
      gun.flash.scale.setScalar(gun.flash.userData.baseScale * (0.7 + k * 0.8));
      gun.light.intensity = k * 9;
    }
    // Recoil: fast kick back right at ignition (first 15% of the recoil
    // window), slower eased return over the rest -- an asymmetric snap
    // rather than a smooth back-and-forth oscillation, translating the
    // barrel/tube mesh along its own fixed muzzle-forward axis
    // (recoilDir, computed once in makeGunRig) rather than rotating it --
    // these barrels are visually mounted in a fixed cradle/shield/split-
    // trail carriage that doesn't move, so only the tube/barrel itself
    // sliding back along its own bore axis reads as recoil; rotating it
    // independently of its mount would look like it was detaching.
    if (cyc < GUN_RECOIL_WINDOW) {
      const rc = cyc / GUN_RECOIL_WINDOW;
      const kick = rc < 0.15 ? (rc / 0.15) : Math.max(0, 1 - (rc - 0.15) / 0.85);
      gun.barrel.position.copy(gun.restPos).addScaledVector(gun.recoilDir, -kick * gun.kickDist);
    } else if (!gun.barrel.position.equals(gun.restPos)) {
      gun.barrel.position.copy(gun.restPos);
    }
  }
}

// Extends the existing t.muzzle cone toggle (both call sites below) with
// the actual bright flash added in buildMilitaryTankMesh/buildSoldierMesh/
// attachRifle -- `cyc`/`on` are exactly the same values the caller already
// computed for the cone, unchanged. t.flash uses the ONE shared
// MIL_MUZZLE_FLASH_MAT across every tank/soldier (see its own comment for
// why), so only `visible` (per-Sprite, safe) is touched here -- NOT
// opacity, which lives on that shared material and would otherwise
// flicker every other currently-visible flash to whichever unit's value
// happened to be set last this frame. t.flashLight (tanks only) is a real
// per-instance PointLight, not shared, so its intensity can safely pulse.
function applyMuzzleFlash(t, cyc, on) {
  if (t.flash) t.flash.visible = on;
  if (t.flashLight) {
    t.flashLight.visible = on;
    if (on) t.flashLight.intensity = Math.max(0, 1 - cyc / 0.1) * 8;
  }
}

export function stepMilitaryPatrol(simTime, dt) {
  for (const t of militaryMeshes) {
    if (t.mixer) t.mixer.update(dt || 0);
    if (t.kind === 'stationary') {
      // Fixed position/heading, set once at spawn (see spawnStandingGroups)
      // -- no path to follow, no muzzle flash/jump cosmetics, just the
      // idle animation clip (already selected at build time) looping.
      const p = toScene(t.worldX, t.worldZ);
      t.group.position.set(p.x, 0, p.z);
      t.group.rotation.y = t.heading;
      continue;
    }
    if (t.kind === 'paradrop') {
      // Four phases on a repeating cycle (see spawnParatroopers for the
      // per-unit durations/phase offset): fall from altitude onto the
      // fixed drop point, a short physics-driven impact stumble, stand
      // there a while with the canopy "gathered" (hidden), then a short
      // fully-hidden gap before redeploying from altitude again -- the
      // gap is what makes the loop-back read as a fresh drop instead of a
      // visible teleport from ground to sky.
      const total = t.fallDuration + t.impactDuration + t.standDuration + t.gapDuration;
      const ct = ((simTime + t.phase * total) % total + total) % total;
      if (ct < t.fallDuration) {
        const p = ct / t.fallDuration;
        // Lazy spiralling drift that decays to zero as the ground nears,
        // so the landing itself lands exactly on the drop point rather
        // than wherever the sway happened to be at that instant.
        const swayR = 26 * (1 - p);
        const sx = Math.sin(p * 5.5 + t.swaySeed) * swayR;
        const sz = Math.cos(p * 4 + t.swaySeed) * swayR;
        const wp = toScene(t.dropX + sx, t.dropZ + sz);
        t.group.position.set(wp.x, t.startY * (1 - p), wp.z);
        t.group.rotation.set(0, simTime * 0.2 + t.swaySeed, 0); // slow twist under canopy
        t.group.scale.setScalar(1);
        t.group.visible = true;
        t.chuteGroup.visible = true;
        stepChuteWave(t.chuteGroup, simTime);
      } else if (ct < t.fallDuration + t.impactDuration) {
        // Landing on your feet dead-still would read as a puppet snapping
        // into a pose, not real momentum hitting the ground -- modelled as
        // a damped oscillator's IMPULSE response for the tumble/recover
        // (theta(t) = theta0 * e^-zeta*t * sin(omega*t) -- sin, not cos:
        // this is the closed-form response to a sudden KICK at t=0, which
        // starts at rest and swings out, unlike cos's response to a sudden
        // DISPLACEMENT, which starts already at full amplitude. Using cos
        // here made the body's tilt pop instantly to its max stumble angle
        // in the very first frame of impact instead of ramping up from the
        // upright rotation=0 the fall phase ends on -- a visible teleport,
        // not a stumble) plus an exponential drag-style skid across the
        // ground (x(t) = x_inf * (1 - e^-t/tau), the closed-form solution
        // to linear drag), instead of a hand-authored fall pose -- so the
        // exact stumble direction/violence still varies per unit (see
        // tumblePitch/tumbleRoll/skidAngle/skidDist) without needing
        // per-frame velocity state (stepMilitaryPatrol recomputes every
        // unit's transform fresh from simTime each call, never integrates
        // frame-to-frame, so the physics has to be a closed-form function
        // of elapsed time, not an accumulated simulation).
        const it = ct - t.fallDuration;
        const zeta = 3.0, omega = 10.0;
        const decay = Math.exp(-zeta * it);
        const wobble = decay * Math.sin(omega * it);
        const skidFrac = 1 - Math.exp(-it / 0.4);
        const sx = Math.cos(t.skidAngle) * t.skidDist * skidFrac;
        const sz = Math.sin(t.skidAngle) * t.skidDist * skidFrac;
        const wp = toScene(t.dropX + sx, t.dropZ + sz);
        t.group.position.set(wp.x, 0, wp.z);
        t.group.rotation.set(t.tumblePitch * wobble, t.swaySeed, t.tumbleRoll * wobble);
        // Impact squash-and-stretch, same decay envelope as the tumble --
        // compresses on the initial hit, volume-preserving (widens as it
        // flattens) so it reads as absorbing impact rather than shrinking.
        const squash = 1 - 0.25 * decay;
        t.group.scale.set(1 / Math.sqrt(squash), squash, 1 / Math.sqrt(squash));
        t.group.visible = true;
        t.chuteGroup.visible = false; // shed the instant they hit the ground
      } else if (ct < t.fallDuration + t.impactDuration + t.standDuration) {
        const wp = toScene(t.dropX + Math.cos(t.skidAngle) * t.skidDist, t.dropZ + Math.sin(t.skidAngle) * t.skidDist);
        t.group.position.set(wp.x, 0, wp.z);
        t.group.rotation.set(0, t.swaySeed, 0);
        t.group.scale.setScalar(1);
        t.group.visible = true;
        t.chuteGroup.visible = false; // chute gathered/discarded after landing
      } else {
        t.group.visible = false; // brief hidden gap before the next drop
      }
      continue;
    }
    if (t.kind === 'air') {
      const loop = militaryAirLoops[t.loopIdx];
      if (!loop) continue;
      const dir = t.dir || 1;
      const tt = dir * (simTime / loop.period) + t.phase;
      const [wx, wy] = circleLoopPos(loop.cx, loop.cy, loop.r, tt);
      const [wx2, wy2] = circleLoopPos(loop.cx, loop.cy, loop.r, tt + 0.003 * dir);
      const hx = wx2 - wx, hy = wy2 - wy;
      const p = toScene(wx, wy);
      t.group.position.set(p.x, loop.alt, p.z);
      t.group.rotation.y = Math.atan2(hy, hx);
      t.group.rotation.z = -dir * 0.26; // bank into the turn
      continue;
    }
    if (t.kind === 'convoyAir') {
      const loop = militaryAirConvoyLoops[t.loopIdx];
      if (!loop) continue;
      const dir = t.dir || 1;
      const tt = dir * (simTime / loop.period) + t.phase;
      const [bx, by] = lineLoopPos(loop.x0, loop.y0, loop.x1, loop.y1, tt);
      const [bx2, by2] = lineLoopPos(loop.x0, loop.y0, loop.x1, loop.y1, tt + 0.002 * dir);
      const hx = bx2 - bx, hy = by2 - by;
      // Each plane's own random lateral offset (see spawnPatrolGroup's
      // lateralSpread, assigned into squadOffset) so the whole squadron
      // fans out across the corridor instead of flying in a single file.
      let wx = bx, wy = by;
      if (t.squadOffset) {
        const hlen = Math.hypot(hx, hy) || 1;
        wx += (-hy / hlen) * t.squadOffset;
        wy += (hx / hlen) * t.squadOffset;
      }
      const p = toScene(wx, wy);
      // Fixed cruise altitude for the whole trip -- no climb-out/descent
      // ramp, so aircraft never dip down to ground level at either end of
      // their route (previously read as taking off/landing there).
      t.group.position.set(p.x, loop.alt, p.z);
      t.group.rotation.y = Math.atan2(hy, hx);
      if (t.mainRotor) t.mainRotor.rotation.y += (dt || 0) * 18;
      if (t.tailRotor) t.tailRotor.rotation.x += (dt || 0) * 26;
      continue;
    }
    if (t.kind === 'convoyGround') {
      const loop = militaryConvoyLoops[t.loopIdx];
      if (!loop) continue;
      const dir = t.dir || 1;
      const tt = dir * (simTime / loop.period) + t.phase;
      const [bx, by] = lineLoopPos(loop.x0, loop.y0, loop.x1, loop.y1, tt);
      const [bx2, by2] = lineLoopPos(loop.x0, loop.y0, loop.x1, loop.y1, tt + 0.002 * dir);
      const hx = bx2 - bx, hy = by2 - by;
      const hlen = Math.hypot(hx, hy) || 1;
      const fwdX = hx / hlen, fwdY = hy / hlen;
      let wx = bx, wy = by;
      if (t.squadOffset || t.squadForwardOffset) {
        // squadOffset: each unit/squad's own random lateral position
        // across the corridor width (see spawnPatrolGroup/spawnSquads'
        // lateralSpread) so the whole group fans out instead of riding
        // single-file down the centre line. squadForwardOffset
        // additionally spreads a marching squad's members into its
        // rectangular block formation, front-to-back.
        const perpX = -fwdY, perpY = fwdX;
        wx += perpX * (t.squadOffset || 0) + fwdX * (t.squadForwardOffset || 0);
        wy += perpY * (t.squadOffset || 0) + fwdY * (t.squadForwardOffset || 0);
      }
      const p = toScene(wx, wy);
      t.group.position.set(p.x, 0, p.z);
      t.group.rotation.y = Math.atan2(hy, hx);
      // Dust trail (tanks/trucks only, see their build functions) --
      // anchor sits a few units behind the vehicle's rear, at the scaled
      // vehicle's actual footprint so it reads as kicked-up dust rather
      // than trailing off to one side.
      if (t.dustAnchor) {
        const trail = (t.dustTrailDist || 6) * MIL_UNIT_SCALE;
        t.dustAnchor.x = p.x - fwdX * trail;
        t.dustAnchor.y = 0.4;
        t.dustAnchor.z = p.z - fwdY * trail;
        t.dustAnchor.active = true;
      }
      // Tank turret independently scans while the hull drives straight --
      // sells "alert" rather than "prop being dragged along a rail".
      if (t.turretGroup) {
        t.turretGroup.rotation.y = Math.sin(simTime * 0.22 + t.phase * 11) * 0.55;
      }
      if (t.muzzle) {
        const cyc = (simTime + t.phase * 23) % 7;
        const on = cyc < 0.1;
        t.muzzle.visible = on;
        applyMuzzleFlash(t, cyc, on);
      }
      // Occasional hop over rough ground -- a different phase multiplier
      // (17) than the muzzle flash's (23) so jumping and firing don't
      // always land on the same beat. No jump animation clip exists in
      // Soldier.glb (only Idle/Run/TPose/Walk), so this is a plain
      // parabolic Y offset layered on top of the run cycle rather than a
      // real jump pose. Long period (30s) and short duration -- with ~150+
      // soldiers on screen, even a "rare per-soldier" hop adds up fast: at
      // the original 8s period roughly 8-9 were airborne at any instant,
      // which read as the whole formation twitching rather than one
      // soldier occasionally hopping a ditch.
      if (t.isSoldier) {
        const jcyc = (simTime + t.phase * 17) % 30;
        if (jcyc < 0.35) {
          t.group.position.y = Math.sin((jcyc / 0.35) * Math.PI) * 10;
        }
      }
      if (t.wheels) {
        // rotateZ, not setting rotation.x/z directly -- each wheel already
        // has a base rotation.x = PI/2 (laying the cylinder on its side to
        // look like a wheel); rotateZ spins it around its own local axle
        // axis on top of that base orientation instead of overwriting it.
        const spin = (dt || 0) * 30 * dir;
        for (const w of t.wheels) w.rotateZ(spin);
      }
      continue;
    }
    const loop = militaryLoops[t.loopIdx];
    if (!loop) continue;
    const dir = t.dir || 1;
    const tt = dir * (simTime / loop.period) + t.phase;
    let [wx, wy] = rectLoopPos(loop.cx, loop.cy, loop.hw, loop.hh, tt);
    const [wx2, wy2] = rectLoopPos(loop.cx, loop.cy, loop.hw, loop.hh, tt + 0.004 * dir);
    const hx = wx2 - wx, hy = wy2 - wy;
    if (t.squadOffset || t.squadForwardOffset) {
      // Shift perpendicular to (abreast) and along (rows front-to-back)
      // the direction of travel, so a squad reads as a marching block
      // instead of every member stacked on the exact same path point.
      const hlen = Math.hypot(hx, hy) || 1;
      const fwdX = hx / hlen, fwdY = hy / hlen;
      const perpX = -fwdY, perpY = fwdX;
      wx += perpX * (t.squadOffset || 0) + fwdX * (t.squadForwardOffset || 0);
      wy += perpY * (t.squadOffset || 0) + fwdY * (t.squadForwardOffset || 0);
    }
    const p = toScene(wx, wy);
    t.group.position.set(p.x, 0, p.z);
    t.group.rotation.y = Math.atan2(hy, hx);
    // Same muzzle-flash/jump cosmetic hooks as the convoyGround branch --
    // border guards (see the two loop-index-5/6 spawnPatrolGroup calls in
    // rebuildMilitaryPatrol) use this rectLoopPos path, not the straight
    // convoy line, but should still look/behave like every other soldier.
    if (t.muzzle) {
      const cyc = (simTime + t.phase * 23) % 7;
      const on = cyc < 0.1;
      t.muzzle.visible = on;
      applyMuzzleFlash(t, cyc, on);
    }
    if (t.isSoldier) {
      const jcyc = (simTime + t.phase * 17) % 30;
      if (jcyc < 0.35) {
        t.group.position.y = Math.sin((jcyc / 0.35) * Math.PI) * 10;
      }
    }
  }
  stepMilitaryGuns(simTime);
  stepMilitaryExplosions(simTime, dt || 0);
}

function pickStyle(tier, rng) {
  // "panel" (vivid flat-colour cladding) mixed into every tier so a
  // skyline isn't wall-to-wall grey stone/metal/glass -- real cities mix
  // in bold accent-coloured buildings too.
  if (tier === 1) return rng() < 0.65 ? 'glass' : 'panel';
  if (tier === 2) return ['brick', 'panel', 'glass'][(rng() * 3) | 0];
  if (tier === 3) return ['stone', 'panel', 'brick'][(rng() * 3) | 0];
  return rng() < 0.75 ? 'planks' : 'panel';
}

// A stepped "wedding cake" tower -- 2-3 tiers of shrinking footprint --
// instead of one flat-sided box, the classic setback-massing trick that
// makes a skyline silhouette actually read as a real tower.
function addSetbackTower(group, w, d, h, wallTex) {
  const tiers = h > 45 ? 3 : 2;
  const tierH = h / tiers;
  let y0 = 0, curW = w, curD = d;
  for (let t = 0; t < tiers; t++) {
    addBlockBox(group, curW, tierH, curD, wallTex, TEX.roof, y0 + tierH / 2);
    y0 += tierH;
    curW *= 0.78;
    curD *= 0.78;
  }
  return { topY: y0, topW: curW, topD: curD };
}

// A real triangular-prism gable roof (ridge + two slopes + two gable-end
// triangles) instead of a flat cap -- built as an explicit hand-indexed
// BufferGeometry rather than composing primitives, so the ridge/eave
// lines land exactly where the wall below expects them. DoubleSide
// avoids any risk of an invisible face from a wrong winding-order guess.
function makeGableRoofGeometry(w, d, ridgeH) {
  const hw = w / 2, hd = d / 2;
  const v = new Float32Array([
    -hw, ridgeH, 0, hw, ridgeH, 0,   // 0,1 ridge
    -hw, 0, -hd, hw, 0, -hd,         // 2,3 front eave
    -hw, 0, hd, hw, 0, hd,           // 4,5 back eave
  ]);
  const idx = [0, 2, 3, 0, 3, 1, 0, 1, 5, 0, 5, 4, 0, 4, 2, 1, 3, 5];
  const uv = new Float32Array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0]);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(v, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}
function addGableRoof(parent, w, d, ridgeH, baseY, roofTex) {
  const geo = makeGableRoofGeometry(w * 1.1, d * 1.1, ridgeH);
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05, 
    map: tiledClone(roofTex, Math.max(1, Math.round(w / 4)), Math.max(1, Math.round(ridgeH / 3))),
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.y = baseY;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
}

// tier -> [minH, maxH] in metres. Industrial stays human/industrial-
// scaled (its chimneys are the tall landmark, see addBuilding below);
// everything else is a genuine skyscraper-to-low-rise vertical range.
const HEIGHT_RANGE = {
  industrial: { 1: [45, 75], 2: [32, 55], 3: [20, 38], 4: [14, 26] },
  city: { 1: [140, 260], 2: [80, 150], 3: [45, 85], 4: [22, 45] },
};

function addBuilding(parent, o, idx, environment) {
  const isIndustrial = environment === 'Industrial Site';
  const tier = o.loss_db >= 27 ? 1 : o.loss_db >= 19 ? 2 : o.loss_db >= 11 ? 3 : 4;
  const rng = mulberry32(hashSeed(String(o.label || idx)));

  // The obstacle rect itself is sized as a fraction of the current node
  // scatter extent in the 2D scene (topology_canvas.py's _bg_*), so it
  // can be anywhere from ~150m to ~900m+ wide -- for a compact "vertical
  // world" instead of vast flat slabs, the VISUAL footprint is its own
  // small, seeded (so it's stable across rebuilds) size at the obstacle's
  // true centre, and height comes from a tier-based range biased tall
  // rather than a flat multiple of the (possibly huge) footprint. The
  // real obstacle rect still drives the actual RF-obstruction math in
  // sim.obstacles; only what gets drawn here is resized.
  const w = isIndustrial ? 20 + rng() * 30 : 20 + rng() * 24;
  const d = isIndustrial ? 15 + rng() * 22 : 18 + rng() * 20;
  const [hLo, hHi] = (isIndustrial ? HEIGHT_RANGE.industrial : HEIGHT_RANGE.city)[tier];
  const h = hLo + rng() * (hHi - hLo);

  const label = String(o.label || '');
  const isPlant = isIndustrial && /plant|process|unit/i.test(label);
  const isOffice = isIndustrial && /office|control/i.test(label);

  const group = new THREE.Group();

  const style = isIndustrial ? (isOffice ? 'stone' : 'metal') : pickStyle(tier, rng);
  const wallTex = pickWallVariant(style, rng);

  if (!isIndustrial && (tier === 1 || tier === 2) && h > 26) {
    addSetbackTower(group, w, d, h, wallTex);
  } else if (!isIndustrial && (tier === 3 || tier === 4) && rng() < 0.7) {
    // Pitched roof instead of a flat cap -- reads far more like a real
    // building than another flat-topped box.
    const wallH = h * 0.7;
    addBlockBox(group, w, wallH, d, wallTex, TEX.roof, wallH / 2);
    addGableRoof(group, w, d, h - wallH, wallH, TEX.roof_shingle);
  } else {
    addBlockBox(group, w, h, d, wallTex, TEX.roof, h / 2);
    addRoofClutter(group, w, d, h, rng);
  }

  if (!isIndustrial && tier === 1) {
    // Mechanical penthouse + beacon light -- the skyline landmark cube.
    const ph = Math.min(6, h * 0.14);
    addBlockBox(group, w * 0.42, ph, d * 0.42, TEX.metal, TEX.roof, h + ph / 2);
    const beaconMat = new THREE.MeshBasicMaterial({ color: 0xff3b30 });
    const beacon = new THREE.Mesh(beaconGeo, beaconMat);
    beacon.position.y = h + ph + 1.2;
    group.add(beacon);
  }

  let plantStack = null;
  if (isPlant) {
    // Blocky smokestack (tapered box column, hazard-striped) -- real
    // emitted smoke handled globally by smoke.js via the world-space
    // anchor registered below, plus a small ground-level tank farm.
    const stackH = Math.max(24, h * 1.5 + rng() * 16);
    const stackR = 2.0 + rng() * 0.6;
    const localX = w * 0.26, localZ = d * 0.18;
    const stackMat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.6, color: 0x45484d });
    const stack = new THREE.Mesh(new THREE.BoxGeometry(stackR * 1.6, stackH, stackR * 1.6), stackMat);
    stack.position.set(localX, h + stackH / 2, localZ);
    stack.castShadow = true;
    group.add(stack);
    const bandMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.5, map: tiledClone(TEX.warning, 1, 1) });
    for (const f of [0.6, 0.85]) {
      const band = new THREE.Mesh(new THREE.BoxGeometry(stackR * 1.8, stackH * 0.08, stackR * 1.8), bandMat);
      band.position.set(localX, h + stackH * f, localZ);
      group.add(band);
    }
    plantStack = { localX, localZ, topY: h + stackH + 1 };

    const tankR = Math.min(w, d) * 0.07 + 1.7;
    const tankH = 7 + rng() * 4;
    const tankX = -w / 2 - tankR - 3;
    addTank(group, tankX, -d * 0.2, tankR, tankH, rng);
    addTank(group, tankX - tankR * 2.3, -d * 0.2, tankR * 0.85, tankH * 0.82, rng);
    addTank(group, tankX - tankR * 4.5, -d * 0.2, tankR * 0.92, tankH * 0.68, rng);
  }

  const c = toScene((o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2);
  group.position.set(c.x, 0, c.z);
  parent.add(group);
  if (plantStack) {
    registerSmokestack({ x: c.x + plantStack.localX, y: plantStack.topY, z: c.z + plantStack.localZ });
  }
  registerFootprint(c.x, c.z, w, d, h);
}

// Only 4-5 real obstacles exist per Industrial Site variant (they're the
// actual RF-obstruction rects the Python 2D scene registers), which reads
// as a nearly-empty complex and makes every variant look the same --
// three chimneys/warehouses/offices are the same three grey boxes
// regardless of which layout is picked. These filler buildings are pure
// scenery (no sim.obstacles entry, no RF effect) that pad the site out
// and, more importantly, differ per variant so "Logistics Park" actually
// looks different from "Business Park" instead of just being a label.
function addFillerBuilding(parent, x, z, rng, styleBias, withChimney) {
  const tier = rng() < 0.4 ? 3 : 4;
  const w = 16 + rng() * 22, d = 13 + rng() * 16;
  const [hLo, hHi] = HEIGHT_RANGE.industrial[tier];
  const h = hLo + rng() * (hHi - hLo);
  const wallTex = pickWallVariant(styleBias(rng), rng);

  const group = new THREE.Group();
  if (rng() < 0.6) {
    const wallH = h * 0.7;
    addBlockBox(group, w, wallH, d, wallTex, TEX.roof, wallH / 2);
    addGableRoof(group, w, d, h - wallH, wallH, TEX.roof_shingle);
  } else {
    addBlockBox(group, w, h, d, wallTex, TEX.roof, h / 2);
    addRoofClutter(group, w, d, h, rng);
  }

  // Probability tuned down from 0.5 alongside the ~2x filler-count bump so
  // smoke.js still ticks roughly the same number of emitters.
  if (withChimney && rng() < 0.3) {
    const stackH = 18 + rng() * 22, stackR = 1.2 + rng() * 0.5;
    const localX = w * 0.28, localZ = d * 0.2;
    const stack = new THREE.Mesh(
      new THREE.BoxGeometry(stackR * 1.6, stackH, stackR * 1.6),
      new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.6, color: 0x45484d }),
    );
    stack.position.set(localX, h + stackH / 2, localZ);
    stack.castShadow = true;
    group.add(stack);
    registerSmokestack({ x: x + localX, y: h + stackH + 1, z: z + localZ });
  }

  group.position.set(x, 0, z);
  parent.add(group);
  registerFootprint(x, z, w, d, h);
}

function fillerLayoutFor(variant) {
  if (variant === 3) {
    // Business Park -- an orderly grid of office-style low-rises.
    return { count: 38, chimneys: false, grid: true, styleBias: (rng) => (rng() < 0.5 ? 'stone' : 'panel') };
  }
  if (variant === 2) {
    // Process Plant -- scattered mixed sheds, many with their own stacks.
    return { count: 32, chimneys: true, grid: false, styleBias: (rng) => (rng() < 0.7 ? 'metal' : 'stone') };
  }
  // Logistics Park (variant 1, and the default) -- scattered plain warehouses.
  return { count: 36, chimneys: false, grid: false, styleBias: (rng) => (rng() < 0.6 ? 'metal' : 'planks') };
}

function addFillerBuildings(parent, rng, R, variant, loops) {
  const layout = fillerLayoutFor(variant);
  // addFillerBuilding's own footprint can be up to ~38m wide (w = 16 +
  // rng()*22) -- floors below keep placements from landing closer than a
  // real industrial plot allows regardless of how small R is, the same
  // reasoning as addCityFillerBuildings' cell floor.
  if (layout.grid) {
    const cols = 6, rows = Math.ceil(layout.count / cols);
    const spacing = Math.max(55, (R * 0.85) / Math.max(cols, rows));
    let i = 0;
    for (let r = 0; r < rows && i < layout.count; r++) {
      for (let c = 0; c < cols && i < layout.count; c++, i++) {
        const x = (c - (cols - 1) / 2) * spacing * 1.7 + (rng() - 0.5) * 8;
        const z = (r - (rows - 1) / 2) * spacing * 1.7 + (rng() - 0.5) * 8;
        // Grid cells the ring roads pass through stay empty -- reads as
        // blocks fronting the road rather than buildings on the pavement.
        if (!clearOfRoads(x, z, loops, 8)) continue;
        // Keep the AP's fixed origin clear -- the annulus-sampled branch
        // below already starts its inner radius at R*0.12, but this grid
        // branch has no such gap and could otherwise drop a filler
        // building right on top of the AP mast.
        if (Math.hypot(x, z) < 45) continue;
        addFillerBuilding(parent, x, z, rng, layout.styleBias, layout.chimneys);
      }
    }
  } else {
    // scatterPos only checks road clearance, not building-to-building
    // distance -- pure random scatter could (and did) land two ~35m-wide
    // warehouses close enough to visually merge. Rejection-sample against
    // every already-placed filler in this pass instead (same pattern as
    // the military squad-line/soldier-cluster placement elsewhere in this
    // file): up to 10 tries, keep the last if none clears the minimum.
    const minGap = 60;
    const placedAt = [];
    for (let i = 0; i < layout.count; i++) {
      let x, z;
      for (let tries = 0; tries < 10; tries++) {
        [x, z] = scatterPos(rng, R * 0.12, R * 0.85, loops, 13);
        if (placedAt.every(p => Math.hypot(p[0] - x, p[1] - z) >= minGap)) break;
      }
      placedAt.push([x, z]);
      addFillerBuilding(parent, x, z, rng, layout.styleBias, layout.chimneys);
    }
  }
}

// Smart City's real obstacles are only the handful of RF-registered
// towers (~8-10 rects), which reads as a hamlet with skyscrapers. Same
// trick as the industrial filler: pure-scenery towers (no sim.obstacles
// entry, no RF effect) packed into a block grid -- reusing the exact
// pickStyle/HEIGHT_RANGE.city/setback-massing machinery the real city
// buildings already get, so filler and real towers are indistinguishable.
function addCityFillerBuilding(parent, x, z, rf, rng, variant = 1, heightMul = 1) {
  // rf: distance from the core as a fraction of the site radius -- the
  // classic downtown taper, tallest at the centre, low-rise at the edge.
  // Business District (variant 2) skews the whole taper taller -- the
  // same distance from the core buys a taller tier than in Downtown Grid,
  // so the skyline reads as denser/taller even with the real towers fixed
  // at the same positions in both variants.
  const tallBias = variant === 2 ? 0.16 : 0;
  const tier = rf < 0.58 + tallBias ? (rng() < 0.5 ? 1 : 2)
    : rf < 0.82 + tallBias ? (rng() < 0.5 ? 2 : 3)
    : rng() < 0.55 ? 3 : 4;
  const w = 17 + rng() * 17, d = 15 + rng() * 16;
  const [hLo, hHi] = HEIGHT_RANGE.city[tier];
  // heightMul: Suburban Corridor (see addSuburbanCorridorFiller) pins rf
  // past every threshold above to force tier 3/4 every time, but even
  // HEIGHT_RANGE.city's tier 3/4 (45-85m / 22-45m) is still a genuinely
  // tall building, not the "low-rise commercial" a suburban corridor is
  // supposed to read as -- those figures are calibrated as the SHORTEST
  // tier of a downtown skyline, not an absolute "low-rise" height. Without
  // this, Suburban Corridor rendered as just another tall-building
  // district, visually indistinguishable from Downtown Grid/Business
  // District despite tier selection correctly forcing 3/4 the whole time
  // -- the bug was in what tier 3/4 actually MEANS in metres, not in
  // which tier got picked. Left at 1 (no change) for Downtown Grid/
  // Business District's own tier-3/4 filler, which legitimately are the
  // shorter buildings of an otherwise tall skyline.
  const h = (hLo + rng() * (hHi - hLo)) * heightMul;
  const wallTex = pickWallVariant(pickStyle(tier, rng), rng);
  const group = new THREE.Group();
  if ((tier === 1 || tier === 2) && h > 26) {
    const top = addSetbackTower(group, w, d, h, wallTex);
    if (tier === 1 && rng() < 0.55) {
      const beacon = new THREE.Mesh(beaconGeo, new THREE.MeshBasicMaterial({ color: 0xff3b30 }));
      beacon.position.y = top.topY + 1.2;
      group.add(beacon);
    }
  } else if (rng() < 0.65) {
    const wallH = h * 0.7;
    addBlockBox(group, w, wallH, d, wallTex, TEX.roof, wallH / 2);
    addGableRoof(group, w, d, h - wallH, wallH, TEX.roof_shingle);
  } else {
    addBlockBox(group, w, h, d, wallTex, TEX.roof, h / 2);
    addRoofClutter(group, w, d, h, rng);
  }
  group.position.set(x, 0, z);
  parent.add(group);
  registerFootprint(x, z, w, d, h);
}

// Suburban Corridor (Smart City variant 3): a single wide boulevard lined
// with low-rise commercial buildings and surface parking lots, with green
// verges between the road and the building line -- no downtown tower grid
// at all, so the fixed real campus towers (which still get drawn by the
// normal per-obstacle pass in rebuildBuildings, untouched by variant) end
// up reading as a small isolated cluster in an otherwise spread-out,
// low-rise district rather than "downtown again." Walks the single
// flattened boulevard loop addCityFillerBuildings's caller now builds for
// this variant (see _road_loops in snapshot.py), alternating a low-rise
// building or a parking lot on each side, same "reuse the existing
// per-lot builder, just steer its inputs" trick as the rest of this file.
function addSuburbanCorridorFiller(parent, rng, loops, obstacles) {
  const l = loops[0];
  if (!l) return;
  const avoid = (obstacles || [])
    .filter(o => o.kind !== 'circle')
    .map(o => toScene((o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2));
  const perim = 2 * (2 * l.hw + 2 * l.hh);
  const spacing = 46;
  const n = Math.max(10, Math.round(perim / spacing));
  const setback = ROAD_HALF_W + SIDEWALK_W + 16;
  for (let i = 0; i < n; i++) {
    for (const side of [1, -1]) {
      const t = (i + (side === 1 ? 0 : 0.5)) / n;
      const p = loopEdgePoint(l, t, side * setback);
      if (Math.hypot(p.x, p.z) < 60) continue; // keep the AP + real towers' own clearing free
      if (avoid.some(a => Math.hypot(p.x - a.x, p.z - a.z) < 65)) continue;
      // Green verge between the road and the building/parking line.
      const vp = loopEdgePoint(l, t, side * (ROAD_HALF_W + SIDEWALK_W + 5));
      addGrassPatch(vp.x, vp.z, spacing * 0.92, 9);
      if (rng() < 0.6) {
        // rf pinned past addCityFillerBuilding's tower thresholds forces
        // tier 3/4 every time; heightMul=0.24 additionally scales tier
        // 3/4's own height range down from "shortest building in a tall
        // skyline" (45-85m/22-45m) to an actual low-rise commercial
        // height (~5-20m) -- see addCityFillerBuilding's own comment for
        // why both were needed (right tier, wrong metres).
        addCityFillerBuilding(parent, p.x, p.z, 0.95, rng, 3, 0.24);
        if (rng() < 0.5) addTree(p.x + Math.cos(p.heading) * 9, p.z - Math.sin(p.heading) * 9, 0.55 + rng() * 0.4);
      } else {
        addParkingLot(p.x, p.z, p.heading + side * Math.PI / 2, 4 + ((rng() * 3) | 0), rng);
      }
    }
  }
}

function addCityFillerBuildings(parent, rng, R, loops, obstacles, variant) {
  if (variant === 3) {
    // Suburban Corridor: no downtown tower grid at all -- a single
    // low-rise strip lining the boulevard loop instead (see
    // addSuburbanCorridorFiller), so the fixed real towers at the centre
    // end up reading as an isolated cluster rather than "downtown again."
    addSuburbanCorridorFiller(parent, rng, loops, obstacles);
    return;
  }
  const inner = loops.reduce((a, b) => (a.hw * a.hh <= b.hw * b.hh ? a : b));
  const outer = loops.reduce((a, b) => (a.hw * a.hh >= b.hw * b.hh ? a : b));
  const ccx = inner.cx, ccz = -inner.cy;
  const avoid = (obstacles || [])
    .filter(o => o.kind !== 'circle')
    .map(o => toScene((o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2));
  // v1 (Downtown Grid) packs the tightest of the two downtown variants;
  // v2 (Business District) packs tighter still -- every block gets a
  // tower, no parks -- so the variants read as different downtowns, not
  // one relabelled map.
  //
  // Floor was 30 -- addCityFillerBuilding's own footprint can be up to
  // ~34m wide (w = 17 + rng()*17), so at the floor, adjacent towers could
  // literally overlap. This floor is what actually governed density for
  // most realistic node counts (R commonly lands well under the
  // R/divisor crossover, e.g. ~390 for variant 1), not the siteRadius()
  // cap -- raising that cap alone (see siteRadius's own comment) only
  // helps the less common large-scatter case. 85 gives clearance for the
  // widest tower plus both sidewalks (SIDEWALK_W=4.5 each side) plus the
  // road (ROAD_HALF_W=9.5 => 19m wide) with room to spare, closer to a
  // real city block's center-to-center spacing than a floor narrower
  // than the buildings sitting on it. Business District's cell is
  // tighter still than either -- a genuinely denser packed core.
  const cell = Math.max(78, R / (variant === 2 ? 15 : 13));
  // Raised from the original 6-16% and mostly greened below (grass patch
  // + trees/bench/garden) instead of left bare -- real downtown blocks
  // aren't wall-to-wall towers, and the user asked for the interior to
  // read as leafier, not just the dedicated plaza. Business District
  // (variant 2) is the one downtown variant that's explicitly "every
  // block holds a tower (no parks)" per its 2D design -- vacancy stays
  // near zero there instead.
  const vacancy = variant === 2 ? 0.04 : 0.30;
  const half = Math.ceil((R * 1.05) / cell);
  let placed = 0;
  for (let gx = -half; gx <= half && placed < 320; gx++) {
    for (let gz = -half; gz <= half && placed < 320; gz++) {
      const x = ccx + gx * cell + (rng() - 0.5) * cell * 0.3;
      const z = ccz + gz * cell + (rng() - 0.5) * cell * 0.3;
      const rf = Math.hypot(x - ccx, z - ccz) / R;
      if (rf > 1.02) continue;
      // R (cityExtentR) is sized to comfortably cover the outer loop's
      // extent so the filler grid never falls entirely inside the plaza
      // exclusion zone (see cityExtentR's own comment) -- but R itself is
      // a single radius, and a rectangular loop's corners sit farther out
      // than its edges, so an unclamped circular rf test would spill
      // filler well past the actual drawn road network near the edges.
      // Stop at the outer loop's own rectangle (plus a sliver of setback
      // for the block fronting it) instead -- a large margin here just
      // grows a second, road-less ring of towers floating past the last
      // drawn road, which looks worse than stopping cleanly at the kerb.
      if (Math.abs(x - ccx) > outer.hw + 12 || Math.abs(z - ccz) > outer.hh + 12) continue;
      // The inner-loop block stays open -- that's the civic plaza
      // (fountain, benches, the AP mast), not another tower lot.
      if (Math.abs(x - ccx) < inner.hw - ROAD_HALF_W && Math.abs(z - ccz) < inner.hh - ROAD_HALF_W) continue;
      // The AP is fixed at world/scene origin regardless of where the
      // node scatter (and therefore this plaza) ends up -- a lopsided
      // topology can leave the origin outside the inner loop entirely, so
      // without this a filler tower could spawn right on top of the AP
      // mast and bury it. Keep a clearing around it independent of the
      // plaza check above.
      if (Math.hypot(x, z) < 45) continue;
      if (!clearOfRoads(x, z, loops, 13)) continue;
      if (avoid.some(a => Math.hypot(x - a.x, z - a.z) < 32)) continue;
      if (rng() < vacancy) {
        if (rng() < 0.88) {
          const gw = cell * (0.7 + rng() * 0.2), gd = cell * (0.6 + rng() * 0.2);
          addGrassPatch(x, z, gw, gd);
          const nTree = 3 + ((rng() * 4) | 0);
          for (let i = 0; i < nTree; i++) {
            const a = rng() * Math.PI * 2, rr = rng() * Math.min(gw, gd) * 0.4;
            addTree(x + Math.cos(a) * rr, z + Math.sin(a) * rr, 0.55 + rng() * 0.5);
          }
          if (rng() < 0.7) addGardenBed(x + (rng() - 0.5) * gw * 0.5, z + (rng() - 0.5) * gd * 0.5, rng() * Math.PI * 2, rng);
          if (rng() < 0.65) addBench(x + (rng() - 0.5) * gw * 0.5, z + (rng() - 0.5) * gd * 0.5, rng() * Math.PI * 2);
        }
        continue;
      }
      addCityFillerBuilding(parent, x, z, rf, rng, variant);
      placed++;
      // A tree at roughly a third of occupied lots too -- softens the
      // building base rather than leaving every tower on bare concrete.
      if (rng() < 0.6) {
        const a = rng() * Math.PI * 2, rr = cell * 0.38;
        addTree(x + Math.cos(a) * rr, z + Math.sin(a) * rr, 0.5 + rng() * 0.4);
        if (rng() < 0.4) {
          const a2 = a + Math.PI + (rng() - 0.5);
          addTree(x + Math.cos(a2) * rr, z + Math.sin(a2) * rr, 0.5 + rng() * 0.4);
        }
      }
    }
  }
}

let lastObstacleKey = '';
export function rebuildBuildings(obstacles, environment, variant, roadLoops) {
  const key = environment + '|' + variant + '|' + JSON.stringify(obstacles.map(o =>
    [o.kind, o.x0 ?? o.cx, o.y0 ?? o.cy, o.x1 ?? o.r, o.y1 ?? 0, o.loss_db, o.label]))
    + cityLoopSig(environment, roadLoops);
  if (key === lastObstacleKey) return;
  lastObstacleKey = key;

  for (const child of [...buildingsGroup.children]) {
    buildingsGroup.remove(child);
    child.traverse((o) => {
      o.geometry?.dispose?.();
      const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
      for (const m of mats) { if (m.map) m.map.dispose(); m.dispose?.(); }
    });
  }
  clearSmokestacks();
  buildingFootprints = [];

  obstacles.forEach((o, idx) => {
    if (o.kind === 'circle') addRock(buildingsGroup, o, idx);
    else if (environment === 'Industrial Site' && String(o.label) === 'tank-farm') addTankFarm(buildingsGroup, o);
    else if (environment === 'Military Zone') addMilitaryStructure(buildingsGroup, o, idx);
    else addBuilding(buildingsGroup, o, idx, environment);
  });

  if (environment === 'Industrial Site') {
    const rng = mulberry32(hashSeed('filler|' + variant + '|' + obstacles.length));
    addFillerBuildings(buildingsGroup, rng, siteRadius(obstacles), variant, industrialRoadLoops(obstacles, variant));
  } else if (environment === 'Smart City' && (roadLoops || []).length) {
    const rng = mulberry32(hashSeed('cityfiller|' + variant + '|' + obstacles.length));
    addCityFillerBuildings(buildingsGroup, rng, cityExtentR(obstacles, roadLoops), roadLoops, obstacles, variant);
  }
}

// ---- roads -- built directly from road_loops, the exact same rectangles
// vehicles drive (see ui/web3d/snapshot.py::_road_loops), so a car can
// never visually stray off the pavement: the road IS the path -----------
// 1.5x wider than the original 9.5/4.5 (per explicit request, to match
// the vehicles' own 200% size bump) -- capped at 1.5x, not more: the
// paved band for a lane at car_lane_offset_m=25 spans lane_y +/- (
// ROAD_HALF_W+SIDEWALK_W)=21 either side, i.e. [4,46], leaving an 8m
// unpaved median before the opposite-direction lane's own [-46,-4] band
// starts -- any wider and the two lanes' pavement would start
// overlapping right at the centreline. cityLoopSig's avenue hw stagger
// (ui/web3d/snapshot.py's _road_loops) was bumped from 40 to 50 to
// stay clear of this band's new total width (2 x 21 = 42), so avenue
// rings still can't z-fight at their end-caps the way fixing that
// depended on.
const ROAD_HALF_W = 14.0, SIDEWALK_W = 7.0;
function ringShape(cx, cy, hw, hh, halfWidth) {
  const shape = new THREE.Shape();
  shape.moveTo(cx - hw - halfWidth, cy - hh - halfWidth);
  shape.lineTo(cx + hw + halfWidth, cy - hh - halfWidth);
  shape.lineTo(cx + hw + halfWidth, cy + hh + halfWidth);
  shape.lineTo(cx - hw - halfWidth, cy + hh + halfWidth);
  shape.closePath();
  const hole = new THREE.Path();
  hole.moveTo(cx - hw + halfWidth, cy - hh + halfWidth);
  hole.lineTo(cx - hw + halfWidth, cy + hh - halfWidth);
  hole.lineTo(cx + hw - halfWidth, cy + hh - halfWidth);
  hole.lineTo(cx + hw - halfWidth, cy - hh + halfWidth);
  hole.closePath();
  shape.holes.push(hole);
  return shape;
}
function ringMesh(cx, cy, hw, hh, halfWidth, material, y) {
  const geo = new THREE.ShapeGeometry(ringShape(cx, cy, hw, hh, halfWidth), 1);
  const mesh = new THREE.Mesh(geo, material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = y;
  mesh.receiveShadow = true;
  return mesh;
}
function centerlineLoop(cx, cy, hw, hh, y) {
  const pts = [
    toScene(cx - hw, cy - hh), toScene(cx + hw, cy - hh),
    toScene(cx + hw, cy + hh), toScene(cx - hw, cy + hh),
    toScene(cx - hw, cy - hh),
  ].map(p => { p.y = y; return p; });
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineDashedMaterial({ color: 0xe8c547, dashSize: 6, gapSize: 5 });
  const line = new THREE.Line(geo, mat);
  line.computeLineDistances();
  return line;
}
// Tracks only the meshes THIS function itself added, not roadGroup's
// full child list -- rebuildCrossStreets shares the same roadGroup (so
// avenues and cross streets composite into one road surface), and a
// blind "clear every child of roadGroup" here would wipe those out too
// whenever this runs after that, and vice versa.
let roadMeshes = [];
let lastRoadKey = '';
export function rebuildRoads(loops) {
  const key = JSON.stringify(loops);
  if (key === lastRoadKey) return;
  lastRoadKey = key;
  for (const child of roadMeshes) {
    roadGroup.remove(child);
    child.geometry?.dispose();
    if (child.material?.map) child.material.map.dispose();
    child.material?.dispose?.();
  }
  roadMeshes = [];
  for (const l of loops) {
    const perim = 2 * (2 * l.hw + 2 * l.hh);
    const rep = Math.max(1, perim / 24);
    // Layer heights spread 0.4 apart, not the original 0.03 -- still
    // visually flush (ground=0, sidewalk=0.4, asphalt=0.8, centreline
    // 1.2 all read as flat pavement at any normal viewing distance), but
    // a standard depth buffer's precision drops fast with camera
    // distance, and three near-coplanar layers only 0.03 apart could
    // fall into the same depth-buffer bucket once zoomed out over this
    // scene's now much bigger (1100m+) footprint -- exactly what reads
    // as flicker ("z-fighting") while zooming. A wider real gap between
    // layers is robust to that regardless of how far the camera is.
    const sidewalkMat = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, map: tiledClone(TEX.sidewalk, rep, rep) });
    const sw = ringMesh(l.cx, l.cy, l.hw, l.hh, ROAD_HALF_W + SIDEWALK_W, sidewalkMat, 0.4);
    roadGroup.add(sw); roadMeshes.push(sw);
    const asphaltMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, map: tiledClone(TEX.road, rep, rep) });
    const rd = ringMesh(l.cx, l.cy, l.hw, l.hh, ROAD_HALF_W, asphaltMat, 0.8);
    roadGroup.add(rd); roadMeshes.push(rd);
    const cl = centerlineLoop(l.cx, l.cy, l.hw, l.hh, 1.2);
    roadGroup.add(cl); roadMeshes.push(cl);
  }
}

// ---- perpendicular cross streets (cars_uavs mode only -- see
// ui/web3d/snapshot.py's _cross_streets) -- solid rectangular strips,
// not rings: a cross street is narrow across and tall along y, where
// ringShape's hole-in-a-rectangle construction would need a hole nearly
// as wide as the road itself (no road left at all) to make sense the
// way it does for the avenues' own long, thin rings. Composites into
// the exact same roadGroup as rebuildRoads (see roadMeshes' own comment
// for why each tracks only its own children), so avenues and cross
// streets read as one continuous paved network with real "+"
// intersections where they cross, not two independent road systems
// layered on top of each other. -------------------------------------------
function stripShape(cx, cyMin, cyMax, halfWidth) {
  const shape = new THREE.Shape();
  shape.moveTo(cx - halfWidth, cyMin);
  shape.lineTo(cx + halfWidth, cyMin);
  shape.lineTo(cx + halfWidth, cyMax);
  shape.lineTo(cx - halfWidth, cyMax);
  shape.closePath();
  return shape;
}
function stripMesh(cx, cyMin, cyMax, halfWidth, material, y) {
  const geo = new THREE.ShapeGeometry(stripShape(cx, cyMin, cyMax, halfWidth), 1);
  const mesh = new THREE.Mesh(geo, material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = y;
  mesh.receiveShadow = true;
  return mesh;
}
function centerlineStrip(cx, cyMin, cyMax, y) {
  const pts = [toScene(cx, cyMin), toScene(cx, cyMax)].map(p => { p.y = y; return p; });
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineDashedMaterial({ color: 0xe8c547, dashSize: 6, gapSize: 5 });
  const line = new THREE.Line(geo, mat);
  line.computeLineDistances();
  return line;
}
let crossStreetMeshes = [];
let lastCrossKey = '';
// overbridge (ui/web3d/snapshot.py's _overbridge) picks exactly one cross
// street to carry a raised interchange over one avenue instead of a flat
// "+" crossing -- see rebuildOverbridge below, which draws the actual
// ramp/deck/pier geometry. This function's own job for THAT one cross
// street is just to stop drawing flat ground-level pavement through the
// stretch the bridge now occupies (a flat strip AND an elevated deck
// both existing at the same place would double up, and the flat one
// would only ever be seen poking out from directly underneath the deck).
export function rebuildCrossStreets(crossStreets, overbridge) {
  const key = JSON.stringify([crossStreets, overbridge]);
  if (key === lastCrossKey) return;
  lastCrossKey = key;
  for (const child of crossStreetMeshes) {
    roadGroup.remove(child);
    child.geometry?.dispose();
    if (child.material?.map) child.material.map.dispose();
    child.material?.dispose?.();
  }
  crossStreetMeshes = [];
  function addFlatSegment(x, yMin, yMax) {
    const span = Math.max(1, yMax - yMin);
    const rep = Math.max(1, span / 24);
    const sidewalkMat = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, map: tiledClone(TEX.sidewalk, rep, rep) });
    const sw = stripMesh(x, yMin, yMax, ROAD_HALF_W + SIDEWALK_W, sidewalkMat, 0.4);
    roadGroup.add(sw); crossStreetMeshes.push(sw);
    const asphaltMat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0, map: tiledClone(TEX.road, rep, rep) });
    const rd = stripMesh(x, yMin, yMax, ROAD_HALF_W, asphaltMat, 0.8);
    roadGroup.add(rd); crossStreetMeshes.push(rd);
    const cl = centerlineStrip(x, yMin, yMax, 1.2);
    roadGroup.add(cl); crossStreetMeshes.push(cl);
  }
  for (const cs of crossStreets || []) {
    const isBridgeStreet = overbridge && Math.abs(cs.x - overbridge.cross_x) < 1e-6;
    if (!isBridgeStreet) {
      addFlatSegment(cs.x, cs.y_min, cs.y_max);
      continue;
    }
    const gapHalf = overbridge.deck_half_len + overbridge.ramp_len;
    const gapMin = overbridge.avenue_y - gapHalf, gapMax = overbridge.avenue_y + gapHalf;
    if (cs.y_min < gapMin) addFlatSegment(cs.x, cs.y_min, gapMin);
    if (gapMax < cs.y_max) addFlatSegment(cs.x, gapMax, cs.y_max);
  }
}

// ---- overbridge: one cross street rises over one avenue instead of a
// flat "+" crossing -- ramp up, flat elevated deck, ramp down, on
// support piers, matching the gap rebuildCrossStreets carves out of
// that cross street's own flat pavement above. rampMesh builds a single
// sloped quad (unlike stripMesh's flat one) via an explicit
// BufferGeometry -- ShapeGeometry (what stripMesh/ringMesh use) is
// always planar, it has no way to give its 4 corners 2 different
// heights. side:DoubleSide is cheap insurance on a handful of ramp
// quads, not a sign the winding is actually expected to be wrong (it's
// been checked: yB > yA always holds at every call site here, which is
// what makes the computed face normal actually point up).
function rampMesh(cx, yA, yB, hA, hB, halfWidth, material) {
  const a0 = toScene(cx - halfWidth, yA); a0.y = hA;
  const a1 = toScene(cx + halfWidth, yA); a1.y = hA;
  const b0 = toScene(cx - halfWidth, yB); b0.y = hB;
  const b1 = toScene(cx + halfWidth, yB); b1.y = hB;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array([
    a0.x, a0.y, a0.z, a1.x, a1.y, a1.z, b1.x, b1.y, b1.z,
    a0.x, a0.y, a0.z, b1.x, b1.y, b1.z, b0.x, b0.y, b0.z,
  ]);
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]), 2));
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, material);
  mesh.receiveShadow = true;
  mesh.castShadow = true;
  return mesh;
}
let overbridgeMeshes = [];
let lastOverbridgeKey = '';
// Live copy of whatever overbridge config is currently active, read by
// bridgeDeckHeightAt below (entities.js calls it through that export,
// not this variable directly, same "module owns its own state, exposes
// a lookup function" shape as buildingFootprints/roofHeightAt above).
let currentOverbridge = null;
export function rebuildOverbridge(overbridge) {
  const key = JSON.stringify(overbridge);
  if (key === lastOverbridgeKey) return;
  lastOverbridgeKey = key;
  currentOverbridge = overbridge || null;
  for (const child of overbridgeMeshes) {
    roadGroup.remove(child);
    child.geometry?.dispose();
    if (child.material?.map) child.material.map.dispose();
    child.material?.dispose?.();
  }
  overbridgeMeshes = [];
  if (!overbridge) return;
  const { avenue_y: ay, cross_x: cx, deck_half_len: dh, ramp_len: rl, height: h } = overbridge;
  const rampRep = Math.max(1, rl / 24);
  const deckRep = Math.max(1, (2 * dh) / 24);
  function addLayer(yA, yB, hA, hB, halfWidth, tex, rep, y0) {
    const mat = new THREE.MeshStandardMaterial({
      roughness: halfWidth > ROAD_HALF_W ? 0.9 : 0.85, metalness: 0,
      map: tiledClone(tex, rep, rep),
    });
    const m = rampMesh(cx, yA, yB, hA + y0, hB + y0, halfWidth, mat);
    roadGroup.add(m); overbridgeMeshes.push(m);
  }
  // Ramp up (ground -> deck height), flat deck, ramp down (deck height
  // -> ground) -- sidewalk layer (wider, y0=0.4) under asphalt (y0=0.8),
  // exactly mirroring the flat road's own two-layer pavement, just
  // sloped/elevated instead of flat-on-the-ground.
  for (const [halfWidth, rep, y0] of [[ROAD_HALF_W + SIDEWALK_W, rampRep, 0.4], [ROAD_HALF_W, rampRep, 0.8]]) {
    const tex = halfWidth > ROAD_HALF_W ? TEX.sidewalk : TEX.road;
    addLayer(ay - dh - rl, ay - dh, 0, h, halfWidth, tex, rep, y0);
    addLayer(ay + dh, ay + dh + rl, h, 0, halfWidth, tex, rep, y0);
  }
  for (const [halfWidth, rep, y0] of [[ROAD_HALF_W + SIDEWALK_W, deckRep, 0.4], [ROAD_HALF_W, deckRep, 0.8]]) {
    const tex = halfWidth > ROAD_HALF_W ? TEX.sidewalk : TEX.road;
    addLayer(ay - dh, ay + dh, h, h, halfWidth, tex, rep, y0);
  }
  // Piers -- placed just outside the avenue's own paved half-width
  // (ROAD_HALF_W+SIDEWALK_W=21) so they stand clear of its lanes, not
  // in the middle of them (deck_half_len=30 from snapshot.py leaves
  // exactly this room). One box per pier, from the ground up to just
  // under the deck's underside.
  const pierMat = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0.05, color: 0x9aa0a6 });
  const pierY = (ROAD_HALF_W + SIDEWALK_W) + 4;
  // h, not h-something -- h is the deck's own base height (the addLayer
  // calls above stack the thin 0.4/0.8 pavement layers ON TOP of h, same
  // as the flat road's own layers sit on top of ground level 0), so
  // flush with h is flush with the underside of the deck's structure,
  // not floating short of it with a visible gap.
  const pierTopY = h;
  for (const py of [ay - pierY, ay + pierY]) {
    const geo = new THREE.BoxGeometry(2 * ROAD_HALF_W * 0.7, pierTopY, 3.2);
    const pier = new THREE.Mesh(geo, pierMat);
    const scenePos = toScene(cx, py);
    pier.position.set(scenePos.x, pierTopY / 2, scenePos.z);
    pier.castShadow = true;
    pier.receiveShadow = true;
    roadGroup.add(pier); overbridgeMeshes.push(pier);
  }
}

// Real car/scooter elevation while it's actually on the bridge's avenue
// AND within the bridge's x-span -- entities.js's altitudeFor calls this
// for car/scooter nodes (which otherwise always render at ground level,
// see its own is_car/is_scooter branch) so a vehicle visibly rises onto
// the deck and back down instead of driving through it. Purely a
// function of live (x, y): works regardless of which way the vehicle is
// currently facing/turning, unlike trying to key off role/heading.
export function bridgeDeckHeightAt(x, y) {
  const ob = currentOverbridge;
  if (!ob) return null;
  // The bridge elevates the CROSS STREET (fixed x = cross_x, travels
  // along y) over the avenue (fixed y = avenue_y, travels along x) --
  // ramps/deck below are built along y at fixed x for exactly that
  // reason, so a vehicle only rises here if IT is the one on that cross
  // street (x close to cross_x), based on how far along y it is from
  // the avenue crossing. An avenue vehicle passing through the same
  // point stays flat -- it's the one going under, not over.
  if (Math.abs(x - ob.cross_x) > ROAD_HALF_W + SIDEWALK_W) return null;
  const dy = Math.abs(y - ob.avenue_y);
  if (dy <= ob.deck_half_len) return ob.height;
  if (dy <= ob.deck_half_len + ob.ramp_len) {
    const t = (dy - ob.deck_half_len) / ob.ramp_len;
    return ob.height * (1 - t);
  }
  return null;
}

// ---- Industrial Site roads + moving trucks -- Smart City gets its
// road_loops/vehicles from Python (see snapshot.py), but Industrial Site
// has no equivalent server-side road data, so both are computed here
// instead. Same "road IS the path" principle either way: the loops fed
// to rebuildRoads and the loops trucks drive are the exact same numbers,
// so a truck can never visually leave the pavement.
export function industrialRoadLoops(obstacles, variant = 1) {
  const R = siteRadius(obstacles);
  if (variant === 2) {
    // Process Plant -- one wide, flattened east-west haul road (mirrors
    // _bg_industrial_v2's single main road + short plant spur) instead of
    // v1's nested truck-yard rings, so the road itself reads as a
    // different kind of site before a single building is drawn.
    return [{ cx: 0, cy: 0, hw: R * 0.82, hh: R * 0.18 }];
  }
  if (variant === 3) {
    // Business Park -- one large, roughly-square campus perimeter ring
    // (road-and-parking loop around the office blocks), no inner yard
    // loop -- a calmer, single-loop layout instead of the truck yard's
    // nested rings or the plant's single haul road.
    return [{ cx: 0, cy: 0, hw: R * 0.62, hh: R * 0.54 }];
  }
  // Logistics Park (variant 1, and the default): tight inner loop around
  // the loading dock plus a wide perimeter loop -- the truck-yard loop.
  return [
    { cx: 0, cy: 0, hw: R * 0.34, hh: R * 0.28 },
    { cx: 0, cy: 0, hw: R * 0.66, hh: R * 0.58 },
  ];
}

function rectLoopPos(cx, cy, hw, hh, t) {
  const perim = 2 * (2 * hw + 2 * hh);
  const dist = (((t % 1) + 1) % 1) * perim;
  if (dist < 2 * hw) return [cx - hw + dist, cy + hh];
  let d2 = dist - 2 * hw;
  if (d2 < 2 * hh) return [cx + hw, cy + hh - d2];
  d2 -= 2 * hh;
  if (d2 < 2 * hw) return [cx + hw - d2, cy - hh];
  d2 -= 2 * hw;
  return [cx - hw, cy - hh + d2];
}

// Circular flight-path position for patrol aircraft -- planes fly loops,
// not rectangles, and rectLoopPos's straight-edge-with-sharp-corners path
// would look wrong for something airborne.
function circleLoopPos(cx, cy, r, t) {
  const a = (((t % 1) + 1) % 1) * Math.PI * 2;
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
}

// Progress fraction (0 = at x0,y0; 1 = at x1,y1) of a ping-pong walk along
// a line -- t in [0,1] is the outbound leg, [1,2] the return leg,
// repeating. Shared by lineLoopPos (the actual position) and the
// convoyAir altitude ramp (stepMilitaryPatrol), which needs to know how
// close a unit is to either endpoint regardless of its actual x/z.
function travelProgress(t) {
  const tt = ((t % 2) + 2) % 2;
  return tt <= 1 ? tt : 2 - tt;
}

// Ping-pong position along a straight line. Used for convoy routes that
// actually travel from one base to the other and back, rather than
// patrolling a fixed loop around a single point.
function lineLoopPos(x0, y0, x1, y1, t) {
  const p = travelProgress(t);
  return [x0 + (x1 - x0) * p, y0 + (y1 - y0) * p];
}

let trafficMeshes = [];
let trafficLoops = [];
let lastTrafficKey = '';
export function rebuildIndustrialTraffic(obstacles, environment, variant = 1) {
  if (environment !== 'Industrial Site') {
    if (trafficMeshes.length) {
      for (const t of trafficMeshes) vehiclesGroup.remove(t.group);
      trafficMeshes = [];
      trafficLoops = [];
    }
    lastTrafficKey = '';
    return;
  }
  const loops = industrialRoadLoops(obstacles, variant);
  const key = variant + '|' + JSON.stringify(loops);
  if (key === lastTrafficKey) return;
  lastTrafficKey = key;
  trafficLoops = loops.map((l, i) => ({ ...l, period: 26 + i * 16 }));

  for (const t of trafficMeshes) vehiclesGroup.remove(t.group);
  trafficMeshes = [];
  const rng = mulberry32(hashSeed('traffic|' + key));
  // Traffic mix differs by variant too: Business Park is office/commuter
  // traffic (cars only), Process Plant is dominated by heavy tanker/semi
  // hauls with only the odd car, Logistics Park keeps the original mixed
  // warehouse-yard fleet.
  const n = variant === 3 ? 10 : variant === 2 ? 14 : 16;
  for (let i = 0; i < n; i++) {
    const roll = rng();
    let built;
    if (variant === 3) built = buildCarMesh(rng);
    else if (variant === 2) built = roll < 0.55 ? buildSemiMesh(rng) : roll < 0.88 ? buildTruckMesh(rng) : buildCarMesh(rng);
    else built = roll < 0.3 ? buildSemiMesh(rng) : roll < 0.62 ? buildCarMesh(rng) : buildTruckMesh(rng);
    vehiclesGroup.add(built.group);
    trafficMeshes.push({
      ...built,
      loopIdx: i % trafficLoops.length,
      // Alternating direction in pairs plus evenly-stepped phases keeps
      // each (loop, direction) lane's vehicles spaced apart -- same-lane
      // vehicles share a speed, so the spacing never collapses.
      dir: (i & 2) ? -1 : 1,
      phase: i / n + rng() * 0.03,
    });
  }
}

export function stepIndustrialTraffic(simTime) {
  for (const t of trafficMeshes) {
    const loop = trafficLoops[t.loopIdx];
    if (!loop) continue;
    const dir = t.dir || 1;
    const tt = dir * (simTime / loop.period) + t.phase;
    const [wx, wy] = rectLoopPos(loop.cx, loop.cy, loop.hw, loop.hh, tt);
    const [wx2, wy2] = rectLoopPos(loop.cx, loop.cy, loop.hw, loop.hh, tt + 0.004 * dir);
    const hx = wx2 - wx, hy = wy2 - wy;
    const hl = Math.hypot(hx, hy) || 1;
    // Two-way traffic: a right-hand lane offset keeps the directions out
    // of each other's way, and 4.4 < ROAD_HALF_W keeps both lanes on the
    // asphalt. The heading sample straddles corners, so the offset swings
    // through them smoothly instead of jumping lanes.
    const p = toScene(wx + (hy / hl) * 4.4, wy - (hx / hl) * 4.4);
    t.group.position.set(p.x, 0, p.z);
    // Same convention as Smart City vehicles: local forward is +X, and
    // world heading = atan2(dy, dx) maps directly onto rotation.y once
    // positions go through toScene() -- no extra sign flip needed.
    t.group.rotation.y = Math.atan2(hy, hx);
  }
}
