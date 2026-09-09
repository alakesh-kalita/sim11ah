// Industrial smokestack plumes: blocky puffs (cubes, not spheres) spawned
// at world-space emitter anchors, rising/drifting/fading each frame.
// Anchors are owned here so buildings.js only needs to register/clear
// them, not manage the particle lifecycle itself.
import * as THREE from 'three';
import { smokeGroup } from './scene.js';

let smokestacks = [];
let smokePuffs = [];
const puffGeo = new THREE.BoxGeometry(1, 1, 1);

export function clearSmokestacks() {
  smokestacks = [];
}

export function registerSmokestack(anchor) {
  smokestacks.push(anchor);
}

// Same reasoning as MAX_DUST_PUFFS below -- this pool used to only ever
// serve a handful of Industrial Site smokestacks, but Military Zone's
// vehicle wrecks now register anchors here too (see addVehicleWreck in
// world.js), and nothing previously stopped the anchor count from
// growing the live puff count without bound.
const MAX_SMOKE_PUFFS = 60;

function spawnSmoke(dt) {
  for (const s of smokestacks) {
    s._t = (s._t ?? Math.random() * 0.6) + dt;
    if (s._t < 0.55) continue;
    s._t = 0;
    if (smokePuffs.length >= MAX_SMOKE_PUFFS) continue;
    const mat = new THREE.MeshLambertMaterial({ color: 0xcfd2d4, transparent: true, opacity: 0.55 });
    const mesh = new THREE.Mesh(puffGeo, mat);
    mesh.position.set(s.x + (Math.random() - 0.5) * 1.4, s.y, s.z + (Math.random() - 0.5) * 1.4);
    mesh.rotation.y = Math.random() * Math.PI;
    mesh.scale.setScalar(1.6 + Math.random() * 0.6);
    smokeGroup.add(mesh);
    smokePuffs.push({
      mesh, age: 0, life: 7 + Math.random() * 2.5,
      drift: [(Math.random() - 0.5) * 2.2, (Math.random() - 0.5) * 2.2],
    });
  }
}

// Ground-hugging dust trail for fast-moving vehicles -- same pooled-puff
// approach as the smokestacks above, but tan-coloured, low and quick to
// dissipate instead of rising like smoke, and only spawns while `active`
// is true (set false to stop a stopped/removed vehicle from still
// puffing). Anchors are owned by the caller (world.js mutates x/y/z each
// frame to trail behind a moving unit) and unregistered via
// clearDustEmitters on every Military Zone rebuild.
let dustEmitters = [];
let dustPuffs = [];

export function clearDustEmitters() {
  dustEmitters = [];
}

export function registerDustEmitter(anchor) {
  dustEmitters.push(anchor);
}

// Hard ceiling on concurrent dust puffs, independent of how many vehicles
// are emitting -- each puff is a real non-instanced Mesh + its own cloned
// Material, so with dozens of tanks/trucks each spawning every ~0.16s
// this previously had no upper bound and steadied out around 350-400+
// extra draw calls, created/destroyed continuously. That's very plausibly
// what one user report described as everything looking "stuck" -- frame
// rate cratering enough to read as frozen rather than slow. Scale-safe
// regardless of topology size, unlike a per-vehicle rate limit alone.
const MAX_DUST_PUFFS = 48;

function spawnDust(dt) {
  for (const s of dustEmitters) {
    if (!s.active) continue;
    s._t = (s._t ?? Math.random() * 0.2) + dt;
    if (s._t < 0.4) continue;
    s._t = 0;
    if (dustPuffs.length >= MAX_DUST_PUFFS) continue;
    // Cloned per puff (not shared) -- each puff fades on its own opacity
    // timeline, a shared material would make every live puff flicker to
    // whichever one last wrote opacity.
    const mat = new THREE.MeshLambertMaterial({ color: 0xa8946a, transparent: true, opacity: 0.4 });
    const mesh = new THREE.Mesh(puffGeo, mat);
    mesh.position.set(s.x + (Math.random() - 0.5) * 2.5, s.y, s.z + (Math.random() - 0.5) * 2.5);
    mesh.rotation.y = Math.random() * Math.PI;
    mesh.scale.setScalar(1.0 + Math.random() * 0.7);
    smokeGroup.add(mesh);
    dustPuffs.push({
      mesh, age: 0, life: 0.9 + Math.random() * 0.6,
      drift: [(Math.random() - 0.5) * 3.4, (Math.random() - 0.5) * 3.4],
    });
  }
}

export function stepSmoke(dt) {
  spawnSmoke(dt);
  for (let i = smokePuffs.length - 1; i >= 0; i--) {
    const p = smokePuffs[i];
    p.age += dt;
    const t = p.age / p.life;
    if (t >= 1) {
      smokeGroup.remove(p.mesh);
      p.mesh.material.dispose();
      smokePuffs.splice(i, 1);
      continue;
    }
    p.mesh.position.y += dt * (3.2 + t * 3.0);
    p.mesh.position.x += p.drift[0] * dt;
    p.mesh.position.z += p.drift[1] * dt;
    p.mesh.scale.setScalar(1.6 + t * 6.0);
    p.mesh.material.opacity = 0.55 * (1 - t);
  }

  spawnDust(dt);
  for (let i = dustPuffs.length - 1; i >= 0; i--) {
    const p = dustPuffs[i];
    p.age += dt;
    const t = p.age / p.life;
    if (t >= 1) {
      smokeGroup.remove(p.mesh);
      p.mesh.material.dispose();
      dustPuffs.splice(i, 1);
      continue;
    }
    p.mesh.position.y += dt * 0.6;
    p.mesh.position.x += p.drift[0] * dt;
    p.mesh.position.z += p.drift[1] * dt;
    p.mesh.scale.setScalar((1.0 + t * 3.2) * (1 - t * 0.2));
    p.mesh.material.opacity = 0.4 * (1 - t);
  }
}
