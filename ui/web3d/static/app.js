// Entry point: wires the decomposed modules together -- loads textures,
// then drives the poll loop (fetch /api/state -> rebuild changed parts of
// the scene) and the render loop. Each concern lives in its own module
// under ./js/: core.js (shared utils), textures.js (icon loading),
// scene.js (renderer/camera/lighting/groups), world.js (buildings/roads/
// props), entities.js (nodes/links/packets/vehicles), smoke.js, hud.js.
import { POLL_MS } from './js/core.js';
import { loadAllTextures } from './js/textures.js';
import { scene, camera, renderer, controls, resize, applyEnvironment } from './js/scene.js';
import {
  rebuildProps, rebuildBuildings, rebuildRoads,
  industrialRoadLoops, rebuildIndustrialTraffic, stepIndustrialTraffic,
  rebuildMilitaryPatrol, stepMilitaryPatrol,
  applyLoadedTextures, siteRadius, militaryExtent,
} from './js/world.js';
import { updateNodes, updateLinks, updatePackets, updateVehicles, stepNodeAnimation } from './js/entities.js';
import { stepSmoke } from './js/smoke.js';
import { updateHud, showStatus, hideStatus } from './js/hud.js';
import './js/interact.js'; // wires node drag-and-drop; no exports, side-effect only

let firstFrame = true;
let latestSimTime = 0;

async function poll() {
  try {
    const res = await fetch('/api/state', { cache: 'no-store' });
    const state = await res.json();
    if (!state.ready) {
      showStatus(state.error ? ('3D view error: ' + state.error) : 'Waiting for simulation…');
      return;
    }
    hideStatus();
    latestSimTime = state.time;

    applyEnvironment(state.environment);
    // Smart City's road/vehicle loops come from Python (snapshot.py);
    // Industrial Site has no server-side road data, so the same loop
    // shape is computed client-side instead -- either way rebuildRoads
    // draws exactly what rebuildIndustrialTraffic/updateVehicles drive
    // on, and props/filler buildings keep off the same pavement.
    const roadLoops = state.environment === 'Smart City' ? (state.road_loops ?? [])
      : state.environment === 'Industrial Site' ? industrialRoadLoops(state.obstacles, state.variant)
      : [];
    rebuildProps(state.environment, state.obstacles, roadLoops, nodeExtentRadius(state.nodes), state.variant);
    rebuildBuildings(state.obstacles, state.environment, state.variant, roadLoops);
    rebuildRoads(roadLoops);
    rebuildIndustrialTraffic(state.obstacles, state.environment, state.variant);
    rebuildMilitaryPatrol(state.obstacles, state.environment);
    updateNodes(state.nodes);
    updateLinks(state.nodes);
    updatePackets(state.packets, state.pulses, state.nodes);
    updateVehicles(state.vehicles);
    updateHud(state);

    if (firstFrame) { firstFrame = false; frameCameraOnNodes(state.nodes, state.obstacles, state.environment); }
  } catch (err) {
    showStatus('3D view: connection lost, retrying…');
  } finally {
    setTimeout(poll, POLL_MS);
  }
}

// 802.11ah's real long range means a handful of outlier STAs can sit
// 5-10x farther out than the rest of the network -- sizing off the single
// farthest one shrinks whatever's using this (camera framing, Paddy
// Field's prop scatter) to a speck relative to a couple of extreme
// outliers. The 80th-percentile distance keeps the dense part of the
// network large instead; a couple of outliers just sit past the built
// scene, same as any other map view.
function nodeExtentRadius(nodesData) {
  if (!nodesData.length) return 0;
  const dists = nodesData.map((n) => Math.hypot(n.pos[0], n.pos[1])).sort((a, b) => a - b);
  const pIdx = Math.min(dists.length - 1, Math.floor(dists.length * 0.8));
  return dists[pIdx];
}

function frameCameraOnNodes(nodesData, obstacles, environment) {
  if (!nodesData.length) return;
  let maxR = Math.max(100, nodeExtentRadius(nodesData));
  // Also frame around the environment's built structures, not just node
  // positions -- a base/campus/site can extend well past a tight node
  // cluster, and the default view should show the built scene without
  // the user needing to manually zoom out to find it. Military Zone uses
  // militaryExtent(), the exact same (uncapped) extent
  // rebuildMilitaryPatrol builds the perimeter/patrol loops from -- using
  // siteRadius() here instead (capped at 700) previously let the
  // patrolling tanks/soldiers sit just outside the camera's default view
  // on any topology whose real extent ran past that cap.
  if (environment === 'Military Zone') {
    const { hw, hd } = militaryExtent(obstacles || []);
    maxR = Math.max(maxR, Math.hypot(hw, hd) * 0.75);
  } else {
    const siteR = siteRadius(obstacles || []);
    maxR = Math.max(maxR, siteR * 1.05);
  }
  const dist = Math.max(220, maxR * 1.15);
  camera.position.set(dist * 0.5, dist * 0.4, dist * 0.62);
  controls.target.set(0, 15, 0);
  controls.update();
}

let lastT = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;
  stepNodeAnimation(dt);
  stepSmoke(dt);
  latestSimTime += dt;
  stepIndustrialTraffic(latestSimTime);
  stepMilitaryPatrol(latestSimTime, dt);
  controls.update();
  renderer.render(scene, camera);
}

async function main() {
  resize();
  await loadAllTextures();
  applyLoadedTextures();
  applyEnvironment('Open Area');
  animate();
  poll();
}
main();
