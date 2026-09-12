// Entry point: wires the decomposed modules together -- loads textures,
// then drives the poll loop (fetch /api/state -> rebuild changed parts of
// the scene) and the render loop. Each concern lives in its own module
// under ./js/: core.js (shared utils), textures.js (icon loading),
// scene.js (renderer/camera/lighting/groups), world.js (buildings/roads/
// props), entities.js (nodes/links/packets/vehicles), smoke.js, hud.js.
import { POLL_MS } from './js/core.js';
import { loadAllTextures } from './js/textures.js';
import { camera, composer, controls, resize, applyEnvironment, updateSunSprite } from './js/scene.js';
import {
  rebuildProps, rebuildBuildings, rebuildRoads, rebuildCrossStreets, rebuildOverbridge,
  setActiveCrossStreets,
  industrialRoadLoops, rebuildIndustrialTraffic, stepIndustrialTraffic,
  rebuildMilitaryPatrol, stepMilitaryPatrol,
  applyLoadedTextures, siteRadius, militaryExtent,
} from './js/world.js';
import {
  updateNodes, updateLinks, updatePackets, updateVehicles, stepNodeAnimation,
  updateApBackbone, updateApRanges,
} from './js/entities.js';
import { stepSmoke } from './js/smoke.js';
import { updateHud, showStatus, hideStatus } from './js/hud.js';
import './js/interact.js'; // wires node drag-and-drop (also exports getSelectedIds, for camera-modes.js's follow mode)
import { getCameraMode, stepCameraModes } from './js/camera-modes.js';

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
    // Before rebuildProps/rebuildBuildings run, not after -- both call
    // clearOfRoads/scatterPos internally (tree/building placement), and
    // cars_uavs's cross streets are straight perpendicular segments, not
    // loops, so they were never being avoided at all (reported: "you put
    // trees and buildings on the road"). No-op (empty array) for every
    // other environment.
    setActiveCrossStreets(state.cross_streets ?? []);
    rebuildProps(state.environment, state.obstacles, roadLoops, nodeExtentRadius(state.nodes), state.variant);
    rebuildBuildings(state.obstacles, state.environment, state.variant, roadLoops);
    rebuildRoads(roadLoops);
    // cars_uavs mode only -- state.cross_streets/overbridge are empty/null
    // everywhere else (ui/web3d/snapshot.py's _cross_streets/_overbridge),
    // so these are no-ops there. rebuildOverbridge before updateNodes: it
    // updates world.js's own live bridge-config state that
    // bridgeDeckHeightAt reads, and updateNodes is what actually
    // positions car/scooter meshes using that lookup this same poll.
    rebuildCrossStreets(state.cross_streets ?? [], state.overbridge ?? null);
    rebuildOverbridge(state.overbridge ?? null);
    rebuildIndustrialTraffic(state.obstacles, state.environment, state.variant);
    rebuildMilitaryPatrol(state.obstacles, state.environment);
    updateNodes(state.nodes);
    updateLinks(state.nodes);
    updateApBackbone(state.nodes, state.ap_links);
    updateApRanges(state.nodes);
    updatePackets(state.packets, state.pulses, state.nodes);
    updateVehicles(state.vehicles);
    updateHud(state);

    // Only when still in the default orbit view -- a rebuild/env change
    // landing mid free-fly or mid chase-cam shouldn't yank the camera
    // back to the auto-framed orbit shot the user has already moved away
    // from (firstFrame itself only ever fires once per page load, but
    // this guard costs nothing and is the correct behaviour either way).
    if (firstFrame) {
      firstFrame = false;
      if (getCameraMode() === 'orbit') frameCameraOnNodes(state.nodes, state.obstacles, state.environment);
    }
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
function nodeExtentRadius(nodesData, originX = 0, originY = 0, pct = 0.8) {
  if (!nodesData.length) return 0;
  const dists = nodesData.map((n) => Math.hypot(n.pos[0] - originX, n.pos[1] - originY))
    .sort((a, b) => a - b);
  const pIdx = Math.min(dists.length - 1, Math.floor(dists.length * pct));
  return dists[pIdx];
}

function frameCameraOnNodes(nodesData, obstacles, environment) {
  if (!nodesData.length) return;
  // Multi-AP topologies (multi_ap / cars_uavs -- see sim11ah/topology.py's
  // MultiApBuilder) place AP0 at the world origin and extend the REST of
  // the topology forward from there in a line, unlike every other
  // topology, which scatters its STAs symmetrically AROUND one central
  // AP that already sits at the origin. Always targeting the fixed
  // origin left most of a multi-AP corridor -- and most of its cars/
  // scooters/UAVs -- sitting off to one side of the frustum by default,
  // invisible without the user manually panning: the camera had enough
  // DISTANCE to encompass the whole corridor (maxR below already
  // accounts for it), just not pointed at its middle. Detected by >1
  // AP-role node, which only single-AP topologies (mode "uav",
  // "aerial_relay", etc.) never have -- for those this is a no-op (the
  // scatter's own centroid already sits at/near the origin, since node 0
  // is always there), so this can't regress the origin-centred framing
  // every other environment's procedural terrain is itself built around.
  const apNodes = nodesData.filter((n) => n.role === 'AP');
  let targetX = 0, targetZ = 0;
  if (apNodes.length > 1) {
    let cxWorld = 0, cyWorld = 0;
    for (const n of nodesData) { cxWorld += n.pos[0]; cyWorld += n.pos[1]; }
    cxWorld /= nodesData.length; cyWorld /= nodesData.length;
    targetX = cxWorld; targetZ = -cyWorld;
  }
  let maxR;
  if (apNodes.length > 1) {
    // Multi-AP corridor (cars_uavs/multi_ap): fitting every AP's own
    // range ring (802.11ah's real range is 900m+) AND the AP grid's own
    // spread (its far ends can be ~1km from the centroid this view is
    // targeted on) at once used to force the camera to an ~880m-high
    // satellite view for a typical cars_uavs scene -- at that distance
    // NOTHING at ground level (buildings, vehicles, road/manhole detail,
    // all of which this scene now actually has) is recognisable as
    // anything but a speck. Framed on the node scatter alone instead, at
    // a tighter percentile than the single-AP branch below (the long
    // corridor's own far ends don't need to fit in the very first frame
    // either) -- a detailed establishing shot over the busiest part of
    // the corridor, not an orthographic map of the whole thing. Scroll/
    // pan/free-fly reach the rest; nothing is actually hidden, just not
    // forced into frame 1.
    maxR = Math.max(100, nodeExtentRadius(nodesData, targetX, -targetZ, 0.5));
  } else {
    maxR = Math.max(100, nodeExtentRadius(nodesData, targetX, -targetZ));
    // Also make sure the single AP's own coverage-range ring
    // (updateApRanges, a fixed circle around its position, independent
    // of where vehicles currently happen to be) fits inside the initial
    // view -- nodeExtentRadius alone only guarantees the live vehicle
    // SCATTER is framed, and a range ring can extend past that scatter's
    // own 80th-percentile radius (especially early on, or wherever
    // traffic happens to be light), which would otherwise leave part of
    // the ring outside the frustum by default. Only reachable when
    // apNodes.length <= 1 -- see the multi-AP branch above for why this
    // exact approach doesn't scale to several APs spread across a
    // corridor.
    for (const n of apNodes) {
      if (typeof n.range_m !== 'number' || n.range_m <= 0) continue;
      const apDist = Math.hypot(n.pos[0] - targetX, n.pos[1] + targetZ);
      maxR = Math.max(maxR, apDist + n.range_m);
    }
  }
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
  camera.position.set(targetX + dist * 0.5, dist * 0.4, targetZ + dist * 0.62);
  controls.target.set(targetX, 15, targetZ);
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
  stepCameraModes(dt);
  updateSunSprite();
  controls.update();
  composer.render();
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
