// Live simulated entities: AP/relay/STA/drone node models, association
// links, in-flight packets/beacon pulses, and Smart City vehicles. All
// driven by the JSON polled from /api/state (see main.js's poll loop) --
// nothing here reads sim state directly.
import * as THREE from 'three';
import { ASSOC, PKT_COLORS, STATUS_COLOR, POLL_MS, toScene } from './core.js';
import { nodesGroup, linksGroup, packetsGroup, vehiclesGroup } from './scene.js';
import { roofHeightAt } from './world.js';

// ---- per-node info label: distance to the AP (metres) always, plus the
// "why hasn't this joined" code (E417/E240/E102 -- see _diagnose_unjoined
// in topology_canvas.py, the single source of truth this scene and the
// real-map twin both read via /api/state) when the node isn't associated.
// One sprite per node, its canvas/texture repainted only when the text
// actually changes (not every frame) -- distance changes continuously as
// nodes move, so this can't be a small shared cache keyed by a handful of
// fixed strings the way a pure error-code label could be. ------------------
const ERROR_SPRITE_COLOR = { E417: '#f87171', E240: '#f87171', E102: '#fbbf24' };

function formatDistM(d) {
  if (typeof d !== 'number') return '';
  return d >= 1000 ? `${(d / 1000).toFixed(2)}km` : `${Math.round(d)}m`;
}

function formatRssi(d) {
  return typeof d === 'number' ? `${Math.round(d)}dBm` : '';
}

function nodeInfoText(n) {
  const parts = [];
  const dist = formatDistM(n.dist_ap_m);
  if (dist) parts.push(dist);
  const rssi = formatRssi(n.rssi_dbm);
  if (rssi) parts.push(rssi);
  let text = parts.join(' · ');
  if (n.error_code) text = text ? `${text} · ${n.error_code}` : n.error_code;
  return text;
}

function nodeInfoColor(n) {
  if (n.error_code) return ERROR_SPRITE_COLOR[n.error_code] || '#f87171';
  return '#9fb0c9';
}

function makeInfoSprite() {
  const canvas = document.createElement('canvas');
  canvas.width = 176; canvas.height = 40;
  const texture = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(15, 3.4, 1);
  sprite.position.set(0, 12, 0);
  sprite.visible = false;
  return { sprite, canvas, texture, lastText: null, lastColor: null };
}

function updateInfoSprite(info, text, color) {
  if (!text) {
    if (info.sprite.visible) info.sprite.visible = false;
    return;
  }
  info.sprite.visible = true;
  if (text === info.lastText && color === info.lastColor) return;
  info.lastText = text;
  info.lastColor = color;
  const c = info.canvas.getContext('2d');
  c.clearRect(0, 0, info.canvas.width, info.canvas.height);
  c.fillStyle = 'rgba(15, 23, 42, 0.80)';
  c.beginPath();
  c.roundRect(2, 4, info.canvas.width - 4, info.canvas.height - 8, 8);
  c.fill();
  c.strokeStyle = color;
  c.lineWidth = 2;
  c.stroke();
  c.fillStyle = color;
  c.font = '700 18px -apple-system, Arial, sans-serif';
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  c.fillText(text, info.canvas.width / 2, info.canvas.height / 2);
  info.texture.needsUpdate = true;
}

// ---- nodes: blocky equipment models with a small status LED accent
// (the enclosure colour stays fixed like real hardware; only the LED
// reflects live association state) --------------------------------------
const nodeMeshes = new Map();
// Clearly distinct, saturated role colours -- not just the status LED
// (which switches to green/amber/red once a real assoc_state exists, so
// a relay's LED stops being purple the moment it associates -- see
// statusHex). These body colours are the ONLY role marker that stays
// constant regardless of live status, same "outline = role, fill = MAC
// state" split the 2D topology canvas already uses (AP blue, relay
// purple outline there too -- see topology_canvas.py's _BLUE/_PURPLE).
// Previously both were near-black/near-identical and easy to lose
// against the terrain or each other.
const apBodyMat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.5, color: 0x3b6fd6 });
const relayBodyMat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.5, color: 0x8a4fd6 });
// Multi-select ring -- one shared geometry/material (per-instance state
// is just visibility), flat on the ground under a node, same amber the
// 2D canvas's own selection ring uses. Hidden unless interact.js's
// ctrl/shift-click selection includes this node.
const selRingGeo = new THREE.RingGeometry(6, 7.4, 24);
const selRingMat = new THREE.MeshBasicMaterial({
  color: 0xf59e0b, side: THREE.DoubleSide, transparent: true, opacity: 0.9, depthWrite: false,
});
const staBodyMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.4, color: 0x454c57 });
// Aerial-relay drones (role 'RELAY', is_drone) reuse relayBodyMat outright --
// same role, same colour, just airborne, exactly like the AP/relay/STA rule
// above. UAV end nodes (role 'STA', is_uav) get their own teal, matching the
// 2D topology canvas's plain-STA outline colour (_TEAL in topology_canvas.py)
// since it's the same role, just a different embodiment (aircraft vs box).
// Previously both shared one droneBodyMat and were told apart only by a tiny
// LED cube coloured purple/red in buildDrone() below -- which statusHex()
// immediately overwrites with the live association-state colour on the very
// first updateNodes() call, so that LED distinction never actually rendered
// and relay drones were visually identical to UAV end nodes in the scene.
const uavBodyMat = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.45, color: 0x14b8a6 });
const armMat = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.5, color: 0x14171c });
const rotorMat = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.5, color: 0x0e1013, transparent: true, opacity: 0.5 });

function statusHex(n) {
  // Everything except the AP has a real, meaningful assoc_state -- STAs,
  // UAV end-nodes (role 'STA'), and RELAY nodes (including aerial-relay
  // drones, whose own uplink association to the AP is exactly as real and
  // worth showing as any STA's). The AP itself has no peer to associate
  // with, so it's the one node excluded.
  if (n.role === 'AP') return null;
  if (n.assoc_state === ASSOC.ASSOCIATED) return STATUS_COLOR.ASSOCIATED;
  if (n.assoc_state === ASSOC.UNASSOCIATED) return STATUS_COLOR.UNASSOCIATED;
  return STATUS_COLOR.PENDING;
}

const plinthMat = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0.1, color: 0x2a2f38 });
const apPanelMat = new THREE.MeshStandardMaterial({ roughness: 0.3, metalness: 0.4, color: 0x93c5fd });
const relayPanelMat = new THREE.MeshStandardMaterial({ roughness: 0.3, metalness: 0.4, color: 0xd8b4fe });

function buildApOrRelay(isAp) {
  const g = new THREE.Group();
  // The AP is the one node that never moves and never sits on a rooftop
  // (it's the fixed reference point every road/building layout is built
  // around) -- it needs to read as a clear comms-tower landmark even next
  // to 140-260-unit downtown skyscrapers, so it's built noticeably taller
  // than a relay mast and gets a second lattice brace partway up.
  const mastH = isAp ? 58 : 24;
  const bodyMat = isAp ? apBodyMat : relayBodyMat;

  // Base plinth: a short, wide concrete-style foundation the mast rises
  // from -- previously the mast just touched the terrain at a single
  // point, which read as a thin pole rather than an anchored structure.
  // Every other mesh below shifts up by its height (baseH) to sit on top
  // of it instead of sinking into it.
  const baseH = isAp ? 1.6 : 1.2;
  const plinth = new THREE.Mesh(
    new THREE.CylinderGeometry(isAp ? 4.2 : 3.0, isAp ? 4.8 : 3.4, baseH, 8), plinthMat);
  plinth.position.y = baseH / 2;
  plinth.castShadow = true;
  plinth.receiveShadow = true;
  g.add(plinth);

  const mast = new THREE.Mesh(new THREE.BoxGeometry(2.2, mastH, 2.2), bodyMat);
  mast.position.y = mastH / 2 + baseH;
  mast.castShadow = true;
  g.add(mast);
  const unit = new THREE.Mesh(new THREE.BoxGeometry(6, 8, 4), bodyMat);
  unit.position.y = mastH * 0.85 + baseH;
  unit.castShadow = true;
  g.add(unit);
  const armCount = isAp ? 3 : 2;
  const braceHeights = isAp ? [0.55, 0.95] : [0.95];
  // Sector-antenna panel at the outer tip of each arm -- previously the
  // arm was just a bare crossbar with no end detail, unlike the 2D
  // topology canvas's own tower glyph (_draw_tower_icon in
  // topology_canvas.py), which already draws a panel at each arm tip;
  // this brings the two views' visual language back in sync.
  const panelMat = isAp ? apPanelMat : relayPanelMat;
  for (const hFrac of braceHeights) {
    for (let i = 0; i < armCount; i++) {
      const ang = (i / armCount) * Math.PI * 2 + (hFrac < 0.9 ? Math.PI / armCount : 0);
      const armLen = isAp ? 12 : 9;
      const armY = mastH * hFrac + baseH;
      const arm = new THREE.Mesh(new THREE.BoxGeometry(armLen, 1.2, 1.2), armMat);
      arm.position.y = armY;
      arm.rotation.y = ang;
      g.add(arm);

      // Same rotation.y convention as the arm above: a local (+X, 0, 0)
      // tip point transforms to world-relative (cos(ang), 0, -sin(ang))
      // * armLen/2 once rotated by ang around Y.
      const panel = new THREE.Mesh(new THREE.BoxGeometry(1.0, 3.0, 2.0), panelMat);
      panel.position.set(Math.cos(ang) * armLen * 0.5, armY, -Math.sin(ang) * armLen * 0.5);
      panel.rotation.y = ang;
      panel.castShadow = true;
      g.add(panel);
    }
  }
  const ledMat = new THREE.MeshBasicMaterial({ color: isAp ? 0x60a5fa : 0xa78bfa });
  const led = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), ledMat);
  led.position.y = mastH + baseH + 3;
  g.add(led);
  return { group: g, ledMat, rotors: null };
}

const staCapMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.4, color: 0x5b6472 });

function buildStation() {
  const g = new THREE.Group();
  const base = new THREE.Mesh(new THREE.BoxGeometry(9, 5, 9), staBodyMat);
  base.position.y = 2.5;
  base.castShadow = true;
  g.add(base);
  // Narrower top plate -- one extra shape gives the box a "stacked
  // equipment unit" silhouette instead of one flat slab, cheap enough to
  // afford even with a large STA population (unlike the AP/relay
  // plinth+panels treatment, which stays reserved for the much smaller
  // number of tower-role nodes).
  const cap = new THREE.Mesh(new THREE.BoxGeometry(6.6, 1.2, 6.6), staCapMat);
  cap.position.y = 5.6;
  cap.castShadow = true;
  g.add(cap);
  const antenna = new THREE.Mesh(new THREE.BoxGeometry(0.8, 8, 0.8), armMat);
  antenna.position.set(2.6, 10.2, 0);
  g.add(antenna);
  const ledMat = new THREE.MeshBasicMaterial({ color: 0x34d399 });
  const led = new THREE.Mesh(new THREE.BoxGeometry(1.8, 1.8, 1.8), ledMat);
  led.position.set(-2.6, 6.5, 0);
  g.add(led);
  return { group: g, ledMat, rotors: null };
}

function buildDrone(isRelay) {
  const g = new THREE.Group();
  const hull = new THREE.Mesh(new THREE.BoxGeometry(6, 2.4, 6), isRelay ? relayBodyMat : uavBodyMat);
  hull.castShadow = true;
  g.add(hull);
  const rotors = [];
  const armLen = 8.0;
  for (const ang of [Math.PI / 4, (3 * Math.PI) / 4, (5 * Math.PI) / 4, (7 * Math.PI) / 4]) {
    const sub = new THREE.Group();
    sub.rotation.y = ang;
    g.add(sub);
    const arm = new THREE.Mesh(new THREE.BoxGeometry(armLen, 0.8, 0.8), armMat);
    arm.position.set(armLen / 2, 0.1, 0);
    sub.add(arm);
    const rotor = new THREE.Mesh(new THREE.BoxGeometry(6.4, 0.3, 1.4), rotorMat);
    rotor.position.set(armLen, 1.2, 0);
    sub.add(rotor);
    rotors.push(rotor);
  }
  // statusHex() overwrites this on the very first updateNodes() call for
  // every non-AP node (see ensureNodeMesh), same as buildStation()'s LED --
  // this initial colour is never actually seen, so it just matches that
  // same ASSOCIATED-green default rather than trying (and failing) to
  // double as a role marker; the hull colour above is the real one now.
  const ledMat = new THREE.MeshBasicMaterial({ color: 0x34d399 });
  const led = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1.6, 1.6), ledMat);
  led.position.set(0, 1.8, 3.2);
  g.add(led);
  // Uniform group scale (same approach as buildCarBody/buildScooter) --
  // 3.0 = a 200% increase (per explicit request) so UAVs/relay drones
  // are as easy to spot as the now much bigger cars/scooters, with
  // every part (hull/arms/rotors/LED) staying proportioned exactly as
  // authored above.
  g.scale.setScalar(3.0);
  return { group: g, ledMat, rotors };
}

function ensureNodeMesh(n) {
  let entry = nodeMeshes.get(n.id);
  if (entry) return entry;
  let built;
  if (n.is_drone || n.is_uav) built = buildDrone(n.role === 'RELAY');
  else if (n.is_car) built = buildCar();
  else if (n.is_scooter) built = buildScooter();
  else if (n.role === 'AP') built = buildApOrRelay(true);
  else if (n.role === 'RELAY') built = buildApOrRelay(false);
  else built = buildStation();
  // The AP is a fixed reference point everything else (roads, filler
  // buildings, the plaza) is laid out around, so -- same rule as the 2D
  // topology canvas -- it's the one node interact.js won't let you drag.
  built.group.userData.nodeId = n.id;
  built.group.userData.draggable = n.role !== 'AP';
  built.group.userData.airborne = !!(n.is_drone || n.is_uav);
  nodesGroup.add(built.group);
  const info = n.role === 'AP' ? null : makeInfoSprite();
  if (info) built.group.add(info.sprite);
  const selRing = new THREE.Mesh(selRingGeo, selRingMat);
  selRing.rotation.x = -Math.PI / 2;
  selRing.position.y = 0.4;
  selRing.visible = false;
  built.group.add(selRing);
  const entryObj = {
    group: built.group, ledMat: built.ledMat, rotors: built.rotors,
    fromPos: null, toPos: null, t: 1, dragging: false, info, selRing,
  };
  nodeMeshes.set(n.id, entryObj);
  return entryObj;
}

// Ground-based nodes (not airborne drones/UAVs, not the fixed AP tower)
// stand on whatever rooftop their true (x, y) position happens to fall
// on -- interact.js's live drag preview uses the same lookup so a node
// visibly rises onto a roof as you drag it there.
function altitudeFor(n, sx, sz) {
  // Real per-node cruise altitude computed server-side (altitude_m_for_node
  // in topology_canvas.py -- relays fly higher than end-node UAVs for
  // line-of-sight, varies node to node, and is the same value the
  // real-map twin reads) -- not a single flat height for every drone/UAV.
  if (n.is_drone || n.is_uav) return typeof n.altitude_m === 'number' ? n.altitude_m : 42;
  if (n.role === 'AP') return 0;
  // A car/scooter is road-bound, not a building occupant -- unlike a plain
  // STA, it should never rise onto a rooftop its (x, y) happens to cross.
  if (n.is_car || n.is_scooter) return 0;
  const roof = roofHeightAt(sx, sz);
  return roof === null ? 0 : roof;
}

export function updateNodes(nodesData) {
  const seen = new Set();
  for (const n of nodesData) {
    seen.add(n.id);
    const entry = ensureNodeMesh(n);
    const st = statusHex(n);
    if (st !== null) entry.ledMat.color.setHex(st);
    if (entry.info) {
      updateInfoSprite(entry.info, nodeInfoText(n), nodeInfoColor(n));
    }
    // A car/scooter's heading changes in fixed 90-degree steps whenever
    // it turns onto the next road (see sim11ah/mobility.py's
    // grid_road_step) -- set directly each poll, not lerped like
    // position below, so a turn reads as a sharp corner rather than a
    // smooth arc (matching how the vehicle's own position actually moves
    // -- straight, then a corner, never a curve). Same world-heading ->
    // rotation.y convention updateVehicles already uses for the
    // decorative Smart City traffic (local model forward is +X; three.js's
    // rotation.y maps that to exactly the world heading, no sign flip
    // needed).
    if ((n.is_car || n.is_scooter) && typeof n.heading === 'number') {
      entry.group.rotation.y = n.heading;
    }

    // While a node is being dragged, interact.js owns its position
    // directly -- skip the server-driven lerp target so the drag doesn't
    // fight the next poll's (now-stale) position.
    if (entry.dragging) continue;

    const target = toScene(n.pos[0], n.pos[1]);
    target.y = altitudeFor(n, target.x, target.z);
    if (!entry.toPos) {
      entry.group.position.copy(target);
      entry.fromPos = target.clone();
      entry.toPos = target.clone();
      entry.t = 1;
    } else if (!target.equals(entry.toPos)) {
      // Real vehicle motion (grid_road_step/uav_bounce_step, both fully
      // continuous -- turn, never teleport) never covers anywhere near
      // this much ground between polls, so a jump this large is only
      // ever a genuine repositioning (a topology rebuild reusing a node
      // id, say). Snap straight to it instead of lerping, or the mesh
      // would visibly fly across the map over the next
      // stepNodeAnimation cycle.
      if (entry.group.position.distanceTo(target) > 60) {
        entry.group.position.copy(target);
        entry.fromPos = target.clone();
        entry.toPos = target.clone();
        entry.t = 1;
      } else {
        entry.fromPos = entry.group.position.clone();
        entry.toPos = target;
        entry.t = 0;
      }
    }
  }
  for (const [id, entry] of [...nodeMeshes.entries()]) {
    if (!seen.has(id)) { nodesGroup.remove(entry.group); nodeMeshes.delete(id); }
  }
}

// ---- drag-and-drop support (see interact.js for the pointer-event side) ----
export function beginDrag(id) {
  const entry = nodeMeshes.get(id);
  if (entry) entry.dragging = true;
}

// A drone/UAV is an aircraft, not a ground vehicle -- repositioning one by
// drag changes where it flies TO, not where it lands, so interact.js uses
// this to skip the ground/rooftop height lookup it applies to everything
// else and keeps the node at its own current altitude through the drag.
export function isAirborne(id) {
  const entry = nodeMeshes.get(id);
  return !!(entry && entry.group.userData.airborne);
}

export function currentAltitude(id) {
  const entry = nodeMeshes.get(id);
  return entry ? entry.group.position.y : 0;
}

// Full current position, not just altitude -- multi-node drag (see
// interact.js) needs each selected node's own starting (x, y, z) to
// compute start+delta per node, and each node's final position again on
// release to POST /api/move_node for it.
export function getNodePosition(id) {
  const entry = nodeMeshes.get(id);
  return entry ? entry.group.position.clone() : null;
}

// Multi-select highlight ring -- see ensureNodeMesh, which adds one
// (hidden by default) to every node's group. interact.js toggles this
// on/off as the ctrl/shift-click selection set changes, same "outline
// shows selection" convention the 2D topology canvas's amber ring uses.
export function setNodeSelected(id, selected) {
  const entry = nodeMeshes.get(id);
  if (entry && entry.selRing) entry.selRing.visible = selected;
}

export function setDragPosition(id, x, y, z) {
  const entry = nodeMeshes.get(id);
  if (!entry) return;
  entry.group.position.set(x, y, z);
  entry.fromPos = entry.group.position.clone();
  entry.toPos = entry.group.position.clone();
  entry.t = 1;
}

export function endDrag(id) {
  const entry = nodeMeshes.get(id);
  if (entry) entry.dragging = false;
}

export function stepNodeAnimation(dt) {
  const step = dt / (POLL_MS / 1000);
  for (const entry of nodeMeshes.values()) {
    if (entry.t < 1) {
      entry.t = Math.min(1, entry.t + step);
      entry.group.position.lerpVectors(entry.fromPos, entry.toPos, entry.t);
    }
    if (entry.rotors) for (const r of entry.rotors) r.rotation.y += dt * 50;
  }
}

// ---- links (live association, not the static build-time assignment) --------
// Antenna-tip height per role, matching the actual masts built in
// buildApOrRelay/buildStation (58/24/9m -- see mastH there). A flat +2m
// for every endpoint put ground-based links only ~2m above the terrain;
// at typical camera distances that's inside the depth buffer's precision
// floor (worse the farther the far-plane reaches, and this scene's is
// 8000), so the line would z-fight with -- and visually appear to get cut
// by -- any ground-level geometry it passes near or over: ground camo
// patches, scorch marks, terrain props. Airborne nodes already carry a
// real altitude_m well clear of the ground and don't need a mast added on
// top of it.
function linkEndpointY(n, sx, sz) {
  const base = altitudeFor(n, sx, sz);
  if (n.is_drone || n.is_uav) return base + 2;
  // AP/RELAY: mastH + baseH + 3 (the LED height buildApOrRelay actually
  // places -- 58+1.6+3 / 24+1.2+3, updated to match when the base plinth
  // was added there; previously 61/27, which no longer quite reached the
  // antenna once the plinth raised everything on top of it by baseH).
  const mastTop = n.role === 'AP' ? 62.6 : n.role === 'RELAY' ? 28.2 : 9;
  return base + mastTop;
}

// Matches ui/topology_canvas.py's own _AP_LINK_COLORS exactly, so a
// given AP reads as the same colour in both views.
const AP_LINK_COLORS = [0x3b82f6, 0xf97316, 0xa855f7, 0x06b6d4, 0xec4899, 0x84cc16, 0xf43f5e, 0x14b8a6];

let linkLines = [];
export function updateLinks(nodesData) {
  for (const l of linkLines) { linksGroup.remove(l); l.geometry.dispose(); l.material.dispose(); }
  linkLines = [];
  const byId = new Map(nodesData.map(n => [n.id, n]));
  // Sorted so a given AP id's colour rank stays stable regardless of
  // nodesData's own (poll-to-poll, not guaranteed stable) ordering.
  const apIds = nodesData.filter(n => n.role === 'AP').map(n => n.id).sort((a, b) => a - b);
  for (const n of nodesData) {
    // Drawn from each node's actual live association peer (assoc_peer is
    // only ever set once ASSOC_RESP genuinely succeeds -- see
    // sim11ah/mac/association.py's on_assoc_resp_received), not assumed --
    // a relay that hasn't associated with the AP yet can't actually relay
    // anything, so it shouldn't be drawn as already connected to it.
    let peerId = null, color = 0x475569, opacity = 0.8;
    if ((n.role === 'RELAY' || n.role === 'STA') && n.assoc_peer !== null) {
      peerId = n.assoc_peer;
      if (n.role === 'RELAY') { color = 0x8b5cf6; opacity = 0.6; }
      else if (n.assoc_state === ASSOC.ASSOCIATED) {
        // Colour-coded per AP under multi-AP (matches the 2D canvas's
        // own _AP_LINK_COLORS exactly) when the peer actually is one --
        // makes "this vehicle is associated with exactly one AP" a
        // literal, unmistakable fact on screen (its link is always one
        // solid colour, never two), and the colour visibly changes at
        // the instant of handover. Falls back to the original flat blue
        // for a STA associated with a RELAY instead (relay topologies
        // aren't multi-AP, nothing to colour-code there). Not green --
        // green is reserved for the per-node status LED (see
        // statusHex()/buildStation()'s ledMat), so an ASSOCIATED link
        // doubling it up made a fully-connected scene read as solid
        // green.
        const apIdx = apIds.indexOf(peerId);
        color = apIdx >= 0 ? AP_LINK_COLORS[apIdx % AP_LINK_COLORS.length] : 0x3b82f6;
      } else {
        color = 0xfbbf24;
      }
    }
    if (peerId === null || !byId.has(peerId)) continue;
    const peer = byId.get(peerId);
    // sx/sz passed in scene space (post toScene) so altitudeFor's rooftop
    // lookup (roofHeightAt) actually has real coordinates to check --
    // omitted before, so it silently always fell back to ground level
    // even for a node genuinely standing on a building's roof.
    const a = toScene(n.pos[0], n.pos[1]); a.y = linkEndpointY(n, a.x, a.z);
    const b = toScene(peer.pos[0], peer.pos[1]); b.y = linkEndpointY(peer, b.x, b.z);
    const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
    const line = new THREE.Line(geo, mat);
    linksGroup.add(line);
    linkLines.push(line);
  }
}

// ---- AP<->AP backbone -- a sparse (grid-adjacent, not all-pairs) render of
// the real backhaul every AP is actually linked to every other one for at
// the PHY/broadcast level (sim11ah/topology.py's MultiApBuilder.build);
// apLinks itself is computed server-side (ui/web3d/snapshot.py's
// _ap_backbone_pairs, mirroring ui/topology_canvas.py's own copy exactly)
// so both views draw the identical sparse backbone. APs are static (never
// move mid-run), so this only actually changes when the AP count/layout
// does, but it's cheap enough (a handful of lines) to just rebuild every
// poll rather than tracking that separately. --------------------------------
let apBackboneLines = [];
export function updateApBackbone(nodesData, apLinks) {
  for (const l of apBackboneLines) { linksGroup.remove(l); l.geometry.dispose(); l.material.dispose(); }
  apBackboneLines = [];
  if (!apLinks || !apLinks.length) return;
  const byId = new Map(nodesData.map(n => [n.id, n]));
  for (const pair of apLinks) {
    const a = byId.get(pair[0]), b = byId.get(pair[1]);
    if (!a || !b) continue;
    const pa = toScene(a.pos[0], a.pos[1]); pa.y = linkEndpointY(a, pa.x, pa.z);
    const pb = toScene(b.pos[0], b.pos[1]); pb.y = linkEndpointY(b, pb.x, pb.z);
    const geo = new THREE.BufferGeometry().setFromPoints([pa, pb]);
    const mat = new THREE.LineDashedMaterial({ color: 0x64748b, dashSize: 8, gapSize: 6, transparent: true, opacity: 0.75 });
    const line = new THREE.Line(geo, mat);
    line.computeLineDistances();
    linksGroup.add(line);
    apBackboneLines.push(line);
  }
}

// ---- AP range rings -- the real PHY range_m_for_node() radius already
// carried on every node as n.range_m (see ui/web3d/snapshot.py's
// _node_dict), not a decorative accent -- lets you see at a glance whether
// a vehicle's actual distance to its AP is inside or outside that AP's real
// range, for every AP at once, matching the 2D canvas's own
// _draw_ap_ranges_overlay exactly (same AP_LINK_COLORS palette, so a
// vehicle's link colour and the ring it should sit inside of are visually
// tied together in both views). -------------------------------------------
let apRangeRings = [];
// Dark and uniform (not per-AP coloured like AP_LINK_COLORS) -- matches
// ui/topology_canvas.py's own _AP_RANGE_RING exactly, so the range
// boundary reads as a fixed technical reference line in both views,
// distinct from and legible against the coloured, moving vehicle links.
const AP_RANGE_RING_COLOR = 0x1e293b;

export function updateApRanges(nodesData) {
  for (const r of apRangeRings) { linksGroup.remove(r); r.geometry.dispose(); r.material.dispose(); }
  apRangeRings = [];
  const apNodes = nodesData.filter(n => n.role === 'AP').sort((a, b) => a.id - b.id);
  const segs = 96;
  apNodes.forEach((n) => {
    if (typeof n.range_m !== 'number' || n.range_m <= 0) return;
    const center = toScene(n.pos[0], n.pos[1]);
    const pts = [];
    for (let i = 0; i <= segs; i++) {
      const ang = (i / segs) * Math.PI * 2;
      pts.push(new THREE.Vector3(center.x + Math.cos(ang) * n.range_m, 0.6, center.z + Math.sin(ang) * n.range_m));
    }
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineDashedMaterial({
      color: AP_RANGE_RING_COLOR, dashSize: 14, gapSize: 10,
      transparent: true, opacity: 0.75,
    });
    const line = new THREE.Line(geo, mat);
    line.computeLineDistances();
    linksGroup.add(line);
    apRangeRings.push(line);
  });
}

// ---- packets / broadcast pulses --------------------------------------------
let packetMeshes = [];
const packetGeo = new THREE.BoxGeometry(1, 1, 1);
export function updatePackets(packets, pulses, nodesData) {
  for (const m of packetMeshes) { packetsGroup.remove(m); m.geometry?.dispose?.(); m.material?.dispose?.(); }
  packetMeshes = [];
  const byId = new Map(nodesData.map(n => [n.id, n]));
  for (const pk of packets) {
    const tx = byId.get(pk.tx), rx = byId.get(pk.rx);
    if (!tx || !rx) continue;
    // Same linkEndpointY the link line itself uses (see updateLinks) --
    // a packet should visibly travel along the actual link line, not at
    // its own separately-computed (and previously ground-hugging, same
    // z-fighting risk) height.
    const a = toScene(tx.pos[0], tx.pos[1]); a.y = linkEndpointY(tx, a.x, a.z);
    const b = toScene(rx.pos[0], rx.pos[1]); b.y = linkEndpointY(rx, b.x, b.z);
    const pos = a.clone().lerp(b, pk.p);
    const mat = new THREE.MeshBasicMaterial({ color: PKT_COLORS[pk.kind] ?? 0x38bdf8 });
    const mesh = new THREE.Mesh(packetGeo, mat);
    mesh.scale.setScalar(3.6);
    mesh.position.copy(pos);
    packetsGroup.add(mesh);
    packetMeshes.push(mesh);
  }
  for (const pl of pulses) {
    const tx = byId.get(pl.tx);
    if (!tx) continue;
    const c = toScene(tx.pos[0], tx.pos[1]); c.y = 3;
    const r = 6 + pl.p * 60;
    const geo = new THREE.RingGeometry(Math.max(0.1, r - 1.4), r, 4);
    const mat = new THREE.MeshBasicMaterial({
      color: PKT_COLORS[pl.kind] ?? 0xfbbf24, transparent: true, opacity: Math.max(0, 1 - pl.p), side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.copy(c);
    packetsGroup.add(mesh);
    packetMeshes.push(mesh);
  }
}

// ---- Car body geometry, shared by two independent consumers:
// buildVehicle (pure Smart City scenery, driven by server-computed
// racetrack positions -- see ui/web3d/snapshot.py::_vehicles) and buildCar
// (real "cars_uavs" topology STAs -- see ensureNodeMesh) below. The shape
// itself doesn't care whether its position/color come from a decorative
// time formula or a real node's live (x, y) and role -- only the two
// callers differ. -----------------------------------------------------------
const wheelMat = new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0.15, color: 0x15171b });
function buildCarBody(bodyColor) {
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.5, color: bodyColor });
  const body = new THREE.Mesh(new THREE.BoxGeometry(4.6, 1.4, 2.1), bodyMat);
  body.position.y = 0.95;
  body.castShadow = true;
  group.add(body);
  const cabinMat = new THREE.MeshStandardMaterial({ roughness: 0.12, metalness: 0.35, color: 0x9fd0ea, transparent: true, opacity: 0.85 });
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(2.2, 1.0, 1.85), cabinMat);
  cabin.position.set(-0.3, 1.75, 0);
  group.add(cabin);
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const wheel = new THREE.Mesh(new THREE.BoxGeometry(0.9, 1.1, 1.1), wheelMat);
      wheel.position.set(sx * 1.55, 0.55, sz * 1.18);
      wheel.castShadow = true;
      group.add(wheel);
    }
  }
  const headlightMat = new THREE.MeshBasicMaterial({ color: 0xfff4d6 });
  const taillightMat = new THREE.MeshBasicMaterial({ color: 0xff5c5c });
  for (const sz of [-1, 1]) {
    const hl = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.3, 0.5), headlightMat);
    hl.position.set(2.3, 1.0, sz * 0.72);
    group.add(hl);
    const tl = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.3, 0.5), taillightMat);
    tl.position.set(-2.3, 1.0, sz * 0.72);
    group.add(tl);
  }
  // Uniform group scale, not larger box dimensions throughout -- a
  // single multiplier keeps every part (body/cabin/wheels/lights)
  // proportioned exactly as authored above while making the whole car
  // noticeably bigger/easier to spot at this scene's scale. 4.8 = 1.6 x
  // 3.0, a 200% increase (per explicit request) on top of the earlier
  // 1.6x bump -- matches the same on-top-of-current-size scaling
  // topology_canvas.py's own 2D car icon just got.
  group.scale.setScalar(4.8);
  return { group, bodyMat };
}

function buildVehicle() {
  return buildCarBody(0xffffff); // recolored per-instance below, in updateVehicles
}

// Real network car (role 'STA', is_car -- see ensureNodeMesh): fixed body
// color as its role marker (matches _REAL_CAR_COLORS[0] in the 2D
// topology canvas's own real-car overlay, for a loose visual echo between
// the two views), plus the same live-status LED every other real node
// gets (statusHex() sets it on every updateNodes() call, same as
// buildStation()'s).
function buildCar() {
  const built = buildCarBody(0xc0392b);
  const ledMat = new THREE.MeshBasicMaterial({ color: 0x34d399 });
  const led = new THREE.Mesh(new THREE.BoxGeometry(1.3, 1.3, 1.3), ledMat);
  led.position.set(0, 2.5, 0);
  built.group.add(led);
  return { group: built.group, ledMat, rotors: null };
}

// Real network scooter (role 'STA', is_scooter -- see ensureNodeMesh):
// same fixed-body-colour + status-LED convention as buildCar, but its own
// smaller, two-wheeled shape (a deck + handlebar stem) rather than
// buildCarBody's boxy sedan, so the two real vehicle kinds read apart by
// silhouette, not just by size. Local model forward is +X, same as
// buildCarBody, so it picks up the exact same world-heading rotation
// convention updateNodes already applies to cars.
function buildScooter() {
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.35, color: 0xf97316 });
  const deck = new THREE.Mesh(new THREE.BoxGeometry(2.3, 0.35, 0.9), bodyMat);
  deck.position.y = 0.55;
  deck.castShadow = true;
  group.add(deck);
  const stemMat = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.6, color: 0x2b2f38 });
  const stem = new THREE.Mesh(new THREE.BoxGeometry(0.2, 1.3, 0.2), stemMat);
  stem.position.set(1.0, 1.2, 0);
  group.add(stem);
  const bar = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.2, 0.9), stemMat);
  bar.position.set(1.0, 1.85, 0);
  group.add(bar);
  for (const sx of [-0.9, 0.9]) {
    const wheel = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.55, 0.3), wheelMat);
    wheel.position.set(sx, 0.28, 0);
    wheel.castShadow = true;
    group.add(wheel);
  }
  const headlightMat = new THREE.MeshBasicMaterial({ color: 0xfff4d6 });
  const hl = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.15, 0.4), headlightMat);
  hl.position.set(1.15, 1.0, 0);
  group.add(hl);
  const ledMat = new THREE.MeshBasicMaterial({ color: 0x34d399 });
  const led = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.9, 0.9), ledMat);
  led.position.set(0, 2.0, 0);
  group.add(led);
  // Same uniform-scale approach as buildCarBody -- see its comment.
  group.scale.setScalar(4.8);
  return { group, ledMat, rotors: null };
}

let vehicleMeshes = [];
export function updateVehicles(vehicles) {
  while (vehicleMeshes.length < vehicles.length) {
    const built = buildVehicle();
    vehiclesGroup.add(built.group);
    vehicleMeshes.push(built);
  }
  while (vehicleMeshes.length > vehicles.length) {
    const v = vehicleMeshes.pop();
    vehiclesGroup.remove(v.group);
  }
  vehicles.forEach((v, i) => {
    const entry = vehicleMeshes[i];
    const p = toScene(v.x, v.y); p.y = 0;
    entry.group.position.copy(p);
    // local model forward is +X; world heading = atan2(dy, dx), and
    // three.js's rotation.y maps local +X to (cosθ, 0, -sinθ) in scene
    // space -- exactly the world heading with no extra sign flip needed.
    entry.group.rotation.y = v.heading;
    entry.bodyMat.color.set(v.color);
  });
}
