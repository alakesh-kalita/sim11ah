// Click-and-drag repositioning for STA/Relay/UAV nodes (the AP is excluded
// -- see entities.js's userData.draggable, same rule as the 2D topology
// canvas). Drag preview is entirely client-side (ground-plane raycast +
// world.js's roofHeightAt lookup, so a node visibly rises onto a roof as
// you drag it over one); the new position is only sent to the simulation
// once per node, on release, via POST /api/move_node -- the next poll
// then picks the authoritative position back up.
//
// Drones/UAVs are aircraft, not ground vehicles: dragging one repositions
// where it flies over, not where it lands, so it keeps its own current
// cruise altitude (isAirborne/currentAltitude) instead of following the
// ground/rooftop height every other node uses.
//
// Multi-select: ctrl/shift-click toggles a node in/out of the selection
// (no drag starts on that click, matching the 2D topology canvas's own
// convention); a plain drag-starting on any already-selected node then
// moves the whole group together, preserving each member's offset from
// the others. Deliberately no rubber-band drag-select here (unlike the
// 2D canvas) -- box-selecting in a perspective 3D view means projecting
// every node to screen space and hit-testing a 2D rect, real but
// meaningfully more work than this feature needed to also cover; ctrl/
// shift-click already covers "select several specific nodes".
import * as THREE from 'three';
import { camera, renderer, controls, nodesGroup } from './scene.js';
import { roofHeightAt } from './world.js';
import {
  beginDrag, setDragPosition, endDrag, isAirborne, currentAltitude,
  getNodePosition, setNodeSelected,
} from './entities.js';
import { getCameraMode } from './camera-modes.js';

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const hitPoint = new THREE.Vector3();
const dom = renderer.domElement;

const selectedIds = new Set();
let dragIds = null;                // Set<id> actively being dragged, or null
let dragAnchor = null;             // THREE.Vector3 ground hit at drag start
const dragStartPos = new Map();    // id -> THREE.Vector3 position at drag start

function setPointerFromEvent(e) {
  const rect = dom.getBoundingClientRect();
  pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
}

function pickDraggableNodeId(e) {
  setPointerFromEvent(e);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(nodesGroup.children, true);
  for (const hit of hits) {
    let o = hit.object;
    while (o && o.userData.nodeId === undefined) o = o.parent;
    if (o) return o.userData.draggable ? o.userData.nodeId : null;
  }
  return null;
}

function groundHit(e) {
  setPointerFromEvent(e);
  raycaster.setFromCamera(pointer, camera);
  return raycaster.ray.intersectPlane(groundPlane, hitPoint) ? hitPoint : null;
}

function isAdditiveClick(e) {
  return e.shiftKey || e.ctrlKey || e.metaKey;
}

function clearSelection() {
  for (const id of selectedIds) setNodeSelected(id, false);
  selectedIds.clear();
}

// camera-modes.js's follow mode chases whichever node this selection set
// currently holds -- read-only from there (it only ever iterates it to
// pick a target, never mutates it).
export function getSelectedIds() {
  return selectedIds;
}

dom.addEventListener('pointerdown', (e) => {
  // Free-fly/follow own the canvas's pointer events while active (mouse-
  // look, or nothing at all) -- node click/select/drag only makes sense
  // in plain orbit mode.
  if (getCameraMode() !== 'orbit') return;
  if (e.button !== 0 || dragIds !== null) return;
  const id = pickDraggableNodeId(e);

  if (id === null) {
    // Empty space: a plain click deselects everything (a modified click
    // on empty space is a no-op, not a clear -- holding ctrl/shift while
    // clicking away from any node isn't a deliberate "deselect" gesture).
    if (!isAdditiveClick(e)) clearSelection();
    return;
  }

  if (isAdditiveClick(e)) {
    // Toggle membership only -- same convention as the 2D canvas: this
    // click builds the selection, a later plain drag moves it.
    if (selectedIds.has(id)) { selectedIds.delete(id); setNodeSelected(id, false); }
    else { selectedIds.add(id); setNodeSelected(id, true); }
    return;
  }

  if (!selectedIds.has(id)) {
    // Plain drag-start on a node outside the current selection collapses
    // the selection down to just this one, same as file managers.
    clearSelection();
    selectedIds.add(id);
    setNodeSelected(id, true);
  }

  const anchor = groundHit(e);
  if (!anchor) return;
  dragAnchor = anchor.clone();
  dragIds = new Set(selectedIds);
  dragStartPos.clear();
  for (const sid of dragIds) {
    beginDrag(sid);
    const p = getNodePosition(sid);
    if (p) dragStartPos.set(sid, p);
  }
  controls.enabled = false;
  dom.setPointerCapture(e.pointerId);
  dom.style.cursor = 'grabbing';
  e.preventDefault();
});

dom.addEventListener('pointermove', (e) => {
  if (getCameraMode() !== 'orbit') return;
  if (dragIds === null) {
    dom.style.cursor = pickDraggableNodeId(e) !== null ? 'grab' : 'default';
    return;
  }
  const p = groundHit(e);
  if (!p) return;
  const dx = p.x - dragAnchor.x, dz = p.z - dragAnchor.z;
  for (const id of dragIds) {
    const start = dragStartPos.get(id);
    if (!start) continue;
    const nx = start.x + dx, nz = start.z + dz;
    // Airborne nodes hold their own starting altitude through the whole
    // drag (only their ground-track moves); everything else follows
    // world.js's roof lookup at its OWN new (x, z), not the drag
    // anchor's -- each member can cross a different rooftop.
    const ny = isAirborne(id) ? start.y : (roofHeightAt(nx, nz) ?? 0);
    setDragPosition(id, nx, ny, nz);
  }
});

function finishDrag() {
  if (dragIds === null) return;
  const ids = dragIds;
  dragIds = null;
  dragAnchor = null;
  controls.enabled = true;
  dom.style.cursor = 'default';
  for (const id of ids) {
    endDrag(id);
    const pos = getNodePosition(id);
    if (!pos) continue;
    // toScene(x, y) = (x, 0, -y) -- invert to recover sim-space (x, y).
    fetch('/api/move_node', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, x: pos.x, y: -pos.z }),
    }).catch(() => {});
  }
  dragStartPos.clear();
}

dom.addEventListener('pointerup', finishDrag);
dom.addEventListener('pointercancel', finishDrag);
