// Game-style camera modes layered on top of the default OrbitControls
// view (scene.js) -- an explicit request to make the 3D view feel more
// like a game to fly/drive through, not just a diorama to orbit around.
//
// Three modes, one at a time:
//   'orbit'   -- the original always-on behaviour (drag/scroll/pan via
//                OrbitControls). Default, and where every other mode
//                returns to.
//   'freefly' -- WASD + mouse-look noclip flycam (Minecraft-spectator-
//                style): looks and moves independently of any node.
//   'follow'  -- third-person chase camera locked onto whichever node
//                is currently selected (interact.js's own selection set,
//                the same one ctrl/shift-click already builds for
//                dragging) -- follows its live position, chasing from
//                behind its direction of travel.
//
// F toggles free-fly, G toggles follow (needs a selection already made
// the normal way first), Escape always forces back to orbit. interact.js
// checks getCameraMode() itself and skips its own node click/drag/select
// handling outside 'orbit' -- both free-fly's mouse-look and follow's
// fully automatic tracking would otherwise fight over the same pointer
// events on the same canvas.
import * as THREE from 'three';
import { camera, controls, renderer } from './scene.js';
import { getNodePosition } from './entities.js';
import { getSelectedIds } from './interact.js';

const dom = renderer.domElement;
const hudCameraEl = document.getElementById('hud-camera');

let mode = 'orbit';
export function getCameraMode() {
  return mode;
}

const MODE_LABEL = { orbit: 'Orbit', freefly: 'Free-fly', follow: 'Chase' };
function setMode(next) {
  mode = next;
  if (hudCameraEl) hudCameraEl.textContent = MODE_LABEL[mode];
}

// ---- free-fly --------------------------------------------------------
const FREEFLY_SPEED = 100;     // metres/sec at the base (non-sprint) rate
const FREEFLY_SPRINT = 3.0;    // Shift multiplier -- crossing a multi-km
                                // scene at the base rate alone would feel
                                // like wading through it
const LOOK_SENSITIVITY = 0.0035;
const MAX_PITCH = Math.PI / 2 - 0.02; // just short of straight up/down --
                                       // exactly +-90deg is where a plain
                                       // YXZ euler's yaw becomes undefined

const keys = { w: false, a: false, s: false, d: false, space: false, ctrl: false, shift: false };
let yaw = 0, pitch = 0;
let looking = false;
let lastLookX = 0, lastLookY = 0;

function syncYawPitchFromCamera() {
  const e = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
  yaw = e.y;
  pitch = e.x;
}

function applyYawPitch() {
  camera.quaternion.setFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));
}

function stepFreeFly(dt) {
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
  const move = new THREE.Vector3();
  if (keys.w) move.add(forward);
  if (keys.s) move.addScaledVector(forward, -1);
  if (keys.d) move.add(right);
  if (keys.a) move.addScaledVector(right, -1);
  const speed = FREEFLY_SPEED * (keys.shift ? FREEFLY_SPRINT : 1.0) * dt;
  if (move.lengthSq() > 1e-9) move.normalize().multiplyScalar(speed);
  if (keys.space) move.y += speed;
  if (keys.ctrl) move.y -= speed;
  camera.position.add(move);
}

// ---- follow (chase cam) ------------------------------------------------
const FOLLOW_DISTANCE = 55;
const FOLLOW_HEIGHT = 26;
const FOLLOW_LOOK_HEIGHT = 6;
const FOLLOW_POS_SMOOTH = 0.06;   // camera position lerp factor/frame
const FOLLOW_DIR_SMOOTH = 0.15;   // chase-direction lerp factor/frame --
                                   // smooths out a vehicle's instantaneous
                                   // turn-corner velocity spike instead of
                                   // snapping the camera to face it exactly
const FOLLOW_MIN_SPEED_SQ = 0.01; // below this, keep the last known
                                   // direction rather than chasing noise
                                   // from a stationary/barely-moving node

let followId = null;
let followPrevPos = null;
let followDir = new THREE.Vector3(0, 0, 1);

function pickFollowTarget() {
  const ids = getSelectedIds();
  for (const id of ids) return id; // first (only, in practice) selection
  return null;
}

function stepFollow(dt) {
  if (followId === null) return exitToOrbit();
  const pos = getNodePosition(followId);
  if (!pos) return exitToOrbit(); // node gone (rebuild, topology switch)

  if (followPrevPos) {
    const vel = pos.clone().sub(followPrevPos);
    if (dt > 1e-6 && vel.lengthSq() / (dt * dt) > FOLLOW_MIN_SPEED_SQ) {
      followDir.lerp(vel.normalize(), FOLLOW_DIR_SMOOTH);
      if (followDir.lengthSq() > 1e-9) followDir.normalize();
    }
  }
  followPrevPos = pos.clone();

  const desired = pos.clone()
    .addScaledVector(followDir, -FOLLOW_DISTANCE)
    .add(new THREE.Vector3(0, FOLLOW_HEIGHT, 0));
  camera.position.lerp(desired, FOLLOW_POS_SMOOTH);
  camera.lookAt(pos.x, pos.y + FOLLOW_LOOK_HEIGHT, pos.z);
}

// ---- mode transitions ---------------------------------------------------
function enterFreeFly() {
  setMode('freefly');
  controls.enabled = false;
  syncYawPitchFromCamera();
}

function enterFollow() {
  const id = pickFollowTarget();
  if (id === null) return; // nothing selected -- no-op, stay in whatever mode we're in
  setMode('follow');
  controls.enabled = false;
  followId = id;
  followPrevPos = null;
  followDir.set(0, 0, 1);
}

function exitToOrbit() {
  if (mode === 'orbit') return;
  setMode('orbit');
  followId = null;
  followPrevPos = null;
  // Aim orbit's own target at whatever the camera was just looking at, so
  // returning to orbit doesn't snap the view to some unrelated old target
  // (e.g. wherever the very first frameCameraOnNodes() pointed it, poll
  // cycles ago).
  const lookDir = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).addScaledVector(lookDir, 120);
  controls.enabled = true;
}

function toggleFreeFly() {
  if (mode === 'freefly') exitToOrbit();
  else enterFreeFly();
}

function toggleFollow() {
  if (mode === 'follow') exitToOrbit();
  else enterFollow();
}

// ---- input ---------------------------------------------------------------
window.addEventListener('keydown', (e) => {
  if (e.repeat) return;
  const k = e.key.toLowerCase();
  if (k === 'f') { toggleFreeFly(); e.preventDefault(); return; }
  if (k === 'g') { toggleFollow(); e.preventDefault(); return; }
  if (k === 'escape') { exitToOrbit(); return; }
  if (mode !== 'freefly') return;
  // preventDefault on the movement keys mainly guards against Space's
  // default page-scroll action -- harmless today (index.html's body has
  // no scrollable overflow) but cheap insurance against that changing.
  if (k === 'w') { keys.w = true; e.preventDefault(); }
  else if (k === 'a') { keys.a = true; e.preventDefault(); }
  else if (k === 's') { keys.s = true; e.preventDefault(); }
  else if (k === 'd') { keys.d = true; e.preventDefault(); }
  else if (k === ' ') { keys.space = true; e.preventDefault(); }
  else if (k === 'control') { keys.ctrl = true; e.preventDefault(); }
  else if (k === 'shift') { keys.shift = true; e.preventDefault(); }
});
window.addEventListener('keyup', (e) => {
  const k = e.key.toLowerCase();
  if (k === 'w') keys.w = false;
  else if (k === 'a') keys.a = false;
  else if (k === 's') keys.s = false;
  else if (k === 'd') keys.d = false;
  else if (k === ' ') keys.space = false;
  else if (k === 'control') keys.ctrl = false;
  else if (k === 'shift') keys.shift = false;
});
// Losing focus (alt-tab, clicking another window) with a key physically
// still held would otherwise leave that key "stuck" on -- nothing ever
// fires its keyup on this page to clear it.
window.addEventListener('blur', () => {
  keys.w = keys.a = keys.s = keys.d = keys.space = keys.ctrl = keys.shift = false;
  looking = false;
});

dom.addEventListener('pointerdown', (e) => {
  if (mode !== 'freefly' || e.button !== 0) return;
  looking = true;
  lastLookX = e.clientX;
  lastLookY = e.clientY;
  dom.setPointerCapture(e.pointerId);
  dom.style.cursor = 'grabbing';
});
dom.addEventListener('pointermove', (e) => {
  if (mode !== 'freefly' || !looking) return;
  const dx = e.clientX - lastLookX, dy = e.clientY - lastLookY;
  lastLookX = e.clientX;
  lastLookY = e.clientY;
  yaw -= dx * LOOK_SENSITIVITY;
  pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, pitch - dy * LOOK_SENSITIVITY));
  applyYawPitch();
});
function stopLooking(e) {
  if (!looking) return;
  looking = false;
  dom.style.cursor = 'default';
}
dom.addEventListener('pointerup', stopLooking);
dom.addEventListener('pointercancel', stopLooking);

export function stepCameraModes(dt) {
  if (mode === 'freefly') stepFreeFly(dt);
  else if (mode === 'follow') stepFollow(dt);
}
