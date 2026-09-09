// Shared constants and small pure-function utilities used across the
// other modules -- no THREE scene state lives here, just building blocks.
import * as THREE from 'three';

export const POLL_MS = 300;
export const ASSOC = { UNASSOCIATED: 0, AUTHENTICATING: 1, AUTHENTICATED: 2, ASSOCIATING: 3, ASSOCIATED: 4 };
export const PKT_COLORS = { DATA: 0x38bdf8, ACK: 0x34d399, BEACON: 0xfbbf24, RTS: 0xf472b6, TWT: 0xa78bfa };
export const STATUS_COLOR = { ASSOCIATED: 0x34d399, UNASSOCIATED: 0xf87171, PENDING: 0xfbbf24 };
export const SKY_COLOR = 0x8fc7ee;

// World (x, y) metres -> scene (x, 0, z). World +y maps to scene -z so a
// top-down "north = +y" reads with the same handedness as the 2D map.
export function toScene(x, y) {
  return new THREE.Vector3(x, 0, -y);
}

export function hashSeed(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Every material needs its own repeat-count, so shared/cached source
// textures always get cloned before a `.repeat` is set on them -- setting
// `.repeat` on a texture two different materials both reference would
// make the later call win for both.
export function tiledClone(tex, repeatX, repeatY) {
  const t = tex.clone();
  t.needsUpdate = true;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(Math.max(1, repeatX), Math.max(1, repeatY));
  return t;
}
