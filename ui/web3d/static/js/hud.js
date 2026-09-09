// DOM overlay: the stat box + the connection-status banner. No THREE
// state here, just element references and text updates.
const hud = {
  env: document.getElementById('hud-env'), time: document.getElementById('hud-time'),
  pdr: document.getElementById('hud-pdr'), thr: document.getElementById('hud-thr'),
  delay: document.getElementById('hud-delay'), nodes: document.getElementById('hud-nodes'),
};
const statusEl = document.getElementById('status');

export function updateHud(state) {
  hud.env.textContent = state.environment ?? '—';
  hud.time.textContent = state.time.toFixed(2) + ' s';
  const m = state.metrics ?? {};
  hud.pdr.textContent = m.pdr !== undefined ? (m.pdr * 100).toFixed(1) + '%' : '—';
  hud.thr.textContent = m.thr_kbps !== undefined ? m.thr_kbps.toFixed(2) + ' kb/s' : '—';
  hud.delay.textContent = m.lat_avg_ms !== undefined ? m.lat_avg_ms.toFixed(1) + ' ms' : '—';
  hud.nodes.textContent = String(state.nodes.length);
}

export function showStatus(text) {
  statusEl.style.display = 'block';
  statusEl.textContent = text;
}

export function hideStatus() {
  statusEl.style.display = 'none';
}
