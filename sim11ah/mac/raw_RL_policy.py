from __future__ import annotations

import collections
import math
import os
import pickle
import random
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from sim11ah.mac.common import (
    MORSE_RAW_MIN_SLOT_DURATION_US,
    RawBeaconSpreading,
    RawConfig,
    RawPeriodic,
    RawSlotDefinition,
)

# ---------------------------------------------------------------------------
# Discrete action indices — 9 actions
# ---------------------------------------------------------------------------
_ACT_KEEP        = 0
_ACT_SLOT_INC    = 1
_ACT_SLOT_DEC    = 2
_ACT_GROUPS_INC  = 3
_ACT_GROUPS_DEC  = 4
_ACT_NSLOTS_INC  = 5
_ACT_NSLOTS_DEC  = 6
_ACT_PHASES_INC  = 7
_ACT_PHASES_DEC  = 8
_N_ACTIONS       = 9

_GROUP_CHOICES_DEFAULT: List[int] = [1, 2, 4, 8]
_NSLOT_CHOICES_DEFAULT: List[int] = [1, 2, 4, 8]

# ---------------------------------------------------------------------------
# Raw feature indices (continuous, used by DQN/DDQN)
# ---------------------------------------------------------------------------
_F_COL    = 0   # collision fraction
_F_PDR    = 1   # per-phase PDR estimate
_F_N      = 2   # normalised total-N estimate
_F_G      = 3   # normalised groups index
_F_SLOT   = 4   # normalised slot fraction
_F_TREND  = 5   # PDR trend (0/1)
_F_RETRY  = 6   # MAC retry rate
_F_PHASE  = 7   # normalised phase count
_N_FEATURES = 8


def _snap_grid_up(x_us: int) -> int:
    """Round UP to the nearest valid RAW slot-duration cslot grid point
    (500 + 120*cslot us, per IEEE 802.11ah Sec 9.4.2.200)."""
    return -(-(int(x_us) - MORSE_RAW_MIN_SLOT_DURATION_US) // 120) * 120 \
        + MORSE_RAW_MIN_SLOT_DURATION_US


def _snap_grid_down(x_us: int) -> int:
    """Round DOWN to the nearest valid RAW slot-duration cslot grid point
    (500 + 120*cslot us, per IEEE 802.11ah Sec 9.4.2.200)."""
    k = max(0, (int(x_us) - MORSE_RAW_MIN_SLOT_DURATION_US) // 120)
    return MORSE_RAW_MIN_SLOT_DURATION_US + k * 120


def _parse_thresholds(raw, defaults: List[float]) -> List[float]:
    if raw is None:
        return defaults
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(x) for x in str(raw).split(",")]


# ---------------------------------------------------------------------------
# Running normaliser (mean / std, online Welford)
# ---------------------------------------------------------------------------
class _RunningNorm:
    """Online mean/std normalisation for neural network input."""

    def __init__(self, n: int) -> None:
        self._n    = 0
        self._mean = np.zeros(n, dtype=np.float64)
        self._M2   = np.ones(n, dtype=np.float64)   # init to 1 avoids div/0 early

    def update(self, x: np.ndarray) -> None:
        self._n += 1
        delta      = x - self._mean
        self._mean += delta / self._n
        self._M2   += delta * (x - self._mean)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self._M2 / max(1, self._n))
        return (x - self._mean) / np.where(std > 1e-8, std, 1.0)

    def to_dict(self) -> Dict:
        return {"n": self._n, "mean": self._mean.tolist(), "M2": self._M2.tolist()}

    def from_dict(self, d: Dict) -> None:
        self._n    = int(d["n"])
        self._mean = np.array(d["mean"], dtype=np.float64)
        self._M2   = np.array(d["M2"],   dtype=np.float64)


# ---------------------------------------------------------------------------
# Incremental PCA (batch-fit after warmup, then project)
# ---------------------------------------------------------------------------
class _OnlinePCA:
    """Fits PCA on the first `warmup` samples, then holds components fixed."""

    def __init__(self, n_features: int, n_components: int, warmup: int) -> None:
        self._nf   = n_features
        self._nc   = n_components
        self._wu   = warmup
        self._buf: List[np.ndarray] = []
        self._components: Optional[np.ndarray] = None   # (n_components, n_features)
        self._mean_fit: Optional[np.ndarray] = None     # training mean for centering

    @property
    def is_fitted(self) -> bool:
        return self._components is not None

    def update(self, x: np.ndarray) -> None:
        if self.is_fitted:
            return
        self._buf.append(x.copy())
        if len(self._buf) >= self._wu:
            X = np.stack(self._buf)          # (N, n_features)
            self._mean_fit = X.mean(axis=0)  # save mean so transform can center
            X -= self._mean_fit
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            self._components = Vt[: self._nc]   # (n_components, n_features)
            self._buf = []

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return x[: self._nc]             # rough fallback before fitting
        return self._components @ (x - self._mean_fit)


# ---------------------------------------------------------------------------
# Tiny 2-layer ReLU network (pure numpy, no external ML libraries)
# ---------------------------------------------------------------------------
class _TinyNet:
    """
    Two-layer ReLU feedforward network.
    Forward:  h = ReLU(x @ W1 + b1),  q = h @ W2 + b2
    Update:   one SGD step minimising MSE on a single (action, target) pair.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 rng: random.Random) -> None:
        # He initialisation for ReLU
        s1 = math.sqrt(2.0 / n_in)
        s2 = math.sqrt(2.0 / n_hidden)
        self.W1 = np.array([[rng.gauss(0, s1) for _ in range(n_hidden)]
                             for _ in range(n_in)], dtype=np.float64)
        self.b1 = np.zeros(n_hidden, dtype=np.float64)
        self.W2 = np.array([[rng.gauss(0, s2) for _ in range(n_out)]
                             for _ in range(n_hidden)], dtype=np.float64)
        self.b2 = np.zeros(n_out, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def update(self, x: np.ndarray, action: int,
               target_q: float, lr: float) -> None:
        """SGD update: minimise (q[action] - target_q)^2."""
        h_pre = x @ self.W1 + self.b1
        h     = np.maximum(0.0, h_pre)
        q     = h @ self.W2 + self.b2

        dq        = np.zeros_like(q)
        dq[action] = q[action] - target_q          # gradient of MSE

        dW2 = np.outer(h, dq)
        db2 = dq
        dh  = self.W2 @ dq
        dh_pre = dh * (h_pre > 0.0)
        dW1 = np.outer(x, dh_pre)
        db1 = dh_pre

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def copy(self) -> "_TinyNet":
        obj = _TinyNet.__new__(_TinyNet)
        obj.W1 = self.W1.copy()
        obj.b1 = self.b1.copy()
        obj.W2 = self.W2.copy()
        obj.b2 = self.b2.copy()
        return obj

    def to_dict(self) -> Dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def from_dict(self, d: Dict) -> None:
        self.W1 = np.array(d["W1"]); self.b1 = np.array(d["b1"])
        self.W2 = np.array(d["W2"]); self.b2 = np.array(d["b2"])


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class _ReplayBuffer:
    """Fixed-capacity circular buffer of (s, a, r, s_next, bkt_next, gamma_bg).

    bkt_next: 8-tuple bucket state for s_next (used to compute action mask on
              the next state, so the Bellman max is taken only over valid actions).
    gamma_bg: bootstrap discount stored at push time (γ^n for n-step returns,
              or plain γ for 1-step), so replay updates use the correct discount
              rather than the discount of the current step.
    """

    def __init__(self, capacity: int) -> None:
        self._buf: List[Tuple] = []
        self._cap = capacity
        self._ptr = 0

    def push(self, s, a: int, r: float, s_next,
             bkt_next=None, gamma_bg: float = 0.0) -> None:
        entry = (s, a, r, s_next, bkt_next, gamma_bg)
        if len(self._buf) < self._cap:
            self._buf.append(entry)
        else:
            self._buf[self._ptr % self._cap] = entry
        self._ptr += 1

    def sample(self, n: int, rng: random.Random) -> List[Tuple]:
        k = min(n, len(self._buf))
        return rng.sample(self._buf, k) if k > 0 else []

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# GAE helper (used by PPO)
# ---------------------------------------------------------------------------
def _compute_gae(
    rewards: List[float],
    values:  List[float],
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    """Generalized Advantage Estimation.

    Returns (advantages, returns) where returns[t] = advantages[t] + values[t].
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        v_next = values[t + 1] if t + 1 < T else next_value
        delta  = rewards[t] + gamma * v_next - values[t]
        gae    = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


# ---------------------------------------------------------------------------
# Policy network with softmax output (for PPO)
# ---------------------------------------------------------------------------
class _PolicyNet:
    """
    2-layer ReLU network with softmax output.
    Used as the actor in PPO.
    update() applies the clipped PPO surrogate gradient + entropy bonus.
    """

    def __init__(self, n_in: int, n_hidden: int, n_actions: int,
                 rng: random.Random) -> None:
        s1 = math.sqrt(2.0 / n_in)
        s2 = math.sqrt(2.0 / n_hidden)
        self.W1 = np.array([[rng.gauss(0, s1) for _ in range(n_hidden)]
                             for _ in range(n_in)], dtype=np.float64)
        self.b1 = np.zeros(n_hidden, dtype=np.float64)
        self.W2 = np.array([[rng.gauss(0, s2) for _ in range(n_actions)]
                             for _ in range(n_hidden)], dtype=np.float64)
        self.b2 = np.zeros(n_actions, dtype=np.float64)

    def _forward_all(self, x: np.ndarray):
        """Returns (h_pre, h, logits, probs)."""
        h_pre  = x @ self.W1 + self.b1
        h      = np.maximum(0.0, h_pre)
        logits = h @ self.W2 + self.b2
        # Numerically stable softmax
        l_s    = logits - logits.max()
        exp_l  = np.exp(l_s)
        probs  = exp_l / exp_l.sum()
        return h_pre, h, logits, probs

    def probs(self, x: np.ndarray,
              mask: Optional[np.ndarray] = None) -> np.ndarray:
        _, _, logits, _ = self._forward_all(x)
        if mask is not None:
            logits = logits.copy()
            logits[~mask] = -1e9
        l_s   = logits - logits.max()
        exp_l = np.exp(l_s)
        return exp_l / exp_l.sum()

    def sample(self, x: np.ndarray,
               mask: Optional[np.ndarray] = None,
               rng: Optional[random.Random] = None) -> Tuple[int, float]:
        """Sample an action; return (action, log_prob)."""
        p = self.probs(x, mask)
        if rng is not None:
            action = rng.choices(range(len(p)), weights=p.tolist())[0]
        else:
            action = int(np.argmax(p))
        log_prob = float(np.log(float(p[action]) + 1e-8))
        return action, log_prob

    def ppo_update(self, x: np.ndarray, action: int,
                   advantage: float, old_log_prob: float,
                   lr: float, clip_eps: float,
                   entropy_coef: float,
                   mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Single PPO gradient step. Returns (actor_loss, entropy).

        mask: boolean array of shape (_N_ACTIONS,).  When provided, the new
        log-probability is computed under the same constrained distribution
        used during data collection so the importance-sampling ratio ρ_t is
        consistent.  Forbidden actions are zeroed out before renormalisation.
        """
        h_pre, h, _, probs = self._forward_all(x)
        if mask is not None:
            probs = probs * mask
            total = probs.sum()
            if total > 1e-12:
                probs = probs / total
        log_probs = np.log(probs + 1e-8)
        log_prob  = float(log_probs[action])
        ratio     = math.exp(log_prob - old_log_prob)

        surr1 = ratio * advantage
        surr2 = float(np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)) * advantage
        actor_loss = -min(surr1, surr2)

        entropy = -float(np.sum(probs * log_probs))

        # d(actor_loss)/d(log_prob_new): 0 when clipping is active
        is_clipped = (advantage > 0 and ratio > 1.0 + clip_eps) or \
                     (advantage < 0 and ratio < 1.0 - clip_eps)
        g_log_prob = 0.0 if is_clipped else -advantage * ratio

        # d(log_prob)/d(logits) = e_a - probs
        e_a = np.zeros_like(probs); e_a[action] = 1.0
        d_logits_actor = g_log_prob * (e_a - probs)

        # d(-entropy)/d(logits) = probs * (log_probs + H)
        # Because d_logits_ent = -dH/dlogits, the entropy bonus
        # L = L_actor - entropy_coef*H needs +entropy_coef here (not minus).
        d_logits_ent = probs * (log_probs + entropy)

        d_logits = d_logits_actor + entropy_coef * d_logits_ent

        dW2    = np.outer(h, d_logits)
        db2    = d_logits
        dh     = self.W2 @ d_logits
        dh_pre = dh * (h_pre > 0.0)
        dW1    = np.outer(x, dh_pre)
        db1    = dh_pre

        self.W1 -= lr * dW1;  self.b1 -= lr * db1
        self.W2 -= lr * dW2;  self.b2 -= lr * db2
        return actor_loss, entropy

    def copy(self) -> "_PolicyNet":
        obj = _PolicyNet.__new__(_PolicyNet)
        obj.W1 = self.W1.copy(); obj.b1 = self.b1.copy()
        obj.W2 = self.W2.copy(); obj.b2 = self.b2.copy()
        return obj

    def to_dict(self) -> Dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def from_dict(self, d: Dict) -> None:
        self.W1 = np.array(d["W1"]); self.b1 = np.array(d["b1"])
        self.W2 = np.array(d["W2"]); self.b2 = np.array(d["b2"])


# ---------------------------------------------------------------------------
# On-policy rollout buffer (for PPO)
# ---------------------------------------------------------------------------
class _RolloutBuffer:
    """Fixed-capacity on-policy buffer; cleared after each PPO update.

    Each transition stores the boolean action mask used at sampling time so
    that ppo_update() can recompute log-probabilities under the SAME constrained
    distribution (ρ_t = π_new(a|s,mask) / π_old(a|s,mask)).
    """

    def __init__(self, capacity: int) -> None:
        self._buf: List[Tuple] = []
        self._cap = capacity

    def push(self, state: np.ndarray, action: int, reward: float,
             log_prob: float, value: float,
             mask: Optional[np.ndarray] = None) -> None:
        if len(self._buf) < self._cap:
            self._buf.append((state, action, reward, log_prob, value, mask))

    def is_full(self) -> bool:
        return len(self._buf) >= self._cap

    def get_and_clear(self) -> List[Tuple]:
        data = list(self._buf)
        self._buf.clear()
        return data

    def __len__(self) -> int:
        return len(self._buf)


# ===========================================================================
# Main policy class
# ===========================================================================
class RLRawPolicy:
    """
    Multi-algorithm adaptive RAW policy for IEEE 802.11ah — v8.

    Supported modes  (rl_raw_mode config key)
    ------------------------------------------
    tabular        — standard Q-table  (default, most lightweight)
    tabular_ddqn   — Double Q-learning (two dict Q-tables, no neural net;
                     reduces overestimation bias; abbreviated Tab-DDQN)
    dqn            — Deep Q-Network    (numpy 2-layer ReLU, generalises across states)
    ddqn           — Double DQN        (online + target net, lowest bias)
    ppo            — Proximal Policy Optimisation (actor-critic, on-policy)

    PCA preprocessing  (rl_raw_use_pca = true, any neural-net mode)
    ------------------------------------------------------------------
    Projects raw 8-dimensional continuous feature vector to k principal
    components (fitted on the first pca_warmup transitions, then fixed).
    Reduces input dimensionality and removes correlated features.

    State (8-tuple for tabular modes)
    -----------------------------------
    (col_bucket, pdr_bucket, sta_bucket, groups_idx, slot_bucket,
     trend, retry_bucket, phase_bucket)

    Bucket encoding notes
    ·  col_bucket   : 0=low(<10%), 1=medium, 2=high(≥30%)  [descending]
    ·  pdr_bucket   : 0=good(≥60%), 1=moderate, 2=poor(<30%) [ascending —
                      descending=False; higher bucket = worse PDR]
    ·  retry_bucket : 0=low(<30%), 1=medium, 2=high(≥60%)  [descending]
                      captures within-slot contention delay
    ·  phase_bucket : 0=P=1, 1=P∈{2,3}, 2=P≥4             [descending]
                      captures inter-beacon waiting delay (not retry_bucket)

    Raw feature vector (8-dim, for DQN/DDQN)
    -----------------------------------------
    [col_rate, pdr_hat, n_norm, groups_frac, slot_frac,
     trend, retry_rate, phase_norm]

    Reward (6 terms)
    ----------------
    r = w_pdr×pdr − w_col×col − w_oh×overhead
        − w_retry×retry_rate − w_phase×(P−1)T_B/T_traffic
        − w_slot×slot_deficit

    All configurable from cfg["mac"]  (prefix: rl_raw_)
    -------------------------------------------------------
    mode, use_pca, pca_components, pca_warmup,
    nn_hidden, nn_lr, target_update_freq,
    ppo_clip_eps, ppo_epochs, ppo_rollout_len, gae_lambda,
    ppo_entropy_coef, ppo_value_coef,
    alpha, gamma, epsilon, epsilon_min, epsilon_decay, cold_start_steps, td_clip,
    replay_capacity, batch_size, updates_per_step,
    w_pdr, w_collision, w_overhead, w_retry, w_phase_delay, w_slot,
    ewma_alpha, ewma_pdr_init, q_keep_bias, q_init_bias,
    target_stas_slot, phase_load_margin,
    col_thresholds, pdr_thresholds, sta_thresholds,
    retry_thresholds, slot_thresholds, phase_thresholds,
    min_slot_us, max_slot_us, step_us, initial_slot_us,
    seed, qtable_path
    """

    def __init__(self, ctx, log_fn) -> None:
        self.ctx  = ctx
        self._log = log_fn

        mac_cfg = self.ctx.cfg["mac"]
        app_cfg = self.ctx.cfg.get("app", {})
        phy_cfg = self.ctx.cfg.get("phy", {})
        net_cfg = self.ctx.cfg.get("net", {})

        # ------------------------------------------------------------------
        # Algorithm mode
        # ------------------------------------------------------------------
        self._mode: str = str(mac_cfg.get("rl_raw_mode", "tabular")).lower()
        assert self._mode in ("tabular", "tabular_ddqn", "dqn", "ddqn", "ppo"), \
            f"rl_raw_mode must be tabular/tabular_ddqn/dqn/ddqn/ppo, got {self._mode!r}"

        # ------------------------------------------------------------------
        # Discrete parameter choices — configurable from cfg
        # ------------------------------------------------------------------
        def _parse_int_list(val, default: List[int]) -> List[int]:
            if val is None:
                return list(default)
            if isinstance(val, (list, tuple)):
                return sorted(int(x) for x in val)
            return sorted(int(x) for x in str(val).split(","))

        self._group_choices: List[int] = _parse_int_list(
            mac_cfg.get("rl_raw_group_choices"), _GROUP_CHOICES_DEFAULT)
        self._nslot_choices: List[int] = _parse_int_list(
            mac_cfg.get("rl_raw_nslot_choices"), _NSLOT_CHOICES_DEFAULT)
        assert len(self._group_choices) >= 2, "rl_raw_group_choices needs ≥2 values"
        assert len(self._nslot_choices) >= 2, "rl_raw_nslot_choices needs ≥2 values"

        # ------------------------------------------------------------------
        # Conservative scheduling-reliability minimum slot duration.
        # This is NOT an absolute physical impossibility threshold; it is
        # the slot width below which a complete DATA-ACK exchange plus
        # average backoff cannot fit, making successful delivery very
        # unlikely under typical contention conditions.
        # ------------------------------------------------------------------
        preamble_us = int(float(phy_cfg.get("preamble_time", 320e-6)) * 1e6)
        header_us   = int(float(phy_cfg.get("header_time",   80e-6))  * 1e6)
        sifs_us     = int(float(mac_cfg.get("sifs",  160e-6)) * 1e6)
        difs_us     = int(float(mac_cfg.get("difs",  264e-6)) * 1e6)
        sigma_us    = int(float(mac_cfg.get("slot_time", 52e-6)) * 1e6)
        pkt_b       = int(app_cfg.get("packet_size_bytes",       128))
        mac_oh      = int(mac_cfg.get("data_mac_overhead_bytes",  36))
        net_oh      = int(net_cfg.get("net_header_bytes",         16))
        ack_b       = int(mac_cfg.get("ack_size_bytes",           14))
        mode_table  = phy_cfg.get("mode_table", {}) or {}
        data_mode   = str(phy_cfg.get("default_mode", "MCS0"))
        rate_bps    = float(mode_table.get(data_mode, 150_000.0))
        t_data_us   = preamble_us + header_us + int(math.ceil(8*(pkt_b+mac_oh+net_oh)/rate_bps*1e6))
        t_ack_us    = preamble_us + header_us + int(math.ceil(8*ack_b/rate_bps*1e6))
        t_suc_us    = t_data_us + sifs_us + t_ack_us + difs_us
        cw_min      = int(mac_cfg.get("cw_min", 15))
        avg_bo_us   = int((cw_min / 2) * sigma_us)
        guard_us    = int(float(mac_cfg.get("raw_guard", 100e-6)) * 1e6)
        self._phys_min_us = max(MORSE_RAW_MIN_SLOT_DURATION_US,
                                t_suc_us + avg_bo_us + guard_us)

        # ------------------------------------------------------------------
        # Slot-duration bounds
        # ------------------------------------------------------------------
        base_slot_us = int(float(getattr(self.ctx, "raw_slot_duration", 0.007)) * 1e6)
        # Snap onto the RAW slot-duration cslot grid (500 + 120*cslot us) so
        # that raw.py's us_to_cslot()/cslot_to_us() round-trip (which floors
        # onto this grid) cannot return a value outside [min_slot_us, max_slot_us].
        self.min_slot_us = _snap_grid_up(max(
            self._phys_min_us,
            int(mac_cfg.get("rl_raw_min_slot_us", self._phys_min_us)),
        ))
        self.max_slot_us = _snap_grid_down(max(self.min_slot_us, int(
            mac_cfg.get(
                "rl_raw_max_slot_us",
                max(self.min_slot_us, base_slot_us * 2, 12_000),
            )
        )))
        self.step_us = max(500, int(mac_cfg.get("rl_raw_step_us", 2_000)))

        # ------------------------------------------------------------------
        # Beacon / traffic timing
        # ------------------------------------------------------------------
        self._beacon_interval_s  = float(mac_cfg.get("beacon_interval", 0.5))
        self._traffic_interval_s = float(app_cfg.get("periodic_interval", 5.0))
        beacon_interval_us       = int(self._beacon_interval_s * 1e6)
        raw_budget_frac          = float(mac_cfg.get("adaptive_raw_budget_fraction", 0.90))
        self._beacon_budget_us   = int(raw_budget_frac * beacon_interval_us)
        self._tx_fraction        = min(
            1.0,
            self._beacon_interval_s / max(1e-9, self._traffic_interval_s),
        )
        self._max_phases: int = max(
            1, int(self._traffic_interval_s / self._beacon_interval_s)
        )

        # Physics-derived max useful slots (G×S s.t. T_slot ≥ T_min)
        _feasible = [g * s for g in self._group_choices for s in self._nslot_choices
                     if g * s <= self._beacon_budget_us // max(1, self._phys_min_us)]
        self._max_useful_slots: int = max(_feasible) if _feasible else 1

        # Heuristic parameters (all configurable)
        self._target_stas_slot:  int   = int(  mac_cfg.get("rl_raw_target_stas_slot",  25))
        self._phase_load_margin: float = float(mac_cfg.get("rl_raw_phase_load_margin", 0.9))

        # ------------------------------------------------------------------
        # Initial RAW parameters
        # ------------------------------------------------------------------
        init_slot_cfg = mac_cfg.get("rl_raw_initial_slot_us", None)
        if init_slot_cfg is not None:
            self._slot_us       = self._quantize(int(init_slot_cfg))
            self._slot_override = True
        else:
            self._slot_us       = self.max_slot_us
            self._slot_override = False

        init_g = max(1, int(mac_cfg.get("raw_num_groups", getattr(self.ctx, "raw_num_groups", 1))))
        init_s = max(1, int(mac_cfg.get("raw_num_slots",  getattr(self.ctx, "raw_num_slots",  1))))
        self._groups_idx: int = self._nearest_idx(self._group_choices, init_g)
        self._nslots_idx: int = self._nearest_idx(self._nslot_choices, init_s)

        # Phase management
        self._num_phases:    int = 1
        self._current_phase: int = 0

        # ------------------------------------------------------------------
        # Q-learning hyper-parameters
        # ------------------------------------------------------------------
        self.alpha            = float(mac_cfg.get("rl_raw_alpha",          0.20))
        self.gamma            = float(mac_cfg.get("rl_raw_gamma",          0.90))
        self.epsilon          = float(mac_cfg.get("rl_raw_epsilon",        0.40))
        self.epsilon_min      = float(mac_cfg.get("rl_raw_epsilon_min",    0.05))
        self.epsilon_decay    = float(mac_cfg.get("rl_raw_epsilon_decay",  0.985))
        self.cold_start_steps = int(  mac_cfg.get("rl_raw_cold_start_steps", 5))
        self._td_clip         = float(mac_cfg.get("rl_raw_td_clip",        5.0))

        self.replay_capacity  = int(mac_cfg.get("rl_raw_replay_capacity",  500))
        self.batch_size       = int(mac_cfg.get("rl_raw_batch_size",        16))
        self.updates_per_step = int(mac_cfg.get("rl_raw_updates_per_step",   5))

        # Reward weights
        self.w_pdr         = float(mac_cfg.get("rl_raw_w_pdr",         1.0))
        self.w_collision   = float(mac_cfg.get("rl_raw_w_collision",    0.4))
        self.w_overhead    = float(mac_cfg.get("rl_raw_w_overhead",     0.1))
        self.w_retry       = float(mac_cfg.get("rl_raw_w_retry",        0.3))
        self.w_phase_delay = float(mac_cfg.get("rl_raw_w_phase_delay",  0.5))
        self.w_slot        = float(mac_cfg.get("rl_raw_w_slot",         0.3))

        # EWMA and Q-init
        self._ewma_alpha    = float(mac_cfg.get("rl_raw_ewma_alpha",    0.3))
        self._ewma_pdr_init = float(mac_cfg.get("rl_raw_ewma_pdr_init", 0.5))
        self._q_keep_bias   = float(mac_cfg.get("rl_raw_q_keep_bias",   0.01))
        self._q_init_bias   = float(mac_cfg.get("rl_raw_q_init_bias",   0.02))

        # ------------------------------------------------------------------
        # UCB1 exploration (tabular modes)
        # When use_ucb=True, replaces ε-greedy with Upper Confidence Bound action
        # selection:  UCB(s,a) = Q(s,a) + C * sqrt(ln(N_s+1) / (N_sa+1))
        # Ensures all valid actions are eventually tried without a fixed schedule.
        # ------------------------------------------------------------------
        self._use_ucb = bool(mac_cfg.get("rl_raw_use_ucb", False))
        self._ucb_c   = float(mac_cfg.get("rl_raw_ucb_c",  1.0))

        # ------------------------------------------------------------------
        # Soft target update for DDQN (tau > 0 enables; tau=0 → hard copy)
        # Hard copy: target ← online  every target_update_freq steps
        # Soft copy: target ← τ*online + (1-τ)*target  every step
        # Typical tau: 0.005–0.05.  Smaller = slower but more stable.
        # ------------------------------------------------------------------
        self._target_update_tau = float(mac_cfg.get("rl_raw_target_update_tau", 0.0))

        # ------------------------------------------------------------------
        # N-step returns (n=1 → standard 1-step TD, default)
        # Primary update uses: R_n = Σ_{i=0}^{n-1} γ^i r_{t+i}
        # Bootstrap: R_n + γ^n * max_a Q(s_{t+n})
        # Propagates credit n-times faster; best n: 3–5 for 235-step episodes.
        # ------------------------------------------------------------------
        self._n_step: int = max(1, int(mac_cfg.get("rl_raw_n_step", 1)))
        self._gamma_n: float = self.gamma ** self._n_step
        self._n_step_buf: Deque[Tuple] = collections.deque(maxlen=self._n_step)

        # ------------------------------------------------------------------
        # State bucketing thresholds
        # ------------------------------------------------------------------
        self._col_thresholds   = _parse_thresholds(mac_cfg.get("rl_raw_col_thresholds"),   [0.10, 0.30])
        self._pdr_thresholds   = _parse_thresholds(mac_cfg.get("rl_raw_pdr_thresholds"),   [0.60, 0.30])
        self._sta_thresholds   = _parse_thresholds(mac_cfg.get("rl_raw_sta_thresholds"),   [100.0, 300.0, 600.0])
        self._retry_thresholds = _parse_thresholds(mac_cfg.get("rl_raw_retry_thresholds"), [0.30, 0.60])
        self._slot_thresholds  = _parse_thresholds(mac_cfg.get("rl_raw_slot_thresholds"),  [0.33, 0.67])
        self._phase_thresholds = _parse_thresholds(mac_cfg.get("rl_raw_phase_thresholds"), [2.0,  4.0])

        # Derived bucket-boundary indices — computed from threshold list lengths so
        # that changing rl_raw_pdr_thresholds / rl_raw_retry_thresholds automatically
        # updates all comparisons without touching any other code.
        # "poor PDR"   = the last (highest-index) pdr bucket   → b_p = len(thresholds)
        # "high retry" = the last (highest-index) retry bucket → b_r = len(thresholds)
        # "good PDR"   = the first bucket, always 0 (ascending encoding)
        self._pdr_poor_bucket   = len(self._pdr_thresholds)    # e.g. 2 with default [0.60,0.30]
        self._retry_high_bucket = len(self._retry_thresholds)  # e.g. 2 with default [0.30,0.60]

        # DDQN: probability of updating Q1 (vs Q2) at each step.  Configurable so
        # asymmetric update schedules can be tested; 0.5 is the standard equal split.
        self._ddqn_flip_p = float(mac_cfg.get("rl_raw_ddqn_flip_p", 0.5))

        # ------------------------------------------------------------------
        # Algorithm-specific structures
        # ------------------------------------------------------------------
        seed = mac_cfg.get("rl_raw_seed", None)
        self._rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)  # only seed numpy when explicitly requested

        self._replay = _ReplayBuffer(self.replay_capacity)

        # Tabular Q-table(s)
        self._qtable:  Dict[Tuple[int, ...], List[float]] = {}
        self._qtable2: Dict[Tuple[int, ...], List[float]] = {}   # DDQN table 2

        # Neural-net structures (DQN / DDQN)
        self._norm:      Optional[_RunningNorm] = None
        self._pca:       Optional[_OnlinePCA]   = None
        self._nn_online: Optional[_TinyNet]     = None
        self._nn_target: Optional[_TinyNet]     = None
        self._pca_enabled = False
        self._nn_lr       = float(mac_cfg.get("rl_raw_nn_lr", 0.005))
        self._target_update_freq = int(mac_cfg.get("rl_raw_target_update_freq", 20))

        # PPO-specific hyper-parameters
        self._ppo_clip_eps     = float(mac_cfg.get("rl_raw_ppo_clip_eps",     0.20))
        self._ppo_epochs       = int(  mac_cfg.get("rl_raw_ppo_epochs",          4))
        self._ppo_rollout_len  = int(  mac_cfg.get("rl_raw_ppo_rollout_len",     16))
        self._gae_lambda       = float(mac_cfg.get("rl_raw_gae_lambda",        0.95))
        self._ppo_entropy_coef = float(mac_cfg.get("rl_raw_ppo_entropy_coef",  0.01))
        self._ppo_value_coef   = float(mac_cfg.get("rl_raw_ppo_value_coef",    0.50))

        # PPO networks (actor + critic) and rollout buffer
        self._policy_net: Optional[_PolicyNet] = None
        self._nn_value:   Optional[_TinyNet]   = None
        self._rollout:    Optional[_RolloutBuffer] = None

        # Running context for PPO (log-prob, value, and mask of the action just taken)
        self._prev_log_prob: float              = 0.0
        self._prev_value:    float              = 0.0
        self._prev_mask:     Optional[np.ndarray] = None

        if self._mode in ("dqn", "ddqn"):
            n_hidden = int(mac_cfg.get("rl_raw_nn_hidden", 64))
            self._pca_enabled = bool(mac_cfg.get("rl_raw_use_pca", False))
            self._norm = _RunningNorm(_N_FEATURES)
            if self._pca_enabled:
                n_pca = int(mac_cfg.get("rl_raw_pca_components", 6))
                warmup = int(mac_cfg.get("rl_raw_pca_warmup", 50))
                self._pca = _OnlinePCA(_N_FEATURES, n_pca, warmup)
                n_input = n_pca
            else:
                n_input = _N_FEATURES
            self._nn_online = _TinyNet(n_input, n_hidden, _N_ACTIONS, self._rng)
            if self._mode == "ddqn":
                self._nn_target = self._nn_online.copy()

        if self._mode == "ppo":
            n_hidden = int(mac_cfg.get("rl_raw_nn_hidden", 64))
            self._pca_enabled = bool(mac_cfg.get("rl_raw_use_pca", False))
            self._norm = _RunningNorm(_N_FEATURES)
            if self._pca_enabled:
                n_pca  = int(mac_cfg.get("rl_raw_pca_components", 6))
                warmup = int(mac_cfg.get("rl_raw_pca_warmup", 50))
                self._pca = _OnlinePCA(_N_FEATURES, n_pca, warmup)
                n_input = n_pca
            else:
                n_input = _N_FEATURES
            self._policy_net = _PolicyNet(n_input, n_hidden, _N_ACTIONS, self._rng)
            self._nn_value   = _TinyNet(n_input, n_hidden, 1, self._rng)
            self._rollout    = _RolloutBuffer(self._ppo_rollout_len)

        # EWMA, visit counts, bookkeeping
        self._ewma_pdr       = self._ewma_pdr_init
        self._visit_counts:  Dict[Tuple[int, ...], List[int]] = {}

        self._prev_state:    Any              = None
        self._prev_action:   Optional[int]    = None
        self._prev_stats:    Dict[str, int]   = {}
        self._prev_n_phase:  int              = 1
        self._update_count:  int              = 0
        self._episode_count: int              = 0

        # ------------------------------------------------------------------
        # Q-table persistence
        # ------------------------------------------------------------------
        self._qtable_path: Optional[str] = mac_cfg.get("rl_raw_qtable_path", None)
        if self._qtable_path and os.path.exists(self._qtable_path):
            self._load_qtable()

    # ======================================================================
    # Public API
    # ======================================================================
    def init_configs(self) -> List[RawConfig]:
        aids    = self._effective_aids([])
        max_aid = max(aids) if aids else max(1, int(getattr(self.ctx, "raw_nodes_per_group", 64)))
        return self._make_configs(max_aid)

    def build_dynamic_configs(self, connected_aids: List[int]) -> List[RawConfig]:
        aids = self._effective_aids(connected_aids)
        if not aids:
            self._log("RL_RAW_NO_AIDS", {})
            return []

        n_active = len(aids)
        max_aid  = max(aids)

        # ------------------------------------------------------------------
        # 0. First-call initialisation: physics-based P, N-aware G/S
        # ------------------------------------------------------------------
        if self._update_count == 0:
            max_s    = self._nslot_choices[-1]
            p_init   = max(1, math.ceil(
                n_active / (self._max_useful_slots * self._target_stas_slot
                            * self._phase_load_margin)
            ))
            self._num_phases = min(self._max_phases, p_init)
            n_phase_init     = max(1, n_active // self._num_phases)
            target_g         = max(1, round(n_phase_init / (self._target_stas_slot * max_s)))
            self._groups_idx = self._nearest_idx(self._group_choices, target_g)
            self._nslots_idx = len(self._nslot_choices) - 1
            if not self._slot_override:
                self._slot_us = self.max_slot_us
            self._enforce_budget()
            self._log("RL_RAW_INIT", {
                "mode": self._mode, "n_active": n_active,
                "num_phases": self._num_phases,
                "num_groups": self._group_choices[self._groups_idx],
                "num_slots":  self._nslot_choices[self._nslots_idx],
                "slot_us":    self._slot_us,
            })

        # ------------------------------------------------------------------
        # 1. Observe state
        # ------------------------------------------------------------------
        self._current_phase = self._update_count % self._num_phases
        n_phase = max(1, n_active // self._num_phases)
        in_cold_start = self._update_count < self.cold_start_steps

        # For DQN/DDQN/PPO: sample raw features BEFORE the EWMA update so that
        # the continuous trend feature uses the same pre-update EWMA as bkt.
        if self._mode in ("dqn", "ddqn", "ppo"):
            raw = self._raw_features(n_phase)   # reads self._ewma_pdr (pre-update)
            self._norm.update(raw)
            if self._pca and not self._pca.is_fitted:
                self._pca.update(self._norm.normalize(raw))

        # bkt is always the 8-tuple (needed for action masks in all modes).
        # update_ewma=True exactly once per step regardless of mode.
        bkt = self._observe_state(n_phase, update_ewma=not in_cold_start)

        if self._mode in ("dqn", "ddqn", "ppo"):
            state = self._preprocess(raw)    # normalised (+PCA if fitted)
        else:
            state = bkt    # tabular modes: state IS the bucket tuple

        # ------------------------------------------------------------------
        # 2. Q-update + replay
        # ------------------------------------------------------------------
        if not in_cold_start and self._prev_state is not None and self._prev_action is not None:
            reward, has_traffic = self._compute_reward()

            if self._mode == "ppo":
                if has_traffic:
                    self._rollout.push(
                        self._prev_state, self._prev_action, reward,
                        self._prev_log_prob, self._prev_value,
                        self._prev_mask,   # stored mask for consistent PPO ratio
                    )
                if self._rollout.is_full():
                    self._do_ppo_update(state)
            else:
                # --- N-step return computation ----------------------------
                # Accumulate (s, a, r) in the n-step buffer. When full,
                # derive the n-step return for the oldest transition and use
                # that as the primary target.  Replay always uses 1-step.
                if has_traffic:
                    self._n_step_buf.append(
                        (self._prev_state, self._prev_action, reward))

                if len(self._n_step_buf) == self._n_step:
                    s0, a0, _ = self._n_step_buf[0]
                    R_n = sum(self.gamma**i * r
                              for i, (_, _, r) in enumerate(self._n_step_buf))
                    primary_s, primary_a = s0, a0
                    primary_r, primary_bg = R_n, self._gamma_n
                else:
                    primary_s, primary_a = self._prev_state, self._prev_action
                    primary_r, primary_bg = reward, self.gamma

                if has_traffic:
                    # Store bkt (next-state bucket) and primary_bg (discount)
                    # so replay can apply the correct mask and discount factor.
                    self._replay.push(primary_s, primary_a, primary_r, state,
                                      bkt_next=bkt, gamma_bg=primary_bg)

                # Primary update — guarded by has_traffic so idle beacons
                # (no receptions, no errors) do not drive Q-value updates.
                if has_traffic:
                    self._q_update_dispatch(primary_s, primary_a, primary_r, state,
                                            bg=primary_bg, bkt_next=bkt)

                # Replay updates — use the discount stored per transition.
                if len(self._replay) > 0:
                    n_batches = min(self.updates_per_step,
                                    max(1, len(self._replay) // self.batch_size))
                    for _ in range(n_batches):
                        for (s, a, r, sn, bkt_n, g_n) in self._replay.sample(self.batch_size, self._rng):
                            self._q_update_dispatch(s, a, r, sn,
                                                    bg=g_n if g_n else self.gamma,
                                                    bkt_next=bkt_n)

                # Target-network update for DDQN
                if self._mode == "ddqn":
                    if self._target_update_tau > 0.0:
                        # Soft update every step: θ_target ← τ θ_online + (1-τ) θ_target
                        tau = self._target_update_tau
                        for attr in ("W1", "b1", "W2", "b2"):
                            tgt = getattr(self._nn_target, attr)
                            src = getattr(self._nn_online, attr)
                            np.copyto(tgt, tau * src + (1.0 - tau) * tgt)
                    elif self._update_count % self._target_update_freq == 0:
                        # Hard copy at fixed interval (original behaviour)
                        self._nn_target = self._nn_online.copy()

        # ------------------------------------------------------------------
        # 3. Action selection
        # ------------------------------------------------------------------
        if in_cold_start:
            action = _ACT_KEEP
            if self._mode == "ppo":
                self._prev_log_prob = 0.0
                self._prev_value    = 0.0
                self._prev_mask     = None
        elif self._mode == "ppo":
            action, log_prob, value, act_mask = self._select_action_ppo(state, bkt)
            self._prev_log_prob = log_prob
            self._prev_value    = value
            self._prev_mask     = act_mask
        else:
            action = self._select_action(state, bkt)

        # Track visit counts (tabular modes only)
        if self._mode in ("tabular", "tabular_ddqn"):
            vc = self._visit_counts.setdefault(state, [0] * _N_ACTIONS)
            vc[action] += 1

        # ------------------------------------------------------------------
        # 4. Apply action
        # ------------------------------------------------------------------
        self._apply_action(action)
        self._enforce_budget()

        # ------------------------------------------------------------------
        # 5. Save context
        # ------------------------------------------------------------------
        self._prev_state   = state
        self._prev_action  = action
        self._prev_stats   = self._snapshot_stats()
        self._prev_n_phase = n_phase

        if not in_cold_start and self._mode != "ppo":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._update_count += 1

        configs = self._make_configs(max_aid)
        self._log("RL_RAW_STEP", {
            "mode":             self._mode,
            "update":           self._update_count,
            "phase":            self._current_phase,
            "num_phases":       self._num_phases,
            "action":           action,
            "cold_start":       int(in_cold_start),
            "epsilon":          round(self.epsilon, 4),
            "ewma_pdr":         round(self._ewma_pdr, 3),
            "num_groups":       self._group_choices[self._groups_idx],
            "num_slots":        self._nslot_choices[self._nslots_idx],
            "slot_duration_us": self._slot_us,
            "n_phase":          n_phase,
            "replay_size":      len(self._replay),
        })
        return configs

    def get_policy_snapshot(self) -> Dict[str, Any]:
        return {
            "policy":           f"rl_raw_v8_{self._mode}",
            "update_count":     self._update_count,
            "num_phases":       self._num_phases,
            "current_phase":    self._current_phase,
            "epsilon":          round(self.epsilon, 4),
            "ewma_pdr":         round(self._ewma_pdr, 3),
            "num_groups":       self._group_choices[self._groups_idx],
            "num_slots":        self._nslot_choices[self._nslots_idx],
            "slot_duration_us": self._slot_us,
        }

    # ======================================================================
    # Q-table persistence
    # ======================================================================
    def _load_qtable(self) -> None:
        try:
            with open(self._qtable_path, "rb") as f:
                data = pickle.load(f)
            self._qtable        = data.get("qtable",        {})
            self._qtable2       = data.get("qtable2",       {})
            self._visit_counts  = data.get("visit_counts",  {})
            self._episode_count = data.get("episode_count", 0)
            saved_eps           = data.get("epsilon", self.epsilon)
            self.epsilon        = max(self.epsilon_min, min(self.epsilon, saved_eps))
            self._ewma_pdr      = data.get("ewma_pdr", self._ewma_pdr_init)
            self._num_phases    = data.get("num_phases", 1)
            if self._nn_online and "nn_online" in data:
                self._nn_online.from_dict(data["nn_online"])
            if self._nn_target and "nn_target" in data:
                self._nn_target.from_dict(data["nn_target"])
            if self._policy_net and "policy_net" in data:
                self._policy_net.from_dict(data["policy_net"])
            if self._nn_value and "nn_value" in data:
                self._nn_value.from_dict(data["nn_value"])
            if self._norm and "norm" in data:
                self._norm.from_dict(data["norm"])
            self._n_step_buf.clear()   # discard transitions from previous episode
            self._log("RL_QTABLE_LOADED", {
                "path":    self._qtable_path,
                "mode":    self._mode,
                "episode": self._episode_count,
                "epsilon": round(self.epsilon, 4),
            })
        except Exception as exc:
            self._log("RL_QTABLE_LOAD_ERR", {"err": str(exc)})

    def save_qtable(self) -> None:
        if not self._qtable_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._qtable_path)), exist_ok=True)
            data: Dict[str, Any] = {
                "qtable":        self._qtable,
                "qtable2":       self._qtable2,
                "visit_counts":  self._visit_counts,
                "epsilon":       self.epsilon,
                "ewma_pdr":      self._ewma_pdr,
                "update_count":  self._update_count,
                "episode_count": self._episode_count + 1,
                "num_phases":    self._num_phases,
            }
            if self._nn_online:
                data["nn_online"] = self._nn_online.to_dict()
            if self._nn_target:
                data["nn_target"] = self._nn_target.to_dict()
            if self._policy_net:
                data["policy_net"] = self._policy_net.to_dict()
            if self._nn_value:
                data["nn_value"] = self._nn_value.to_dict()
            if self._norm:
                data["norm"] = self._norm.to_dict()
            with open(self._qtable_path, "wb") as f:
                pickle.dump(data, f)
            self._episode_count += 1
        except Exception as exc:
            self._log("RL_QTABLE_SAVE_ERR", {"err": str(exc)})

    # ======================================================================
    # Phase helpers
    # ======================================================================
    def _phase_aid_range(self, max_aid: int) -> Tuple[int, int]:
        span = max(1, int(math.ceil(max_aid / self._num_phases)))
        sa   = self._current_phase * span + 1
        ea   = min(max_aid, sa + span - 1)
        return sa, ea

    # ======================================================================
    # State observation
    # ======================================================================
    def _snapshot_delta(self) -> Tuple[int, int, int, int]:
        """Return (delta_rx, delta_err, delta_tx, delta_rtr) since prev beacon."""
        stats = self._snapshot_stats()
        prev  = self._prev_stats
        return (
            max(0, stats.get("rx_unicast",      0) - prev.get("rx_unicast",      0)),
            max(0, stats.get("rx_crc_error",    0) - prev.get("rx_crc_error",    0)),
            max(1, stats.get("mac_tx_attempts", 1) - prev.get("mac_tx_attempts", 1)),
            max(0, stats.get("mac_retries",     0) - prev.get("mac_retries",     0)),
        )

    def _raw_features(self, n_phase: int) -> np.ndarray:
        """8-dim continuous feature vector for neural-net modes."""
        d_rx, d_err, d_tx, d_rtr = self._snapshot_delta()
        total     = d_rx + d_err
        col_rate  = d_err / total if total > 0 else 0.0
        expected  = max(1.0, n_phase * self._tx_fraction)
        pdr_hat   = min(1.0, d_rx / expected)
        trend     = 0.0 if pdr_hat >= self._ewma_pdr else 1.0
        retry     = min(1.0, d_rtr / d_tx)
        total_n   = n_phase * self._num_phases
        n_norm    = min(1.0, total_n / max(1.0, max(self._sta_thresholds)))
        g_frac    = self._groups_idx / max(1, len(self._group_choices) - 1)
        slot_range = max(1, self.max_slot_us - self.min_slot_us)
        slot_frac  = (self._slot_us - self.min_slot_us) / slot_range
        phase_norm = (self._num_phases - 1) / max(1, self._max_phases - 1)
        return np.array([col_rate, pdr_hat, n_norm, g_frac, slot_frac,
                         trend, retry, phase_norm], dtype=np.float64)

    def _preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Normalise + optional PCA transform (for DQN/DDQN/PPO).

        When PCA is enabled but not yet fitted, _OnlinePCA.transform() returns
        x[:n_components] (truncation fallback), keeping the output dimension
        consistent with the network's expected input size throughout warmup.
        """
        x = self._norm.normalize(raw)
        if self._pca:
            x = self._pca.transform(x)   # always call; fallback handles pre-fit
        return x

    def _observe_state(
        self, n_phase: int, update_ewma: bool = True
    ) -> Tuple[int, int, int, int, int, int, int, int]:
        """8-tuple bucketed state for tabular modes."""
        d_rx, d_err, d_tx, d_rtr = self._snapshot_delta()

        total    = d_rx + d_err
        col_rate = d_err / total if total > 0 else 0.0
        col_bucket = self._bucket(col_rate, self._col_thresholds, descending=True)

        expected  = max(1.0, n_phase * self._tx_fraction)
        pdr_hat   = min(1.0, d_rx / expected)
        # descending=False → b_p=0 good (≥60 %), b_p=1 moderate (30–60 %), b_p=2 poor (<30 %)
        pdr_bucket = self._bucket(pdr_hat, self._pdr_thresholds, descending=False)

        trend = 0 if pdr_hat >= self._ewma_pdr else 1
        if update_ewma:
            self._ewma_pdr = (self._ewma_alpha * pdr_hat
                              + (1.0 - self._ewma_alpha) * self._ewma_pdr)

        total_n    = n_phase * self._num_phases
        sta_bucket = self._bucket(float(total_n), self._sta_thresholds, descending=True)

        slot_range  = max(1, self.max_slot_us - self.min_slot_us)
        slot_frac   = (self._slot_us - self.min_slot_us) / slot_range
        slot_bucket = self._bucket(slot_frac, self._slot_thresholds, descending=True)

        retry_rate   = min(1.0, d_rtr / d_tx)
        retry_bucket = self._bucket(retry_rate, self._retry_thresholds, descending=True)

        phase_bucket = self._bucket(float(self._num_phases), self._phase_thresholds, descending=True)

        return (col_bucket, pdr_bucket, sta_bucket,
                self._groups_idx, slot_bucket, trend,
                retry_bucket, phase_bucket)

    # ======================================================================
    # Reward
    # ======================================================================
    def _compute_reward(self) -> Tuple[float, bool]:
        """6-term reward: PDR, collision, overhead, retry delay, phase delay, slot adequacy."""
        d_rx, d_err, d_tx, d_rtr = self._snapshot_delta()
        has_traffic = (d_rx > 0) or (d_err > 0)

        expected_tx = max(1.0, self._prev_n_phase * self._tx_fraction)
        pdr_hat     = min(1.0, d_rx / expected_tx)
        total       = d_rx + d_err
        col_rate    = d_err / total if total > 0 else 0.0

        G        = self._group_choices[self._groups_idx]
        S        = self._nslot_choices[self._nslots_idx]
        overhead = min(1.0, (G * S * self._slot_us) / max(1, self._beacon_budget_us))

        retry_rate = min(1.0, d_rtr / d_tx)

        # Phase cycling delay: 0 at P=1, grows with P
        phase_wait = min(1.0,
            (self._num_phases - 1) * self._beacon_interval_s
            / max(1e-9, self._traffic_interval_s)
        )

        # Slot adequacy deficit: 0 when slot ≥ T_min
        slot_deficit = max(0.0, 1.0 - self._slot_us / max(1, self._phys_min_us))

        reward = float(
              self.w_pdr         * pdr_hat
            - self.w_collision   * col_rate
            - self.w_overhead    * overhead
            - self.w_retry       * retry_rate
            - self.w_phase_delay * phase_wait
            - self.w_slot        * slot_deficit
        )
        return reward, has_traffic

    # ======================================================================
    # Q-value access (tabular)
    # ======================================================================
    def _q_values_table(self, state: Tuple[int, ...],
                        table: Dict) -> List[float]:
        if state not in table:
            q = [0.0] * _N_ACTIONS
            q[_ACT_KEEP] = self._q_keep_bias
            # State-context priors (configurable bias magnitude)
            _, pdr_bucket, _, _, _, _, retry_bucket, phase_bucket = state
            b = self._q_init_bias
            if pdr_bucket == self._pdr_poor_bucket:       # poorest PDR bucket
                q[_ACT_PHASES_INC] += b;  q[_ACT_GROUPS_INC] += b;  q[_ACT_NSLOTS_INC] += b
            if retry_bucket == self._retry_high_bucket:   # highest retry bucket
                q[_ACT_SLOT_INC]   += b;  q[_ACT_PHASES_INC] += b
            if phase_bucket >= 1 and pdr_bucket == 0:     # multi-phase AND good PDR
                q[_ACT_PHASES_DEC] += b
            table[state] = q
        return table[state]

    def _q_values(self, state: Tuple[int, ...]) -> List[float]:
        return self._q_values_table(state, self._qtable)

    # ======================================================================
    # Q-update dispatch
    # ======================================================================
    def _q_update_dispatch(self, s, a: int, r: float, s_next,
                           bg: Optional[float] = None,
                           bkt_next=None) -> None:
        """Dispatch Q-update.

        bg:       bootstrap discount (γ^n for n-step; None → γ).
        bkt_next: 8-tuple bucket state for s_next.  Used to mask invalid
                  actions from the Bellman max / argmax.  For tabular modes
                  s_next IS the bucket tuple; for neural modes bkt_next must
                  be supplied explicitly.
        """
        g = bg if bg is not None else self.gamma
        # For tabular modes the next state is already the bucket tuple.
        bkt_n = s_next if self._mode in ("tabular", "tabular_ddqn") else bkt_next
        if self._mode == "tabular":
            self._q_update_tabular(s, a, r, s_next, g, self._qtable, self._qtable, bkt_n)
        elif self._mode == "tabular_ddqn":
            self._q_update_double_tabular(s, a, r, s_next, g, bkt_n)
        elif self._mode == "dqn":
            self._q_update_dqn(s, a, r, s_next, g, bkt_n)
        elif self._mode == "ddqn":
            self._q_update_ddqn(s, a, r, s_next, g, bkt_n)

    def _q_update_tabular(self, s, a: int, r: float, s_next, g: float,
                          update_table: Dict, eval_table: Dict,
                          bkt_next=None) -> None:
        """Standard Q-learning TD update; g is the bootstrap discount factor."""
        q      = self._q_values_table(s,      update_table)
        q_next = self._q_values_table(s_next, eval_table)
        # Mask invalid next-state actions from the Bellman max.
        excl_next = self._action_mask(bkt_next) if bkt_next is not None else set()
        valid_next = [v for i, v in enumerate(q_next) if i not in excl_next]
        td     = r + g * max(valid_next) - q[a]
        td     = max(-self._td_clip, min(self._td_clip, td))
        q[a]  += self.alpha * td

    def _q_update_double_tabular(self, s, a: int, r: float, s_next,
                                  g: float, bkt_next=None) -> None:
        """Double Q-learning: alternate update/eval tables; g is bootstrap discount."""
        excl_next = self._action_mask(bkt_next) if bkt_next is not None else set()
        valid_idx = [i for i in range(_N_ACTIONS) if i not in excl_next]
        if self._rng.random() < self._ddqn_flip_p:
            q1      = self._q_values_table(s,      self._qtable)
            q1_next = self._q_values_table(s_next, self._qtable)
            q2_next = self._q_values_table(s_next, self._qtable2)
            best_a  = max(valid_idx, key=lambda i: q1_next[i])
            target  = r + g * q2_next[best_a]
            td      = max(-self._td_clip, min(self._td_clip, target - q1[a]))
            q1[a]  += self.alpha * td
        else:
            q2      = self._q_values_table(s,      self._qtable2)
            q2_next = self._q_values_table(s_next, self._qtable2)
            q1_next = self._q_values_table(s_next, self._qtable)
            best_a  = max(valid_idx, key=lambda i: q2_next[i])
            target  = r + g * q1_next[best_a]
            td      = max(-self._td_clip, min(self._td_clip, target - q2[a]))
            q2[a]  += self.alpha * td

    def _q_update_dqn(self, s: np.ndarray, a: int,
                      r: float, s_next: np.ndarray, g: float,
                      bkt_next=None) -> None:
        """Standard DQN; g is bootstrap discount (γ^n for n-step)."""
        q_next = self._nn_online.forward(s_next).copy()
        if bkt_next is not None:
            for ea in self._action_mask(bkt_next):
                q_next[ea] = -float("inf")
        target = r + g * float(np.max(q_next))
        q_now  = self._nn_online.forward(s)
        td     = max(-self._td_clip, min(self._td_clip, target - q_now[a]))
        self._nn_online.update(s, a, float(q_now[a] + td), self._nn_lr)

    def _q_update_ddqn(self, s: np.ndarray, a: int,
                       r: float, s_next: np.ndarray, g: float,
                       bkt_next=None) -> None:
        """DDQN: online selects action, target evaluates; g is bootstrap discount."""
        q_online_next = self._nn_online.forward(s_next).copy()
        if bkt_next is not None:
            for ea in self._action_mask(bkt_next):
                q_online_next[ea] = -float("inf")
        best_a        = int(np.argmax(q_online_next))
        q_target_next = self._nn_target.forward(s_next)
        target        = r + g * float(q_target_next[best_a])
        q_now         = self._nn_online.forward(s)
        td            = max(-self._td_clip, min(self._td_clip, target - q_now[a]))
        self._nn_online.update(s, a, float(q_now[a] + td), self._nn_lr)

    # ======================================================================
    # Action selection
    # ======================================================================
    def _action_mask(self, bkt: Tuple[int, ...]) -> set:
        """Return the set of action indices excluded from random exploration.

        Masking rationale:
        • NSlotsDec    — always excluded: silently worsens per-slot contention.
        • PhasesDec    — excluded when retries are elevated (b_r > 0); reducing
                         phases when contention is already high extends inter-
                         beacon waiting and worsens collisions.
        • PhasesInc    — excluded when phases are already elevated AND PDR is
                         good (b_P ≥ 1, b_p = 0); no benefit from more delay.
        • GroupsDec, SlotDec, NSlotsDec — excluded when PDR is already poor
                         (b_p = pdr_poor_bucket); reducing groups or shrinking
                         slots increases per-group contention and makes recovery
                         even harder.  These actions are still available on the
                         greedy path if the Q-table learns they help.
        """
        _, pdr_bucket, _, _, _, _, retry_bucket, phase_bucket = bkt
        excluded: set = {_ACT_NSLOTS_DEC}
        if retry_bucket > 0:                                     # elevated retries
            excluded.add(_ACT_PHASES_DEC)
        if phase_bucket >= 1 and pdr_bucket == 0:                # multi-phase + good PDR
            excluded.add(_ACT_PHASES_INC)
        if pdr_bucket == self._pdr_poor_bucket:                  # PDR already poor
            excluded.add(_ACT_GROUPS_DEC)
            excluded.add(_ACT_SLOT_DEC)
        return excluded

    def _select_action(self, state, bkt: Tuple[int, ...]) -> int:
        excluded = self._action_mask(bkt)

        # --- UCB1 (tabular modes, when enabled) ----------------------------
        # Replaces ε-greedy entirely: selects argmax UCB(s,a) over valid actions.
        # UCB(s,a) = Q(s,a) + C * sqrt(ln(N_s + 1) / (N_sa + 1))
        if self._use_ucb and self._mode in ("tabular", "tabular_ddqn"):
            vc      = self._visit_counts.get(state, [0] * _N_ACTIONS)
            n_total = sum(vc) + 1
            log_n   = math.log(n_total)
            if self._mode == "tabular_ddqn":
                q1 = self._q_values_table(state, self._qtable)
                q2 = self._q_values_table(state, self._qtable2)
                q  = [(q1[a] + q2[a]) / 2.0 for a in range(_N_ACTIONS)]
            else:
                q = self._q_values(state)
            ucb = [
                (q[a] + self._ucb_c * math.sqrt(log_n / (vc[a] + 1)))
                if a not in excluded else -float("inf")
                for a in range(_N_ACTIONS)
            ]
            return int(max(range(_N_ACTIONS), key=lambda a: ucb[a]))

        # --- ε-greedy (all modes) ------------------------------------------
        if self._rng.random() < self.epsilon:
            candidate = [a for a in range(_N_ACTIONS) if a not in excluded]
            if self._mode in ("tabular", "tabular_ddqn"):
                vc   = self._visit_counts.get(state, [0] * _N_ACTIONS)
                unvx = [a for a in candidate if vc[a] == 0]
                return self._rng.choice(unvx if unvx else candidate)
            return self._rng.choice(candidate)

        # --- Greedy ----------------------------------------------------------
        # Mask is applied to ALL greedy paths so forbidden actions are never
        # chosen during exploitation, not just during random exploration.
        valid = [a for a in range(_N_ACTIONS) if a not in excluded]
        if self._mode in ("dqn", "ddqn"):
            q = self._nn_online.forward(state).copy()
            for a in excluded:
                q[a] = -float("inf")
            return int(np.argmax(q))
        elif self._mode == "tabular_ddqn":
            q1    = self._q_values_table(state, self._qtable)
            q2    = self._q_values_table(state, self._qtable2)
            q_avg = [q1[i] + q2[i] for i in range(_N_ACTIONS)]
            best  = max(q_avg[i] for i in valid)
            ties  = [i for i in valid if q_avg[i] == best]
            return self._rng.choice(ties)
        else:
            q    = self._q_values(state)
            best = max(q[i] for i in valid)
            ties = [i for i in valid if q[i] == best]
            return self._rng.choice(ties)

    # ======================================================================
    # PPO-specific methods
    # ======================================================================
    def _select_action_ppo(
        self, state: np.ndarray, bkt: Tuple[int, ...]
    ) -> Tuple[int, float, float, np.ndarray]:
        """Sample action from policy network; return (action, log_prob, value, mask).

        The boolean mask is returned so the caller can store it in the rollout
        buffer and pass it to ppo_update() for a consistent ratio estimate.
        """
        excluded = self._action_mask(bkt)
        mask = np.ones(_N_ACTIONS, dtype=bool)
        for idx in excluded:
            mask[idx] = False

        action, log_prob = self._policy_net.sample(state, mask, self._rng)
        value = float(self._nn_value.forward(state)[0])
        return action, log_prob, value, mask

    def _do_ppo_update(self, next_state: np.ndarray) -> None:
        """Compute GAE on the collected rollout and run PPO epochs."""
        transitions = self._rollout.get_and_clear()
        if not transitions:
            return

        states    = [t[0] for t in transitions]
        actions   = [t[1] for t in transitions]
        rewards   = [t[2] for t in transitions]
        log_probs = [t[3] for t in transitions]
        values    = [t[4] for t in transitions]
        masks     = [t[5] for t in transitions]   # boolean mask used at sampling

        next_val  = float(self._nn_value.forward(next_state)[0])
        advantages, returns = _compute_gae(
            rewards, values, next_val, self.gamma, self._gae_lambda
        )

        # Normalise advantages for stable gradient magnitudes
        adv = np.array(advantages, dtype=np.float64)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = list(range(len(transitions)))
        for _ in range(self._ppo_epochs):
            self._rng.shuffle(indices)
            for i in indices:
                # Pass the stored mask so new log-prob is computed under the
                # same constrained distribution as the old log-prob.
                self._policy_net.ppo_update(
                    states[i], actions[i], float(adv[i]),
                    log_probs[i], self._nn_lr,
                    self._ppo_clip_eps, self._ppo_entropy_coef,
                    mask=masks[i],
                )
                # Value (critic) update — MSE on bootstrapped returns
                v_pred = float(self._nn_value.forward(states[i])[0])
                td = max(-self._td_clip,
                         min(self._td_clip, float(returns[i]) - v_pred))
                self._nn_value.update(
                    states[i], 0, v_pred + td,
                    self._nn_lr * self._ppo_value_coef,
                )

    # ======================================================================
    # Action application
    # ======================================================================
    def _apply_action(self, action: int) -> None:
        if action == _ACT_SLOT_INC:
            self._slot_us    = self._quantize(self._slot_us + self.step_us)
        elif action == _ACT_SLOT_DEC:
            self._slot_us    = self._quantize(self._slot_us - self.step_us)
        elif action == _ACT_GROUPS_INC:
            self._groups_idx = min(len(self._group_choices) - 1, self._groups_idx + 1)
        elif action == _ACT_GROUPS_DEC:
            self._groups_idx = max(0, self._groups_idx - 1)
        elif action == _ACT_NSLOTS_INC:
            self._nslots_idx = min(len(self._nslot_choices) - 1, self._nslots_idx + 1)
        elif action == _ACT_NSLOTS_DEC:
            self._nslots_idx = max(0, self._nslots_idx - 1)
        elif action == _ACT_PHASES_INC:
            self._num_phases = min(self._max_phases, self._num_phases + 1)
        elif action == _ACT_PHASES_DEC:
            self._num_phases = max(1, self._num_phases - 1)

    def _enforce_budget(self) -> None:
        max_iters = (
            (self.max_slot_us - self.min_slot_us) // max(1, self.step_us)
            + len(self._nslot_choices) + len(self._group_choices) + 1
        )
        for _ in range(max_iters):
            G = self._group_choices[self._groups_idx]
            S = self._nslot_choices[self._nslots_idx]
            if G * S * self._slot_us <= self._beacon_budget_us:
                break
            if self._slot_us > self.min_slot_us:
                self._slot_us = self._quantize(self._slot_us - self.step_us)
            elif self._nslots_idx > 0:
                self._nslots_idx -= 1
            elif self._groups_idx > 0:
                self._groups_idx -= 1
            else:
                break

    # ======================================================================
    # RawConfig construction
    # ======================================================================
    def _make_configs(self, max_aid: int) -> List[RawConfig]:
        G        = self._group_choices[self._groups_idx]
        S        = self._nslot_choices[self._nslots_idx]
        slot_us  = max(MORSE_RAW_MIN_SLOT_DURATION_US, self._slot_us)
        start_us = int(self.ctx.raw_start_time_us)
        per_grp  = S * slot_us

        if self._num_phases > 1:
            sa, ea = self._phase_aid_range(max_aid)
            if sa > ea or sa > max_aid:
                return self._fallback_config(max_aid)
            phase_span = ea - sa + 1
            span = max(1, int(math.ceil(phase_span / G)))
            configs: List[RawConfig] = []
            for g in range(G):
                g_sa = sa + g * span
                g_ea = min(ea, g_sa + span - 1)
                if g_sa > ea:
                    break
                configs.append(RawConfig(
                    id=g + 1, raw_type=self.ctx.raw_type,
                    start_aid=g_sa, end_aid=g_ea,
                    start_time_us=start_us + g * per_grp,
                    slot_definition=RawSlotDefinition(
                        num_slots=S, slot_duration_us=slot_us,
                        cross_slot_boundary=self.ctx.raw_cross_slot),
                    beacon_spreading=RawBeaconSpreading(),
                    periodic=RawPeriodic(), enabled=True,
                ))
            return configs if configs else self._fallback_config(max_aid)

        span    = max(1, int(math.ceil(max_aid / G)))
        configs = []
        for g in range(G):
            sa_all = 1 + g * span
            ea_all = min(max_aid, sa_all + span - 1)
            if sa_all > max_aid:
                break
            configs.append(RawConfig(
                id=g + 1, raw_type=self.ctx.raw_type,
                start_aid=sa_all, end_aid=ea_all,
                start_time_us=start_us + g * per_grp,
                slot_definition=RawSlotDefinition(
                    num_slots=S, slot_duration_us=slot_us,
                    cross_slot_boundary=self.ctx.raw_cross_slot),
                beacon_spreading=RawBeaconSpreading(),
                periodic=RawPeriodic(), enabled=True,
            ))
        return configs if configs else self._fallback_config(max_aid)

    def _fallback_config(self, max_aid: int) -> List[RawConfig]:
        slot_us = max(MORSE_RAW_MIN_SLOT_DURATION_US, self.min_slot_us)
        return [RawConfig(
            id=1, raw_type=self.ctx.raw_type,
            start_aid=1, end_aid=max(1, max_aid),
            start_time_us=int(self.ctx.raw_start_time_us),
            slot_definition=RawSlotDefinition(
                num_slots=1, slot_duration_us=slot_us,
                cross_slot_boundary=self.ctx.raw_cross_slot),
            beacon_spreading=RawBeaconSpreading(),
            periodic=RawPeriodic(), enabled=True,
        )]

    # ======================================================================
    # Helpers
    # ======================================================================
    def _quantize(self, slot_us: int) -> int:
        step = max(1, self.step_us)
        q    = int(round(slot_us / step)) * step
        # Snap onto the RAW slot-duration cslot grid (500 + 120*cslot us, per
        # IEEE 802.11ah Sec 9.4.2.200) so raw.py's us_to_cslot()/cslot_to_us()
        # round-trip (which floors onto this grid) returns q unchanged.
        q = _snap_grid_up(q)
        return max(self.min_slot_us, min(self.max_slot_us, q))

    @staticmethod
    def _bucket(value: float, thresholds: List[float], descending: bool) -> int:
        if descending:
            for i, t in enumerate(thresholds):
                if value < t:
                    return i
            return len(thresholds)
        else:
            for i, t in enumerate(thresholds):
                if value >= t:
                    return i
            return len(thresholds)

    def _snapshot_stats(self) -> Dict[str, int]:
        try:
            return {k: int(v) for k, v in (self.ctx._stats or {}).items()}
        except Exception:
            return {}

    def _effective_aids(self, aids: List[int]) -> List[int]:
        out = sorted(int(a) for a in aids if int(a) > 0)
        if out:
            return out
        try:
            assoc = getattr(self.ctx, "_associated_stas", None)
            if isinstance(assoc, dict):
                out = sorted(int(v) for v in assoc.values() if int(v) > 0)
                if out:
                    return out
        except Exception:
            pass
        try:
            nodes = getattr(self.ctx.sim, "nodes", {})
            out   = sorted(int(k) for k in nodes if int(k) > 0)
            if out:
                return out
        except Exception:
            pass
        return []

    @staticmethod
    def _nearest_idx(choices: List[int], value: int) -> int:
        best_d, best_i = float("inf"), 0
        for i, c in enumerate(choices):
            d = abs(c - value)
            if d < best_d or (d == best_d and c > choices[best_i]):
                best_d, best_i = d, i
        return best_i
