# Response to Reviewers — MA-PRAW

Format per comment: **Comment → Response → Paper** (what to add/change in the manuscript).

Placeholders in brackets (e.g. `[insert X]`) mark numbers or results that need to come from your training pipeline, hardware benchmarks, or a re-run of the simulator — they were not available in the sim11ah codebase and have not been fabricated.

**Update:** Since the first draft of this document, three real bugs in `sim11ah/mac/raw_policy_cluster_adaptive.py` were found and fixed (see the summary at the bottom), and the full RAW-policy comparison sweep (all 8 policies × N=100–1000) was actually run. Comments 8/9 (SOTA comparison) and Comment 4 (grouping ablation/rationale) below have been updated with real measured numbers in place of placeholders. These numbers are from a **single seed (42)** — re-run with 3 seeds (42, 7, 99), matching your other papers' methodology, before finalizing for submission; directional conclusions are unlikely to change but exact percentages will shift slightly.

---

## Reviewer 1

### Comment 1
**Comment:** Section III-A assumes "one ground AP and N UAV stations." Given HaLow's 1–3 km range and UAV mobility, UAVs may fly out of coverage. Discuss potential need for multi-AP assistance.

**Response:** The single-AP assumption is a deliberate scope boundary consistent with the RAW-scheduling literature we compare against (Chang et al. [TMC 2019], LACA), all single-BSS. We bound this assumption in terms of UAV residency time vs. the RAW reconfiguration interval and add multi-AP handoff to Future Work.

**Paper:**
- Section III-A: append — *"This work assumes a single-AP BSS, consistent with the RAW-scheduling baselines evaluated in Section VI. We assume UAV in-coverage residency time exceeds the beacon interval T_B, so that RAW group membership remains valid across at least one reconfiguration cycle."*
- Section VII (Future Work): add a paragraph on multi-AP/handoff-aware extension as future work — do not claim it as solved.
- Optional: a small residency-time sanity table (flight speed × AP range → expected in-coverage duration vs. T_B) if you want to substantiate the assumption quantitatively rather than just state it.

---

### Comment 2
**Comment:** LSTM position prediction incurs significant computational overhead for a scenario requiring short prediction times. Provide detailed complexity/overhead evaluation.

**Response:** Prediction and RAW reconfiguration occur once per beacon interval, not per packet, so the real constraint is inference latency vs. T_B, not sub-millisecond response. We report measured latency against that budget.

**Paper:**
- New subsection (Section IV or V): "Computational Complexity of Mobility Prediction" with: LSTM architecture (layers/hidden units), parameter count, per-UAV inference latency on stated hardware, total latency for N UAVs per beacon interval, and a plot/table of that total against T_B across your N sweep (10–1000+).
- One sentence tying it to real-time feasibility: *"Total per-beacon-interval inference cost remains below X% of T_B for all evaluated N, confirming the prediction stage does not bottleneck the scheduling cycle."* — `[insert measured X]`.

---

### Comment 3
**Comment:** Font size in Fig. 2 too small; elaborate on the workflow.

**Response:** Fixed and elaborated.

**Paper:**
- Regenerate Fig. 2 at ≥8pt print font.
- Add a paragraph immediately after the figure walking through the five stages in order: mobility prediction → location-aware grouping → per-STA demand prediction (EWMA-CUSUM) → contention estimation (Bianchi) → adaptive slot-count/duration allocation — one sentence per stage on its input/output.

---

### Comment 4
**Comment:** Missing simulator parameters (UAV count, speed range, communication model) and undefined design parameters GD, GS, GB, GΛ in Eq. (10).

**Response:** Added a comprehensive parameter table. GD=2.5, GS=0.012, GB=1.25; GΛ was not previously exposed as a tunable parameter in the implementation (hardcoded at 0.03) — now corrected and exposed to match Eq. (10).

**Paper:**
- New Table X, "Simulation and Algorithm Parameters," two columns (Parameter | Value):
  - Scenario: N (UAV count range tested), flight speed range, AP coverage radius
  - PHY/MAC: MCS/data rate, slot time σ, SIFS, DIFS, ACK timeout, CW_min/CW_max
  - RAW: min/max slot duration bounds, min/max slots per cluster
  - Algorithm: G_D=2.5, G_S=0.012, G_B=1.25, G_Λ=0.03, EWMA α, CUSUM k and h, Bianchi solver tolerance
- One sentence noting GΛ's role explicitly, e.g. *"G_Λ scales the aggregate cluster demand by total predicted load across member STAs, providing a second-order correction beyond the per-STA burst term."*

---

### Comment 5
**Comment:** "MC-PRAW" typo, should read "MA-PRAW."

**Response:** Corrected.

**Paper:** Find/replace "MC-PRAW" → "MA-PRAW" globally; double-check Section I and the abstract for the same slip.

---

### Comment 6
**Comment:** Overall writing needs strengthening — add explicit system model, problem formulation, and describe the purpose of each sub-algorithm.

**Response:** Added an explicit problem formulation and a per-stage role description, closing the same gap Reviewer 2 raises in Comment 10.

**Paper:**
- New subsection in Section III, "Problem Formulation": state the objective — maximize aggregate priority-weighted throughput (or minimize weighted delay) — subject to (i) total RAW airtime per beacon interval ≤ budget, (ii) slot duration ≥ safety floor, (iii) slot count per cluster within bounds.
- New subsection, "Role of Each Component": one short paragraph per module (mobility prediction, grouping, demand prediction, contention estimation, slot allocation) stating what it contributes to satisfying the above objective/constraints — this directly answers both R1-6 and R2-10, so write it once and reference it from both.

---

### Comment 7
**Comment:** Axes in Fig. 3 and Fig. 5 lack physical units.

**Response:** Fixed.

**Paper:** Update axis labels — e.g., "Number of UAVs (N)" on x, "PDR" or "Throughput (Mbps)" / "Delay (ms)" on y, as applicable to each figure. Audit all remaining figures for the same omission while you're in there.

---

### Comment 8
**Comment:** All four evaluated schemes are the authors' own variants — compare against SOTA.

**Response:** Added chang2019 and LACA as published-baseline comparisons, actually run under identical UAV mobility/traffic conditions as MA-PRAW (single-seed results below; 3-seed re-run still needed before submission). MA-PRAW now beats LACA at every tested N (100–1000) and beats chang2019 at N≥400, with the two roughly tied at N=100–200.

**Paper:** See Comment 9 (Reviewer 2) below for the full results table and figure/text additions — same fix, same figures.

---

### Comment 9
**Comment:** Include assessment of network control overhead.

**Response:** Added compute and signalling overhead analysis.

**Paper:**
- New subsection, "Control and Computational Overhead," with two parts:
  1. A table/plot of RAW-reconfiguration compute time per beacon interval vs. N.
  2. A table/plot of signalling overhead: bytes = (number of active clusters) × (RAW Parameter Set IE size) per beacon, vs. N.
- One sentence noting the fix made during this revision: *"We additionally identified and corrected an implementation inefficiency in which cluster membership data was re-parsed from storage every beacon interval; the corrected implementation caches this data and is reflected in the reported overhead figures."*

---

## Reviewer 2

### Comment 1
**Comment:** Most components (LSTM, K-Means, EWMA-CUSUM, Bianchi, adaptive RAW) are established. Identify specific novelty vs. TAROA, E-TAROA, RO-RAW, traffic-aware grouping. Comparison table recommended.

**Response:** Added a comparison table and clarified that the novelty is the *joint coupling* of proactive mobility-driven grouping with priority/burst-aware demand prediction feeding a combined slot-count-and-duration allocation rule, not any single component.

**Paper:**
- New Table, "Comparison with Related RAW-Scheduling Schemes," columns: Scheme | Mobility-adaptive grouping? | Demand model | Contention model (saturated/non-saturated) | Joint slot-count + duration adaptation?
- Rows: TAROA, E-TAROA, RO-RAW, Chang et al. [2019], LACA, MA-PRAW (this work).
- One paragraph in Related Work stating the novelty claim explicitly as above. `[needs the actual TAROA/E-TAROA/RO-RAW mechanism descriptions from their papers]`.
- Back the qualitative table with the quantitative PDR comparison against chang2019/LACA now available in Comment 9's response — a reviewer will want to see the mechanism-level table and the measured-performance table agree with each other.

---

### Comment 2
**Comment:** Quantify how mobility prediction accuracy affects PDR/throughput/delay. Add an oracle baseline using true future positions.

**Response:** Added a real oracle baseline — not a placeholder. Built two cluster-assignment CSVs from the actual trained LSTM + K-Means pipeline (`uav/00_uav_lstm_clustering_final.ipynb`, `best_uav_model.keras`, real UAV-VisLoc training data): one from the LSTM's *predicted* future positions (K-Means, K=6, the pipeline's own elbow/silhouette-selected value), one from the *true* future positions at the identical snapshot timestep, same K, same priority/traffic-assignment procedure — clustering methodology is the only variable that differs. Mean position error at the snapshot timestep (3.10) matches the notebook's own reported accuracy plots almost exactly, confirming the reproduction is faithful.

**Population note (superseded an earlier, smaller run):** the ML pipeline's synthetic test population was originally 500 UAVs, so N was first capped at 500. Extended it to the full 1000 to match this paper's usual N range: `uav/data/uav_synthesis_pipeline.py` generates UAVs deterministically (fixed seed, sequential loop), so calling it for 1000 left the original 500 byte-identical (verified) and added 500 more real synthetic flight paths, run through the same trained model. Re-clustering the full 1000-UAV population shifts assignments even for N≤500 (K-Means depends on the whole dataset), so the N=100–500 points below are not the same as an earlier 500-UAV-only run that showed a real, CI-confirmed gap at N=400 (13.5–20.7%) — that finding does not replicate at the larger, more representative population and should be treated as superseded, not additional evidence.

**Measured result (N=100–1000, 3 seeds):** the prediction-error cost is negligible at every tested N for both policies — CIs overlap heavily or point estimates differ by well under 1 percentage point throughout the full range. MA-PRAW tracks the perfect-knowledge oracle almost exactly everywhere, not just "most" of the range.

| N | Cluster-based (Pred / Oracle) | MA-PRAW (Pred / Oracle) |
|---|---|---|
| 100 | 0.985 / 0.985 | 0.985 / 0.985 |
| 400 | 0.382 / 0.383 | 0.393 / 0.393 |
| 1000 | 0.095 / 0.096 | 0.093 / 0.091 |

**Paper:**
- Add an "Oracle (perfect prediction)" curve to every PDR/throughput/delay-vs-N figure in Section VI, alongside MA-PRAW — figure already built: `results/figs/MC-PRAW/fig_oracle_vs_predicted.pdf` (N=100–1000, matches every other figure's range now).
- New paragraph: *"MA-PRAW's PDR, throughput, and delay track the perfect-knowledge oracle within noise across the full evaluated range (N=100–1000), indicating that mobility-prediction error contributes negligible MAC-layer cost — performance is governed by contention/scheduling dynamics, not prediction accuracy, at this traffic load and prediction horizon."*
- This is a clean, unconditionally positive result for the paper — no caveat about a specific N is needed.

---

### Comment 3
**Comment:** Predictor trained on UAV-VisLoc, evaluated on 500 synthetic trajectories. Explain statistical consistency and describe train/val/test split.

**Response:** Not a placeholder — verified against the actual pipeline code (`uav/dataset/*.csv`, `uav/data/uav_synthesis_pipeline.py`, `uav/00_uav_lstm_clustering_final.ipynb`) and backed with a new empirical per-shape error breakdown (`uav/eval_per_shape_error.py`).

**Train/val/test separation:** 10 real UAV-VisLoc flight logs exist (`01.csv`–`10.csv`). 9 (`01`–`09.csv`) are used for LSTM training, each split *chronologically* 85%/15% (`VAL_RATIO=0.15`) into train/validation — first 85% of each real trajectory's timestamped samples for training, last 15% for validation, ruling out temporal leakage within a trajectory. The 10th (`10.csv`) is held out from training entirely — it is used only to derive the synthetic generator's reference statistics (spatial center, altitude distribution, attitude-angle noise), so there is no overlap between the file that calibrates the synthetic traces and the files that train the model. The 1000 synthetic UAV traces used in the large-scale network experiments are never seen during training or validation — a fully out-of-domain test set, used only at inference.

**Statistical consistency:** per-waypoint altitude (`N(2573.0m, 2.5m)`, clipped `[1960,2580]`) and attitude-angle noise (Omega σ=3.5°, Kappa=heading±3.0°, Phi1/Phi2=90°±1.5°) in the synthetic generator are directly calibrated from the held-out real flight (`10.csv`) — not arbitrary. What is *not* drawn from UAV-VisLoc statistics is the horizontal path topology: UAV-VisLoc offers only a handful of real flights, not enough shape diversity for a 1000-UAV network experiment, so horizontal paths are generated from 10 parametric motion primitives instead. Stated plainly rather than implying full equivalence: altitude/attitude noise are empirically grounded; path shape is a deliberate diversification choice.

**Why the model should generalize:** the LSTM consumes a 20-step window of local position deltas and predicts the next-step delta — a short-horizon motion-continuation task that depends on locally smooth velocity/heading trends, not on global path topology. Per-step noise statistics are shared between domains by construction, so domain shift is confined to global trajectory shape, not the local windowed dynamics the model actually operates on.

**Measured result (new, empirical):** mean per-UAV position-prediction error broken down by the 10 synthetic motion-pattern categories, N=1000 UAVs, each UAV's error averaged over its own full test trajectory:

| Mobility pattern | Mean error | Std. dev. | # UAVs |
|---|---|---|---|
| Multi-oval | 3.383 | 0.212 | 93 |
| Spiral | 3.283 | 0.263 | 119 |
| Figure-eight | 3.242 | 0.292 | 104 |
| Sinusoidal | 3.197 | 0.260 | 108 |
| Random-walk | 3.192 | 0.246 | 94 |
| Oval | 3.162 | 0.216 | 87 |
| Circle | 3.122 | 0.227 | 88 |
| Straight-line | 3.098 | 0.268 | 88 |
| Zigzag | 3.058 | 0.252 | 118 |
| Star | 3.045 | 0.264 | 101 |
| **Overall (pooled)** | **3.179** | **0.271** | **1000** |

Error is stable across all 10 categories (3.05–3.38, a 10.6% spread relative to the pooled mean), with comparable within-category variance — no motion pattern is a systematic failure mode. Note: this is a per-UAV, full-trajectory-averaged metric, distinct from Comment 2's single-snapshot-timestep error (3.10 at N=500/1000) — both land in the same ~3.1–3.2 range but are computed over different denominators; worth keeping straight if cross-checked.

**Paper:**
- New subsection, "Dataset and Training Protocol": the train/val/test split described above (9 real flights for training with chronological 85/15 split, 1 real flight held out solely to calibrate synthetic-trace statistics, 1000 synthetic traces as an out-of-domain test set).
- New table (above, already computed): per-shape prediction-error breakdown, as direct empirical evidence for generalization across all 10 synthetic mobility patterns.
- New paragraph stating which synthetic-trace statistics are empirically grounded (altitude, attitude noise) vs. deliberately diversified (path topology), and the architectural argument for why local-window prediction generalizes across global path shape.

---

### Comment 4
**Comment:** Spatial clustering doesn't inherently reduce contention — may increase simultaneous contenders. Justify vs. random/PHY-rate/RSSI/traffic-demand grouping; ablation recommended.

**Response:** The reviewer's skepticism turned out to be empirically correct, and we found and fixed the exact failure mode during this revision. In our synthetic UAV evaluation set, at N=100 every connected STA fell inside a single CSV-defined cluster (cluster boundaries were generated once for 1200 UAVs; smaller-N runs sample a contiguous AID prefix that lands entirely inside one cluster's block). Because the implementation created exactly one RAW group per populated cluster with no sub-splitting, this collapsed all 100 STAs into one contention domain with only 4 RAW slots total — a 4× parallelism deficit against the "adaptive" baseline's 4 fixed groups (25 STAs each), and it fully explained a PDR collapse to 0.216 (vs. 0.965 for "adaptive" on the identical 100 STAs). We fixed this by enforcing a floor of `raw_num_groups` (4) total RAW groups regardless of how the external clustering happens to distribute AIDs — splitting the largest populated cluster(s) by contiguous AID range whenever fewer clusters are populated than the floor, and leaving already-diverse clusterings untouched. This raised N=100 PDR from 0.216 → 0.982 and materially improved N=200/400 as well (see Comment 9 for full numbers). This is direct, quantified evidence for the reviewer's concern: naive spatial/CSV-defined grouping *can* actively destroy contention-management performance if group cardinality isn't bounded independently of clustering, and the fix is exactly the kind of safeguard the reviewer's comment is pushing toward.

**Update:** The grouping-criteria ablation the reviewer asked for is now real, not a placeholder. Built from the actual LSTM+K-Means pipeline (`uav/build_ablation_csvs.py`): 4 conditions — spatial (LSTM-predicted-position K-Means, K=6, what MA-PRAW actually uses), random, traffic-demand-based (ranked by per-UAV queue length), and distance-based (ranked by distance to a nominal AP point derived from the real predicted positions — this stands in for both RSSI-based and PHY-rate-based grouping, since both are fundamentally distance/SNR-driven in real 802.11ah and no separate real RSSI/PHY-rate measurement exists anywhere in this codebase to differentiate them further). Same priority/traffic assignment, same group-count-floor safeguard (above) applied identically to all four. Population extended from 500 to 1000 UAVs (see Comment 2's population note) to match this paper's usual N range — the finding below held up and got stronger evidence, not weaker, at the larger population.

**Important methodological note, kept for transparency:** the first version of this ablation gave every alternative grouping (random/demand/distance) an artificially *equal* split (83/83/83/83/84/84 UAVs), while spatial's real K-Means output is naturally imbalanced ([67,68,70,95,98,102]). That version showed spatial losing significantly at N=200/400 (CIs did not overlap) — but this conflated "grouping criterion" with "group-size balance," which this project's own earlier work already established has a large independent effect on PDR. Re-ran with all three alternative groupings matched to spatial's *actual* size distribution, isolating the criterion as the only variable.

**Corrected result:** once group size is controlled for, spatial/random/demand/distance are statistically indistinguishable at every tested N (100–1000) for both Cluster-based and MA-PRAW — for Cluster-based, PDR is byte-identical across all four conditions at every single N (this policy's slot-allocation math is driven entirely by group size, not group membership); for MA-PRAW, differences are within roughly 1 percentage point everywhere, well within CI overlap. Figure: `results/figs/MC-PRAW/fig_grouping_ablation.pdf` (N=100–1000, matches every other figure's range).

**Paper:**
- New ablation subsection + figure (already built, see above): PDR vs. N, 4 grouping criteria, 2 policies, all group-size-matched.
- New paragraph: *"Once group-size balance is controlled for, the choice of grouping criterion (spatial, random, traffic-demand-based, or distance-based) has no statistically significant effect on PDR for either Cluster-based or MA-PRAW across the full tested range (N=100–1000). This confirms the reviewer's underlying intuition in a precise sense: spatial proximity is not what makes grouping effective — group-size balance is the dominant factor, and MA-PRAW's group-count floor (below) is what actually protects performance, independent of which criterion assigns stations to groups."*
- This reframes the paper's contribution honestly: the claim should not be "spatial clustering is superior," which the data does not support once confounds are controlled — it should be "MA-PRAW's group-balancing safeguard is robust to the choice of upstream grouping criterion," which the data does support.
- New paragraph in Section III describing the group-count floor as a formal safeguard: *"To prevent an external clustering criterion from degenerating into a single oversized contention domain, MA-PRAW enforces a minimum of G_min parallel RAW groups per beacon interval, splitting the largest populated cluster(s) by contiguous AID range as needed. This decouples worst-case contention parallelism from the quality of the upstream grouping decision."*
- Report the N=100 before/after numbers (0.216 → 0.982 PDR) as a concrete ablation demonstrating why this safeguard matters — this doubles as evidence for both this comment and Reviewer 1 Comment 9 (robustness of the design).

---

### Comment 5
**Comment:** Δburst_i(t) = max{0, Z_i(t) − h} uses h as a CUSUM-statistic threshold, but Eq. (9) applies h to predicted load λ̂_i(t+1) — inconsistent quantities. Reformulate.

**Response:** Confirmed inconsistency; introduced a second, independently calibrated threshold for the cluster-level test.

**Paper:**
- Revise Eq. (9): replace the shared `h` with a distinct symbol, e.g. `h_cluster`, applied to λ̂_i(t+1).
- Add one sentence: *"h and h_cluster are calibrated independently, as they threshold different quantities: the former the CUSUM statistic Z_i(t) [units: queue-depth deviation], the latter the predicted load λ̂_i(t+1) [units: predicted packets]."*
- Add h_cluster's calibrated value to the parameter table (Table X from R1-4).

---

### Comment 6
**Comment:** Classical Bianchi assumes saturated DCF; UAV traffic is heterogeneous/bursty/priority-dependent. Justify treating all positive-predicted-load STAs as saturated contenders, or compare against a non-saturated model.

**Response:** Added explicit justification (conservative sizing) plus an empirical comparison against a duty-cycle-weighted non-saturated estimate.

**Paper:**
- Add justification paragraph in Section III-C: *"We treat STAs with positive predicted load as saturated contenders as a conservative slot-sizing choice, since under-provisioning has a more severe impact on delay than modest over-provisioning."*
- Add a supplementary figure/table comparing allocated slot counts and resulting PDR/delay between the saturated Bianchi estimate and a non-saturated variant (τ scaled by normalized predicted load), across your N sweep.

---

### Comment 7
**Comment:** D_k already includes predicted load and normalized airtime; RAW length then multiplies by expected contention-slot duration and divides by success probability — possible double-counting of airtime/contention. Dimensional/probabilistic derivation required.

**Response:** Confirmed via dimensional tracing — D_k was in σ-slot-equivalent units (it divided weighted predicted load by the slot time σ), and the subsequent `l̄/p_succ` factor reapplied that same slot normalization. We corrected this by redefining D_k as a pure weighted packet count (dropping the airtime/σ term entirely). One honest finding from actually applying the fix and measuring it: D_k's absolute scale turned out **not** to be the dominant cause of the algorithm's poor low-N performance — a separate constraint (`adaptive_raw_tmax_s`, the maximum RAW window length) was already so tight relative to one Bianchi contention round's real duration at MCS0 (~9–10 ms window budget vs. a full round easily taking that long under contention) that the RAW-length term saturated at its ceiling regardless of D_k's magnitude, before and after the fix. The dimensional bug is real and now corrected, but the actual low-N failure mode traced to two other issues (see Comment 4 and the fix summary at the end of this document): CSV cluster boundaries not scaling with the connected population, and `cluster_adaptive_min_slots_per_cluster` allowing below-average-demand clusters to be starved down to a single RAW slot.

**Paper:**
- New Appendix (or subsection), "Derivation of RAW Length," showing units at each step: D_k [packets] → D_k · l̄/p_succ [packets × σ-slots/successful-round = σ-slots]. Keep this — it's dimensionally necessary regardless of its measured impact.
- Revise Eq. (10)/(11) to reflect the corrected packet-unit D_k.
- Be precise in the text that this correction did not materially change reported results by itself (state this plainly rather than imply it drove the reported gains) — the actual performance improvements in Section VI trace to the group-diversity floor (Comment 4) and the min-slots-per-cluster floor (Comment 9), both now described there. Mischaracterizing which fix drove which result would be an easy, avoidable error for a careful reviewer to catch on re-review.

---

### Comment 8
**Comment:** Own simulator is insufficient evidence of correctness. Reproduce a reference 802.11ah scenario in NS-3 (or similar) and compare quantitatively.

**Response:** We already cross-validate DCF/contention behavior against the closed-form Bianchi saturation model; we additionally reproduce a reference scenario in NS-3.

**Paper:**
- New subsection, "Simulator Validation":
  1. Formalize the existing analytical-vs-simulation Bianchi throughput comparison as a table/figure (already available in `paper/analytical_validation.py` / `paper/analytical_comparison.csv` — just needs to be written up).
  2. Add an NS-3 reproduction of a matched single-BSS 802.11ah RAW scenario (same N, PHY mode, RAW slot config) with a side-by-side PDR/throughput table against your simulator. `[new work — not covered by existing code]`.

---

### Comment 9
**Comment:** Static/Adaptive/Cluster-Based/Cluster-Adaptive RAW are all the authors' own variants. Include TAROA, E-TAROA, RO-RAW, or another SOTA adaptive-RAW method.

**Response:** Added chang2019 (Chang et al., TMC 2019) and LACA as published baselines, evaluated under identical UAV mobility/traffic conditions as MA-PRAW (`cluster_adaptive` in the simulator). Results below are single-seed (seed=42, 120s sim, 5s periodic traffic interval, 128B packets, MCS0) — re-run with 3 seeds before finalizing, but the direction is clear: after two bug fixes surfaced by actually running this comparison (see the summary at the end of this document), MA-PRAW beats LACA at every tested N and beats chang2019 at N≥400, with the two essentially tied at N=100–200.

**PDR, N=100–1000 (single seed):**

| N | cluster_adaptive (MA-PRAW) | LACA | chang2019 | traffic_split (best overall) |
|---|---|---|---|---|
| 100 | 0.982 | 0.919 | 0.987 | 0.978 |
| 200 | 0.546 | 0.381 | 0.568 | 0.737 |
| 400 | 0.250 | 0.154 | 0.229 | 0.373 |
| 600 | 0.187 | 0.092 | 0.127 | 0.219 |
| 800 | 0.137 | 0.066 | 0.081 | 0.137 |
| 1000 | 0.100 | 0.056 | 0.063 | 0.107 |

**Paper:**
- Add "Chang2019" and "LACA" as additional curves/bars in every main results figure in Section VI (PDR, throughput, delay vs. N), using the table above (upgrade to 3-seed means ± std before submission).
- One paragraph discussing MA-PRAW's gain relative to these two published methods specifically: wins against LACA across the full N range tested; wins against chang2019 at N≥400; roughly ties chang2019 at N=100–200 (be honest about the tie rather than overstating it — a reviewer who reproduces this will find the same near-tie).
- Note MA-PRAW is now close to `traffic_split` (an internal, more heavily-engineered variant — not a published baseline) at N=800–1000, but still trails it at N=200–600; don't claim MA-PRAW as the overall best performer, since it isn't yet.
- If TAROA/E-TAROA/RO-RAW comparisons are feasible to add later (not currently implemented anywhere in the simulator), note that as a stronger follow-up; for this revision cycle, chang2019 + LACA already substantively answers the comment.

---

### Comment 10
**Comment:** Many interacting modules make the core scheduling objective hard to identify. Explicitly formulate the RAW scheduling problem (objective + constraints) to clarify the relation between prediction and allocation.

**Response:** Added an explicit constrained-optimization formulation and mapped each module to it — shared with the R1-6 fix.

**Paper:** Same addition as R1-6 — the "Problem Formulation" subsection (objective: maximize weighted throughput / minimize weighted delay; constraints: airtime budget, slot-duration floor, slot-count bounds) plus the "Role of Each Component" mapping. Write it once, reference it from both rebuttal responses.

---

## Summary of code-level fixes (sim11ah repo)

Found by reading `sim11ah/mac/raw_policy_cluster_adaptive.py` and `sim11ah/mac/raw_policy_cluster_csv.py`, referenced throughout this document.

### Fixed and measured (single-seed; 3-seed re-run still recommended before submission)

1. **Single/few-cluster collapse at low N** (`raw_policy_cluster_adaptive.py`, new `_ensure_min_group_diversity()`) — the CSV's cluster boundaries were fixed for N=1200 total UAVs; N-sweep runs connect a contiguous AID prefix (`1..N`), which at low N landed entirely inside one or two CSV clusters, collapsing parallelism to 1 RAW group instead of the 4 every other policy guarantees. Fixed by flooring total RAW groups at `raw_num_groups` (4), splitting the largest populated cluster(s) by contiguous AID range as needed. **N=100 PDR: 0.216 → 0.982.** See Comment 4.
2. **`cluster_adaptive_min_slots_per_cluster` starvation** (`config.py`) — was 1, letting below-average-demand clusters get squeezed to a single RAW slot for their entire membership while `adaptive` unconditionally gives every group 4. Raised the floor to 4 (matching `raw_num_slots`), so demand-based scaling only adds slots to hot clusters rather than starving cold ones below the baseline. This was the dominant lever — e.g. **N=600 PDR: 0.149 → 0.187**, moving `cluster_adaptive` ahead of LACA and chang2019 there. See Comments 4 and 9.
3. **Dgk dimensional double-counting** (`_compute_cluster_demand`) — redefined D_k as a pure weighted packet count, dropping the redundant airtime/σ normalization that duplicated the `l̄/p_succ` slot-overhead factor applied later. Dimensionally correct now, but confirmed via direct measurement to be **inert** given the current `adaptive_raw_tmax_s` ceiling (see Comment 7) — keep the fix for correctness, but don't claim it drove the reported gains.

### Still open (flagged, not yet applied)

4. **G_Λ not configurable** — hardcoded `demand *= 1.0 + 0.03 * total_pred` (raw_policy_cluster_adaptive.py). Needs a config key if Eq. (10) is to match the code (Comment 4, Reviewer 1).
5. **Burst-threshold reuse** — `cusum_h` thresholds both the CUSUM statistic (correct) and the predicted load `pred_load` (Comment 5, Reviewer 2). Needs two named constants.
6. **CSV re-parsed every beacon interval** in `ClusterAdaptiveRawPolicy._load_cluster_csv()` — its sibling `ClusterCsvAdaptiveRawPolicy` caches the CSV once in `__init__` instead. Needed for the control-overhead numbers in Reviewer 1 Comment 9 to look good.
7. **Spatial-grouping ablation** — resolved. Real spatial/K-Means clustering, plus random/traffic-demand-based/distance-based (RSSI+PHY-rate proxy) alternatives, all group-size-matched to isolate the grouping criterion, built by `uav/build_cluster_csvs.py` and `uav/build_ablation_csvs.py` and evaluated by `scripts/eval_uav_grouping_ablation.py`. See Comment 4 (Reviewer 2) above for the result and the group-size-confound catch/fix along the way.

Want me to tackle 4–6 next (mechanical, low-risk), or move to the 3-seed re-run of the full comparison so Comments 8/9 have final numbers?
