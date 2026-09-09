"""
Grouping-strategy ablation for Reviewer 2 Comment 4: "explain why spatial
clustering is superior to random grouping, PHY-rate-based grouping,
airtime-based grouping, RSSI-based grouping, or traffic-demand-based
grouping."

Builds 3 additional simulator-ready cluster CSVs, all covering the SAME
500 real-pipeline UAVs, K=6, and the SAME priority/tx_rate/queue_len
assignment procedure as uav_cluster_data_predicted.csv -- only the
GROUPING CRITERION differs, so this is a controlled ablation against the
existing spatial (LSTM-predicted-position K-Means) clustering:

  uav_cluster_data_random.csv    -- uniform random assignment to 6 groups
  uav_cluster_data_demand.csv    -- grouped by traffic-demand rank (queue_len)
  uav_cluster_data_distance.csv  -- grouped by rank-distance to a nominal AP
                                     point, computed from the REAL predicted
                                     positions already built by
                                     build_cluster_csvs.py. This is a
                                     genuine position-derived criterion, so
                                     it stands in for BOTH RSSI-based and
                                     PHY-rate-based grouping (both are
                                     fundamentally distance/SNR-driven in
                                     real 802.11ah -- there is no separate
                                     real RSSI or PHY-rate measurement
                                     anywhere in this codebase to
                                     differentiate them further).

Reuses build_cluster_csvs.py's pipeline-reproduction functions rather than
duplicating the LSTM inference logic.
"""
import csv
import glob
import os
import random

import numpy as np
from sklearn.preprocessing import StandardScaler

import build_cluster_csvs as bc

UAV_DIR = bc.UAV_DIR
OUT_DIR = bc.OUT_DIR
K = bc.K


def write_ablation_csv(order_by_group, uav_priority_queue, out_path):
    """order_by_group: list of lists of uav_ids (local indices), one per
    group, in the order groups should be assigned contiguous AID blocks.
    Priority/queue_len come from the SAME clustering-independent map
    build_cluster_csvs.py uses for predicted.csv/oracle.csv (fixed by
    uav_id, not by iteration order), so that axis is held constant across
    every condition in this ablation too."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["aid", "cluster_id", "priority", "tx_rate_bps", "queue_len", "packet_size_bytes"])
        aid = 1
        for cid, members in enumerate(order_by_group, start=1):
            for uid in members:
                priority, queue_len = uav_priority_queue[uid]
                writer.writerow([aid, cid, priority, 300000, queue_len, 128])
                aid += 1
    print(f"Wrote {out_path} ({sum(len(m) for m in order_by_group)} rows, "
          f"group sizes={sorted(len(m) for m in order_by_group)})")


def main():
    train_files = sorted(glob.glob(os.path.join(UAV_DIR, "dataset", "*.csv")))[:9]
    test_files2 = sorted(glob.glob(os.path.join(UAV_DIR, "data", "uav_synthetic_output", "csv", "*.csv")))
    print(f"train_files: {len(train_files)}, test_file_2: {len(test_files2)}")

    all_train_rows = []
    for f in train_files:
        raw = bc.load_sorted(f)
        split = int(len(raw) * (1 - bc.VAL_RATIO))
        all_train_rows.append(bc.add_velocity_features(raw[:split]))
    all_train_rows = np.concatenate(all_train_rows, axis=0)
    x_scaler = StandardScaler()
    x_scaler.fit(all_train_rows)
    n_features = all_train_rows.shape[1]

    y_train_list = []
    for f in train_files:
        raw = bc.load_sorted(f)
        split = int(len(raw) * (1 - bc.VAL_RATIO))
        raw_split = raw[:split]
        if len(raw_split) <= bc.SEQ_LEN:
            continue
        enriched = bc.add_velocity_features(raw_split)
        _, ys = bc.make_sequences_delta(enriched, raw_split, bc.SEQ_LEN)
        y_train_list.append(ys)
    y_train = np.concatenate(y_train_list, axis=0)
    y_scaler = StandardScaler()
    y_scaler.fit(y_train)

    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(UAV_DIR, "best_uav_model.keras"))

    uav_predicted_positions = {}
    for i, f in enumerate(test_files2):
        test_raw = bc.load_sorted(f)
        test_enriched = bc.add_velocity_features(test_raw)
        X_test, _ = bc.make_sequences_delta(test_enriched, test_raw, bc.SEQ_LEN)
        if len(X_test) == 0:
            continue
        X_test_scaled = x_scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        pred_delta_scaled = model.predict(X_test_scaled, verbose=0)
        pred_delta = y_scaler.inverse_transform(pred_delta_scaled)
        anchor = test_raw[bc.SEQ_LEN - 1: bc.SEQ_LEN - 1 + len(pred_delta), :3]
        uav_predicted_positions[i] = anchor + pred_delta
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(test_files2)} UAVs")

    min_timesteps = min(pos.shape[0] for pos in uav_predicted_positions.values())
    uav_ids = sorted(uav_predicted_positions.keys())
    n_uavs = len(uav_ids)
    pred_matrix = np.stack([uav_predicted_positions[i][:min_timesteps] for i in uav_ids], axis=1)
    t_mid = min_timesteps // 2
    positions_t = pred_matrix[t_mid]  # (n_uavs, 3) lat, lon, height -- same snapshot as build_cluster_csvs.py
    print(f"n_uavs={n_uavs}, snapshot t={t_mid}")

    # Same clustering-independent priority/queue_len map predicted.csv and
    # oracle.csv use -- held constant across every condition here too, so
    # only the grouping criterion varies.
    uav_priority_queue = bc.assign_priority_queue(uav_ids, seed=42)

    # Use the SAME group-size distribution as the spatial (K-Means)
    # clustering, not an artificial equal split -- K-Means naturally
    # produces imbalanced groups (e.g. [67,68,70,95,98,102] here), and this
    # session's own earlier work established that group-size imbalance
    # alone measurably affects PDR. Using equal splits for the alternative
    # groupings would confound "grouping criterion" with "group balance";
    # matching spatial's real sizes isolates the criterion as the only
    # variable, which is what this ablation is actually supposed to test.
    predicted_csv_path = os.path.join(OUT_DIR, "uav_cluster_data_predicted.csv")
    cluster_counts = {}
    with open(predicted_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cid = int(row["cluster_id"])
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
    sizes = [cluster_counts[cid] for cid in sorted(cluster_counts)]
    assert sum(sizes) == n_uavs, f"size mismatch: {sum(sizes)} != {n_uavs}"
    print(f"Using spatial clustering's real group sizes: {sizes}")

    # --- 1. Random grouping ---
    shuffled = uav_ids.copy()
    random.Random(42).shuffle(shuffled)
    groups_random, idx = [], 0
    for s in sizes:
        groups_random.append(shuffled[idx: idx + s])
        idx += s
    write_ablation_csv(groups_random, uav_priority_queue,
                        os.path.join(OUT_DIR, "uav_cluster_data_random.csv"))

    # --- 2. Traffic-demand-based grouping (rank by queue_len, contiguous bins) ---
    by_demand = sorted(uav_ids, key=lambda uid: (uav_priority_queue[uid][1], uid))
    groups_demand, idx = [], 0
    for s in sizes:
        groups_demand.append(by_demand[idx: idx + s])
        idx += s
    write_ablation_csv(groups_demand, uav_priority_queue,
                        os.path.join(OUT_DIR, "uav_cluster_data_demand.csv"))

    # --- 3. Distance-to-AP-based grouping (real position-derived; stands in
    # for RSSI-based / PHY-rate-based grouping, both fundamentally
    # distance/SNR-driven in real 802.11ah). AP = centroid of all UAV
    # positions at the snapshot timestep (a stand-in ground-station location).
    ap_pos = positions_t.mean(axis=0)
    dist = {uid: float(np.linalg.norm(positions_t[local_idx] - ap_pos))
            for local_idx, uid in enumerate(uav_ids)}
    by_distance = sorted(uav_ids, key=lambda uid: (dist[uid], uid))
    groups_distance, idx = [], 0
    for s in sizes:
        groups_distance.append(by_distance[idx: idx + s])
        idx += s
    write_ablation_csv(groups_distance, uav_priority_queue,
                        os.path.join(OUT_DIR, "uav_cluster_data_distance.csv"))


if __name__ == "__main__":
    main()
