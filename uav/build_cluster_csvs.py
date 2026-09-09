"""
Build real predicted-position and oracle (true-position) cluster CSVs for
the MA-PRAW UAV RAW-scheduling simulator, from the actual trained LSTM +
K-Means pipeline in 00_uav_lstm_clustering_final.ipynb -- reproduced here
as a script (notebooks aren't easily run headlessly, and the notebook's
training cells 9-10 would retrain with randomness we want to avoid; this
loads the already-trained best_uav_model.keras instead, matching the
notebook's own inference cells 1-8, 19-22 exactly).

Addresses Reviewer 2 Comment 2 (oracle baseline using true future
positions to establish the MAC-layer cost of prediction error) and grounds
Comment 4's grouping-methodology question in the paper's actual claimed
spatial clustering, replacing sim11ah/utils/generate_cluster_csv.py's
random placeholder.

Produces two simulator-compatible CSVs (same schema and same
priority/tx_rate/queue_len assignment PROCEDURE as generate_cluster_csv.py,
so clustering methodology is the only variable that differs between
conditions):
  uav_cluster_data_predicted.csv  -- K-Means on LSTM-PREDICTED positions
  uav_cluster_data_oracle.csv     -- K-Means on TRUE future positions
                                      (same K, same snapshot timestep)

500 UAVs (the notebook's actual test population, data/uav_synthetic_output/
csv/*.csv), K=6 (the notebook's own elbow/silhouette-selected OPTIMAL_K,
see optimal_k_selection.png), snapshot at the midpoint timestep (matching
kmeans_cluster_dynamics.png / 3d_kmeans_cluster_snapshot.png's own
convention, t_mid = min_timesteps // 2).
"""
import csv
import glob
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

UAV_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(UAV_DIR, "..")

FEATURES = ["lat", "lon", "height", "Omega", "Kappa", "Phi1", "Phi2"]
DATE_COL = "date"
SEQ_LEN = 20
VAL_RATIO = 0.15
K = 6  # OPTIMAL_K from the notebook's own elbow/silhouette search (cell 21-22)


def load_sorted(filepath):
    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df[FEATURES].values.astype(np.float32)


def add_velocity_features(data):
    velocity = np.zeros_like(data[:, :3])
    velocity[1:] = data[1:, :3] - data[:-1, :3]
    return np.concatenate([data, velocity], axis=1).astype(np.float32)


def make_sequences_delta(data_with_vel, raw_data, seq_len):
    X, y = [], []
    for i in range(len(data_with_vel) - seq_len):
        X.append(data_with_vel[i: i + seq_len])
        delta = raw_data[i + seq_len, :3] - raw_data[i + seq_len - 1, :3]
        y.append(delta)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def assign_priority_queue(uav_ids, seed=42):
    """Assign priority/queue_len ONCE per uav_id, in a fixed uav_id-sorted
    order that is INDEPENDENT of any clustering -- this is what makes
    "only the clustering methodology differs" actually true across
    predicted.csv / oracle.csv / the grouping-ablation CSVs. (An earlier
    version assigned these while iterating in cluster-sorted order, so a
    given UAV could get a different priority in different conditions,
    since predicted/oracle clusterings order UAVs differently -- fixed
    here.) Same distributions as sim11ah/utils/generate_cluster_csv.py."""
    rng = random.Random(seed)
    priorities = ["critical", "high", "normal"]
    priority_prob = [0.10, 0.25, 0.65]
    out = {}
    for uid in sorted(uav_ids):
        priority = rng.choices(priorities, weights=priority_prob, k=1)[0]
        if priority == "critical":
            queue_len = rng.randint(8, 25)
        elif priority == "high":
            queue_len = rng.randint(4, 15)
        else:
            queue_len = rng.randint(0, 8)
        out[uid] = (priority, queue_len)
    return out


def write_cluster_csv(uav_ids, labels, uav_priority_queue, out_path):
    """Same schema as sim11ah/utils/generate_cluster_csv.py -- only
    cluster_id/AID-ordering differs between conditions, since
    priority/queue_len come from the pre-assigned, clustering-independent
    uav_priority_queue map. UAVs are sorted by cluster label so each
    cluster becomes a contiguous AID block, matching the RAW mechanism's
    start_aid/end_aid requirement."""
    order = sorted(range(len(uav_ids)), key=lambda idx: (int(labels[idx]), uav_ids[idx]))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["aid", "cluster_id", "priority", "tx_rate_bps", "queue_len", "packet_size_bytes"])
        aid = 1
        for idx in order:
            cid = int(labels[idx]) + 1  # 1-indexed, matching existing convention
            priority, queue_len = uav_priority_queue[uav_ids[idx]]
            writer.writerow([aid, cid, priority, 300000, queue_len, 128])
            aid += 1

    print(f"Wrote {out_path} ({len(uav_ids)} rows)")


def main():
    train_files = sorted(glob.glob(os.path.join(UAV_DIR, "dataset", "*.csv")))[:9]
    test_files2 = sorted(glob.glob(os.path.join(UAV_DIR, "data", "uav_synthetic_output", "csv", "*.csv")))
    print(f"train_files: {len(train_files)}, test_file_2 (clustering population): {len(test_files2)}")

    # --- Refit scalers exactly as notebook cells 6-8 (deterministic) ---
    all_train_rows = []
    for f in train_files:
        raw = load_sorted(f)
        split = int(len(raw) * (1 - VAL_RATIO))
        enriched = add_velocity_features(raw[:split])
        all_train_rows.append(enriched)
    all_train_rows = np.concatenate(all_train_rows, axis=0)
    x_scaler = StandardScaler()
    x_scaler.fit(all_train_rows)
    n_features = all_train_rows.shape[1]

    y_train_list = []
    for f in train_files:
        raw = load_sorted(f)
        split = int(len(raw) * (1 - VAL_RATIO))
        raw_split = raw[:split]
        if len(raw_split) <= SEQ_LEN:
            continue
        enriched = add_velocity_features(raw_split)
        _, ys = make_sequences_delta(enriched, raw_split, SEQ_LEN)
        y_train_list.append(ys)
    y_train = np.concatenate(y_train_list, axis=0)
    y_scaler = StandardScaler()
    y_scaler.fit(y_train)
    print(f"Scalers refit. n_features={n_features}, y_train rows={y_train.shape[0]}")

    model = tf.keras.models.load_model(os.path.join(UAV_DIR, "best_uav_model.keras"))

    # --- Predicted + true positions for all test UAVs (cell 19, extended with ground truth) ---
    uav_predicted_positions = {}
    uav_true_positions = {}
    for i, f in enumerate(test_files2):
        test_raw = load_sorted(f)
        test_enriched = add_velocity_features(test_raw)
        X_test, _ = make_sequences_delta(test_enriched, test_raw, SEQ_LEN)
        if len(X_test) == 0:
            continue
        X_test_scaled = x_scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        pred_delta_scaled = model.predict(X_test_scaled, verbose=0)
        pred_delta = y_scaler.inverse_transform(pred_delta_scaled)
        anchor = test_raw[SEQ_LEN - 1: SEQ_LEN - 1 + len(pred_delta), :3]
        uav_predicted_positions[i] = anchor + pred_delta
        uav_true_positions[i] = test_raw[SEQ_LEN: SEQ_LEN + len(pred_delta), :3]
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(test_files2)} UAVs")

    print(f"Collected positions for {len(uav_predicted_positions)} UAVs")

    # --- Align to common time grid (cell 20) ---
    min_timesteps = min(pos.shape[0] for pos in uav_predicted_positions.values())
    uav_ids = sorted(uav_predicted_positions.keys())
    n_uavs = len(uav_ids)

    pred_matrix = np.stack([uav_predicted_positions[i][:min_timesteps] for i in uav_ids], axis=1)
    true_matrix = np.stack([uav_true_positions[i][:min_timesteps] for i in uav_ids], axis=1)
    print(f"Position matrices: {pred_matrix.shape} (timesteps x uavs x [lat,lon,height]), n_uavs={n_uavs}")

    t_mid = min_timesteps // 2
    print(f"Using snapshot timestep t={t_mid} of {min_timesteps} (matches notebook's own diagnostic convention)")

    def cluster_at(matrix, t):
        pos_t = matrix[t]
        pos_sc = StandardScaler().fit_transform(pos_t)
        km = KMeans(n_clusters=K, random_state=42, n_init=5)
        return km.fit_predict(pos_sc)

    predicted_labels = cluster_at(pred_matrix, t_mid)
    oracle_labels = cluster_at(true_matrix, t_mid)

    pred_sizes = sorted(int((predicted_labels == k).sum()) for k in range(K))
    oracle_sizes = sorted(int((oracle_labels == k).sum()) for k in range(K))
    print(f"Predicted-position cluster sizes (sorted): {pred_sizes}")
    print(f"Oracle (true-position) cluster sizes (sorted): {oracle_sizes}")

    # Per-UAV position error at the snapshot timestep, for reporting
    err = np.linalg.norm(pred_matrix[t_mid] - true_matrix[t_mid], axis=1)
    print(f"Position error @ t={t_mid}: mean={err.mean():.4f}, std={err.std():.4f}, "
          f"min={err.min():.4f}, max={err.max():.4f}")

    uav_priority_queue = assign_priority_queue(uav_ids, seed=42)
    write_cluster_csv(uav_ids, predicted_labels, uav_priority_queue,
                       os.path.join(OUT_DIR, "uav_cluster_data_predicted.csv"))
    write_cluster_csv(uav_ids, oracle_labels, uav_priority_queue,
                       os.path.join(OUT_DIR, "uav_cluster_data_oracle.csv"))


if __name__ == "__main__":
    main()
