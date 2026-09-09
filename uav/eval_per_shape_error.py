"""
Per-mobility-shape prediction-error breakdown for Reviewer 2 Comment 3:
"why should a model trained on UAV-VisLoc generalize to the ten generated
mobility patterns?" -- this computes the SAME LSTM position-prediction
error build_cluster_csvs.py reports as a single pooled mean, but broken
down per shape category (straight_line, zigzag, oval, multi_oval, circle,
spiral, star, figure8, random_walk, sinusoidal), by joining each test
UAV's snapshot-timestep error against its shape label in
uav_synthetic_output/summary.csv.

Reuses build_cluster_csvs.py's exact scaler-refit + inference pipeline so
this is the identical model/data, not a re-derivation.

Output: results/uav_per_shape_prediction_error.csv
"""
import csv
import glob
import os

import numpy as np
import pandas as pd

import build_cluster_csvs as bc

UAV_DIR = bc.UAV_DIR
ROOT = os.path.join(UAV_DIR, "..")


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
    from sklearn.preprocessing import StandardScaler
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

    # uav_id (from filename) -> per-timestep position error array
    uav_errors = {}
    for f in test_files2:
        uid = int(os.path.basename(f).replace("uav_", "").replace(".csv", ""))
        test_raw = bc.load_sorted(f)
        test_enriched = bc.add_velocity_features(test_raw)
        X_test, _ = bc.make_sequences_delta(test_enriched, test_raw, bc.SEQ_LEN)
        if len(X_test) == 0:
            continue
        X_test_scaled = x_scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        pred_delta_scaled = model.predict(X_test_scaled, verbose=0)
        pred_delta = y_scaler.inverse_transform(pred_delta_scaled)
        anchor = test_raw[bc.SEQ_LEN - 1: bc.SEQ_LEN - 1 + len(pred_delta), :3]
        pred_pos = anchor + pred_delta
        true_pos = test_raw[bc.SEQ_LEN: bc.SEQ_LEN + len(pred_delta), :3]
        err = np.linalg.norm(pred_pos - true_pos, axis=1)  # per-timestep error, this UAV
        uav_errors[uid] = err.mean()  # per-UAV mean error across its own full trajectory
        if uid % 100 == 0:
            print(f"  processed uav_{uid:03d}")

    print(f"Collected errors for {len(uav_errors)} UAVs")

    summary = pd.read_csv(os.path.join(UAV_DIR, "data", "uav_synthetic_output", "summary.csv"))
    summary = summary[summary["uav_id"].isin(uav_errors.keys())].copy()
    summary["mean_pred_error"] = summary["uav_id"].map(uav_errors)

    grouped = summary.groupby("shape")["mean_pred_error"].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.sort_values("mean", ascending=False)

    out_path = os.path.join(ROOT, "results", "uav_per_shape_prediction_error.csv")
    grouped.to_csv(out_path, index=False)

    overall_mean = summary["mean_pred_error"].mean()
    overall_std = summary["mean_pred_error"].std()
    print(f"\nOverall: mean={overall_mean:.4f}, std={overall_std:.4f}, n={len(summary)}")
    print("\nPer-shape breakdown:")
    print(grouped.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
