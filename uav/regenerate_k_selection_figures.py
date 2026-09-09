"""
Re-run the notebook's own elbow+silhouette K-selection (cell 21) and 3D
cluster-snapshot plot (cell 24) on the extended 1000-UAV population, to
verify K*=6 (selected on the original 500-UAV set) still holds and to
regenerate the two paper figures (fig:optk, fig:cluster3d) consistently
with the population actually used in the Reviewer 2 Comment 2/4 rebuttal
experiments.

Reuses build_cluster_csvs.py's scaler-refit + LSTM-inference pipeline for
the predicted-position matrix (identical model/data/procedure), then
reproduces the notebook's cells verbatim:
  - Elbow/silhouette sweep: K=2..30, 10 sampled timestamps, KMeans
    n_init=10, silhouette sample_size=min(500, n_uavs).
  - 3D snapshot at t_mid = min_timesteps // 2, K=OPTIMAL_K, with cluster
    heads (UAV closest to centroid) and convex-hull floor projections.

Outputs (new files, originals NOT overwritten):
  uav/optimal_k_selection_1000uav.png
  uav/3d_kmeans_cluster_snapshot_1000uav.png
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import build_cluster_csvs as bc

UAV_DIR = bc.UAV_DIR
CACHE_PATH = os.path.join(UAV_DIR, "_position_matrix_cache_1000.npz")

plt.rcParams.update({
    "font.size": 14,
    "font.family": "sans-serif",
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


def compute_position_matrix():
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached position matrix from {CACHE_PATH}")
        data = np.load(CACHE_PATH)
        return data["position_matrix"], list(data["uav_ids"]), int(data["min_timesteps"]), int(data["n_uavs"])
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
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(test_files2)} UAVs")

    min_timesteps = min(pos.shape[0] for pos in uav_predicted_positions.values())
    uav_ids = sorted(uav_predicted_positions.keys())
    n_uavs = len(uav_ids)
    position_matrix = np.stack(
        [uav_predicted_positions[i][:min_timesteps] for i in uav_ids], axis=1
    )
    print(f"Position matrix: {position_matrix.shape} (timesteps x uavs x [lat,lon,height]), n_uavs={n_uavs}")
    np.savez(CACHE_PATH, position_matrix=position_matrix, uav_ids=np.array(uav_ids),
             min_timesteps=min_timesteps, n_uavs=n_uavs)
    print(f"Cached position matrix to {CACHE_PATH}")
    return position_matrix, uav_ids, min_timesteps, n_uavs


def elbow_silhouette(position_matrix, n_uavs, min_timesteps, out_path):
    MAX_K = 30
    N_SAMPLE_TIMESTAMPS = 10
    sample_ts = np.linspace(0, min_timesteps - 1, N_SAMPLE_TIMESTAMPS, dtype=int)

    inertias, silhouettes = [], []
    K_range = range(2, MAX_K + 1)

    print("Running Elbow + Silhouette analysis on 1000-UAV population ...")
    for K in K_range:
        inertia_k, sil_k = [], []
        for t in sample_ts:
            pos_t = position_matrix[t]
            pos_sc = StandardScaler().fit_transform(pos_t)
            km = KMeans(n_clusters=K, random_state=42, n_init=10)
            labels = km.fit_predict(pos_sc)
            inertia_k.append(km.inertia_)
            if len(set(labels)) > 1:
                sil_k.append(silhouette_score(pos_sc, labels, sample_size=min(500, n_uavs)))
        inertias.append(np.mean(inertia_k))
        silhouettes.append(np.mean(sil_k) if sil_k else 0)
        print(f"  K={K:2d}  inertia={inertias[-1]:8.1f}  silhouette={silhouettes[-1]:.4f}")

    inertias = np.array(inertias)
    silhouettes = np.array(silhouettes)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

    axes[0].plot(list(K_range), inertias, 'o-', color='royalblue', linewidth=2,
                 markersize=4)
    axes[0].set_title('Elbow Method', pad=8)
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Avg. Inertia')
    axes[0].grid(alpha=0.35, linewidth=0.6)
    axes[0].set_axisbelow(True)

    axes[1].plot(list(K_range), silhouettes, 's-', color='seagreen', linewidth=2,
                 markersize=4)
    axes[1].set_title('Silhouette Score', pad=8)
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Avg. Silhouette')
    axes[1].grid(alpha=0.35, linewidth=0.6)
    axes[1].set_axisbelow(True)

    best_sil_k = list(K_range)[np.argmax(silhouettes)]
    axes[1].axvline(best_sil_k, color='crimson', linestyle='--', linewidth=1.4)
    # Annotate away from the curve (top area, right of the marked line) so
    # the label never sits on top of a data point.
    y_top = silhouettes.max() + (silhouettes.max() - silhouettes.min()) * 0.12
    axes[1].annotate(f'$K^{{*}}={best_sil_k}$', xy=(best_sil_k, y_top),
                      xytext=(best_sil_k + 3.5, y_top), fontsize=13,
                      color='crimson', va='center',
                      arrowprops=dict(arrowstyle='-', color='crimson', linewidth=1.0))
    axes[1].set_ylim(top=y_top + (silhouettes.max() - silhouettes.min()) * 0.15)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nBest K by silhouette score (1000-UAV population): {best_sil_k}")
    print(f"Saved {out_path}")
    return best_sil_k


def cluster_snapshot(position_matrix, uav_ids, min_timesteps, optimal_k, out_path):
    t_sample = min_timesteps // 2
    pos_t = position_matrix[t_sample]
    pos_sc = StandardScaler().fit_transform(pos_t)
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=5)
    labels_sample = km.fit_predict(pos_sc)

    heads_sample = {}
    for cid in range(optimal_k):
        members = np.where(labels_sample == cid)[0]
        centroid = pos_t[members].mean(axis=0)
        dists = np.linalg.norm(pos_t[members] - centroid, axis=1)
        heads_sample[cid] = uav_ids[members[np.argmin(dists)]]

    palette = [
        '#E63946', '#06A77D', '#F4A261', '#7209B7', '#118AB2', '#073B4C',
        '#9D0208', '#FB8500', '#0077B6', '#6A040F',
    ]

    fig = plt.figure(figsize=(8.5, 5.2))
    ax = fig.add_axes([0.00, 0.02, 0.58, 0.94], projection='3d')
    z_floor = pos_t[:, 2].min() - 0.5

    for cid in range(optimal_k):
        mask = labels_sample == cid
        pts = pos_t[mask]
        color = palette[cid % len(palette)]
        ax.scatter(pts[:, 1], pts[:, 0], pts[:, 2], c=color, s=26, alpha=0.85,
                   edgecolors='black', linewidths=0.3, label=f'Cluster {cid}',
                   rasterized=True)
        if len(pts) >= 4:
            try:
                hull = ConvexHull(pts[:, [1, 0]])
                hull_pts = pts[hull.vertices][:, [1, 0]]
                verts = [list(zip(hull_pts[:, 0], hull_pts[:, 1], [z_floor] * len(hull_pts)))]
                poly = Poly3DCollection(verts, alpha=0.18, facecolor=color,
                                         edgecolor=color, linewidth=1.2)
                ax.add_collection3d(poly)
            except Exception:
                pass

    for cid, uav_global_idx in heads_sample.items():
        li = uav_ids.index(uav_global_idx)
        color = palette[cid % len(palette)]
        ax.scatter(pos_t[li, 1], pos_t[li, 0], pos_t[li, 2], c=color, s=260,
                   marker='*', edgecolors='gold', linewidths=1.8, zorder=20,
                   rasterized=True)

    ax.set_xlabel('Longitude', fontsize=13, labelpad=8)
    ax.set_ylabel('Latitude', fontsize=13, labelpad=8)
    ax.set_zlabel('Height (m)', fontsize=13, labelpad=8)
    ax.set_title(f'$t={t_sample}$, $K={optimal_k}$ ($N=1000$ UAVs)', fontsize=15, y=0.97)
    ax.view_init(elev=18, azim=-65)
    ax.grid(alpha=0.3)
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    ax.tick_params(labelsize=10)

    # Legend placed at a fixed figure-fraction position, just past the
    # axes rect's right edge (rect width 0.60) -- avoids relying on
    # Axes3D.transAxes, whose reported bounding box is much larger than
    # the visible cube and misplaces anything anchored to it.
    handles, labels = ax.get_legend_handles_labels()
    star_handle = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                              markeredgecolor='gold', markeredgewidth=1.5, markersize=14,
                              label='Cluster head')
    fig.legend(handles + [star_handle], labels + ['Cluster head'],
               loc='center left', bbox_to_anchor=(0.67, 0.5), fontsize=11,
               frameon=True, framealpha=0.95, borderaxespad=0.3)

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {out_path}")

    sizes = sorted(int((labels_sample == k).sum()) for k in range(optimal_k))
    print(f"Cluster sizes at snapshot t={t_sample}: {sizes}")


def main():
    position_matrix, uav_ids, min_timesteps, n_uavs = compute_position_matrix()

    optk_path = os.path.join(UAV_DIR, "optimal_k_selection_1000uav.pdf")
    best_sil_k = elbow_silhouette(position_matrix, n_uavs, min_timesteps, optk_path)

    snapshot_path = os.path.join(UAV_DIR, "3d_kmeans_cluster_snapshot_1000uav.pdf")
    cluster_snapshot(position_matrix, uav_ids, min_timesteps, best_sil_k, snapshot_path)

    print(f"\nOriginal (500-UAV) K*: 6")
    print(f"1000-UAV population K*: {best_sil_k}")
    if best_sil_k == 6:
        print("K*=6 CONFIRMED still optimal at 1000-UAV scale -- no change needed to the paper's K* claim.")
    else:
        print(f"K* CHANGED at 1000-UAV scale (6 -> {best_sil_k}) -- this affects the "
              f"paper's K*=6 claim and every downstream figure/CSV built with K=6. "
              f"build_cluster_csvs.py/build_ablation_csvs.py hardcode K=6 and would need updating.")


if __name__ == "__main__":
    main()
