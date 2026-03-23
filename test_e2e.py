"""
End-to-end test of the duster library on SemanticKITTI sequence 00 (first 100 frames).
Tests: segmentation -> deduplication -> disk storage -> retrieval -> accuracy check.
"""
import os
import sys
import time
import numpy as np
import shutil
import argparse
import laspy
from laspy.compression import LazBackend
from io import BytesIO

# Ensure DUSTER root is on path for configs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from duster.core.deduplicator import Deduplicator
from duster.core.disk_manager import DiskManager
from duster.core.indexer import Indexer, SequenceIndexer
from duster.segmentation.segmentator import Segmentator
from duster.utils.logger import create_logger
from duster.utils.scheduler import HardwareConfig, get_config

# ── Config ──────────────────────────────────────────────
NUM_FRAMES = 100
SEQUENCE_ID = "00"
BASE_PATH = f"/home/user1/jiangneng/pcdb/OpenPCDet/data/semantic_kitti/dataset/sequences/{SEQUENCE_ID}"
STORE_PATH = "/tmp/duster_test_store"
LOG_FILE = "/tmp/duster_test.log"


def get_transformation_matrix(calib_file):
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith("Tr:"):
                values = list(map(float, line.split("Tr:")[1].strip().split()))
                tr_matrix = np.array(values).reshape(3, 4)
                tr_matrix = np.vstack((tr_matrix, [0, 0, 0, 1]))
                return tr_matrix
    raise ValueError("Tr matrix not found in the calibration file.")


class FakeArgs:
    """Mimics the argparse namespace from configs/config.py"""
    def __init__(self):
        self.distance_threshold = 0.03
        self.max_cluster_length = 3
        self.relative_absolute = 'both'
        self.deduplicate_intensity = False
        self.intensity_threshold = 0.01
        self.intensity_scale = 32767.0
        self.soft_distance_ratio = 2.0
        self.workload_aware = False
        self.root_store_path = STORE_PATH
        self.root_store_path_wa = STORE_PATH + "_wa"
        self.same_segment = True
        self.time_budget = 0.25
        self.ext = '.bin'
def get_total_memory_used(directory):
    total_size = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path) / 1024
    return total_size


def compute_chamfer_distance(pc_original, pc_reconstructed):
    """Compute one-directional Chamfer distance (original -> reconstructed)."""
    from sklearn.neighbors import NearestNeighbors
    if pc_original.shape[0] == 0 or pc_reconstructed.shape[0] == 0:
        return float('inf')
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
    nn.fit(pc_reconstructed[:, :3])
    distances, _ = nn.kneighbors(pc_original[:, :3])
    return float(np.mean(distances))


def run_test(mode="auto"):
    hw = get_config(mode, force_new=True)

    logger = create_logger(LOG_FILE)
    logger.info(f"=== DUSTER E2E Test: {NUM_FRAMES} frames, mode={hw.mode} ===")
    logger.info(f"Hardware config:\n{hw.summary()}")

    # ── Load data ──
    velodyne_dir = os.path.join(BASE_PATH, "velodyne")
    poses_file = os.path.join(BASE_PATH, "poses.txt")
    calib_file = os.path.join(BASE_PATH, "calib.txt")

    with open(poses_file, 'r') as f:
        all_poses = [list(map(float, line.strip().split())) for line in f]

    Tr = get_transformation_matrix(calib_file)

    pc_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])[:NUM_FRAMES]
    poses = all_poses[:NUM_FRAMES]

    logger.info(f"Loaded {len(pc_files)} frames, {len(poses)} poses")

    args = FakeArgs()

    # Compute original data size
    original_size_kb = sum(
        os.path.getsize(os.path.join(velodyne_dir, f)) / 1024 for f in pc_files
    )
    logger.info(f"Original data size: {original_size_kb:.1f} KB ({original_size_kb/1024:.2f} MB)")

    # ── Laszip baseline: compress each frame individually ──
    logger.info("--- Laszip baseline ---")
    laz_store = "/tmp/duster_test_laz"
    if os.path.exists(laz_store):
        shutil.rmtree(laz_store)
    os.makedirs(laz_store)

    t_laz_start = time.time()
    for pc_file in pc_files:
        file_path = os.path.join(velodyne_dir, pc_file)
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

        header = laspy.LasHeader(point_format=3, version="1.2")
        header.x_scale, header.y_scale, header.z_scale = 0.0001, 0.0001, 0.0001

        las = laspy.LasData(header)
        las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
        las.intensity = (points[:, 3] * args.intensity_scale).astype(np.uint16)

        laz_path = os.path.join(laz_store, pc_file.replace(".bin", ".laz"))
        with laspy.open(laz_path, mode="w", header=las.header, laz_backend=LazBackend.Laszip) as writer:
            writer.write_points(las.points)
    t_laz = time.time() - t_laz_start

    laz_size_kb = get_total_memory_used(laz_store)
    laz_ratio = laz_size_kb / original_size_kb if original_size_kb > 0 else 0
    logger.info(f"Laszip size:     {laz_size_kb:.1f} KB ({laz_size_kb/1024:.2f} MB)")
    logger.info(f"Laszip ratio:    {laz_ratio*100:.2f}% of raw")
    logger.info(f"Laszip time:     {t_laz:.2f}s")

    # ── Clean store ──
    if os.path.exists(STORE_PATH):
        shutil.rmtree(STORE_PATH)
    os.makedirs(STORE_PATH)

    # ── Generate queries for evaluation ──
    random_queries = list(range(0, NUM_FRAMES, 10))  # every 10th frame
    sequential_queries = [[0, 10], [50, 60]]
    queries = {"random": random_queries, "sequential": sequential_queries}

    # ── Step 1: Segmentation + Deduplication ──
    logger.info("--- Step 1: Segmentation ---")
    t0 = time.time()

    segmentator = Segmentator(
        velodyne_dir, pc_files, queries, logger, poses, Tr,
        args.distance_threshold, args, args.max_cluster_length, 4,
        max_processes=6, exploration_param=0.1, hw_config=hw
    )
    segment_counts, sequence_indexer = segmentator.segment_with_ip_mab(pc_files, 0, args.max_cluster_length)

    t_construct = time.time() - t0
    logger.info(f"Segmentation + dedup done in {t_construct:.1f}s")
    logger.info(f"Segment counts: {segment_counts}")

    # ── Step 2: Measure compressed size ──
    compressed_size_kb = get_total_memory_used(STORE_PATH)
    compression_ratio = compressed_size_kb / original_size_kb if original_size_kb > 0 else 0
    logger.info(f"Compressed size: {compressed_size_kb:.1f} KB ({compressed_size_kb/1024:.2f} MB)")
    logger.info(f"Compression ratio: {compression_ratio:.4f} ({compression_ratio*100:.2f}%)")
    # ── Step 3: Retrieval accuracy (random queries) ──
    logger.info("--- Step 3: Retrieval accuracy ---")
    chamfer_distances = []
    io_costs = []
    load_times = []

    for query_id in random_queries:
        # Retrieve from compressed store
        pc_reconstructed, io_cost, load_time = sequence_indexer.random_search(query_id)

        # Load original
        original_path = os.path.join(velodyne_dir, pc_files[query_id])
        pc_original = np.fromfile(original_path, dtype=np.float32).reshape(-1, 4)

        # Compute Chamfer distance (xyz only)
        cd = compute_chamfer_distance(pc_original, pc_reconstructed)
        chamfer_distances.append(cd)
        io_costs.append(io_cost)
        load_times.append(load_time)

        logger.info(f"  Frame {query_id}: points orig={pc_original.shape[0]} recon={pc_reconstructed.shape[0]}, "
                     f"chamfer={cd:.6f}, io={io_cost:.1f}KB, time={load_time:.4f}s")

    # ── Step 4: Sequential retrieval ──
    logger.info("--- Step 4: Sequential retrieval ---")
    seq_io_costs = []
    seq_load_times = []

    for seq_query in sequential_queries:
        pc_frames, io_cost, load_time = sequence_indexer.sequential_search(seq_query)
        seq_io_costs.append(io_cost)
        seq_load_times.append(load_time)
        logger.info(f"  Range {seq_query}: {len(pc_frames)} frames, io={io_cost:.1f}KB, time={load_time:.4f}s")

    # ── Summary ──
    logger.info("=" * 60)
    logger.info(f"RESULTS SUMMARY  [{hw.mode.upper()} mode]")
    logger.info("=" * 60)
    logger.info(f"Frames:              {NUM_FRAMES}")
    logger.info(f"Mode:                {hw.mode} (block//={hw.block_parallel}, workers={hw.worker_parallel})")
    logger.info(f"Segment counts:      {segment_counts}")
    logger.info(f"")
    logger.info(f"--- Storage ---")
    logger.info(f"Raw .bin size:       {original_size_kb/1024:.2f} MB")
    logger.info(f"Laszip size:         {laz_size_kb/1024:.2f} MB  ({laz_ratio*100:.2f}% of raw)")
    logger.info(f"DUSTER size:         {compressed_size_kb/1024:.2f} MB  ({compression_ratio*100:.2f}% of raw)")
    duster_vs_laz = compressed_size_kb / laz_size_kb if laz_size_kb > 0 else 0
    logger.info(f"DUSTER vs Laszip:    {duster_vs_laz*100:.2f}%  (saving {(1-duster_vs_laz)*100:.1f}% over laszip)")
    logger.info(f"")
    logger.info(f"--- Time ---")
    logger.info(f"Laszip build time:   {t_laz:.2f}s")
    logger.info(f"DUSTER build time:   {t_construct:.2f}s")
    logger.info(f"")
    logger.info(f"--- Retrieval Quality ---")
    logger.info(f"Avg Chamfer dist:    {np.mean(chamfer_distances):.6f}")
    logger.info(f"Max Chamfer dist:    {np.max(chamfer_distances):.6f}")
    dedup_pct = sequence_indexer.calculate_deduplicated_percentage(velodyne_dir, STORE_PATH)
    logger.info(f"Dedup percentage:    {dedup_pct:.4f} ({dedup_pct*100:.2f}%)")
    logger.info(f"")
    logger.info(f"--- Retrieval Latency ---")
    logger.info(f"Avg random IO:       {np.mean(io_costs):.1f} KB")
    logger.info(f"Avg random latency:  {np.mean(load_times):.4f}s")
    logger.info(f"Avg seq IO:          {np.mean(seq_io_costs):.1f} KB")
    logger.info(f"Avg seq latency:     {np.mean(seq_load_times):.4f}s")
    logger.info("=" * 60)

    # cleanup laz baseline
    shutil.rmtree(laz_store, ignore_errors=True)

    print(f"\n✓ E2E test [{hw.mode}] completed. See log: " + LOG_FILE)

    return {
        "mode": hw.mode,
        "raw_mb": original_size_kb / 1024,
        "laz_mb": laz_size_kb / 1024,
        "duster_mb": compressed_size_kb / 1024,
        "laz_ratio": laz_ratio,
        "duster_ratio": compression_ratio,
        "duster_vs_laz": compressed_size_kb / laz_size_kb if laz_size_kb > 0 else 0,
        "build_time": t_construct,
        "laz_time": t_laz,
        "avg_chamfer": float(np.mean(chamfer_distances)),
        "max_chamfer": float(np.max(chamfer_distances)),
        "dedup_pct": dedup_pct,
        "avg_random_io_kb": float(np.mean(io_costs)),
        "avg_random_latency": float(np.mean(load_times)),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cpu", "gpu", "both"], default="both",
                        help="cpu, gpu, or both (run both and compare)")
    cli_args = parser.parse_args()

    if cli_args.mode == "both":
        results = {}
        for m in ["gpu", "cpu"]:
            try:
                results[m] = run_test(m)
            except Exception as e:
                print(f"[{m}] FAILED: {e}")

        if len(results) >= 2:
            print("\n" + "=" * 65)
            print(f"{'COMPARISON':^65}")
            print("=" * 65)
            print(f"{'Metric':<25} {'GPU':>12} {'CPU':>12} {'Laszip':>12}")
            print("-" * 65)
            r_gpu, r_cpu = results["gpu"], results["cpu"]
            print(f"{'Size (MB)':<25} {r_gpu['duster_mb']:>12.2f} {r_cpu['duster_mb']:>12.2f} {r_gpu['laz_mb']:>12.2f}")
            print(f"{'% of raw':<25} {r_gpu['duster_ratio']*100:>11.2f}% {r_cpu['duster_ratio']*100:>11.2f}% {r_gpu['laz_ratio']*100:>11.2f}%")
            print(f"{'Build time (s)':<25} {r_gpu['build_time']:>12.1f} {r_cpu['build_time']:>12.1f} {r_gpu['laz_time']:>12.1f}")
            print(f"{'Avg Chamfer':<25} {r_gpu['avg_chamfer']:>12.6f} {r_cpu['avg_chamfer']:>12.6f} {'0 (lossless)':>12}")
            print(f"{'Dedup %':<25} {r_gpu['dedup_pct']*100:>11.2f}% {r_cpu['dedup_pct']*100:>11.2f}% {'N/A':>12}")
            print(f"{'Avg query IO (KB)':<25} {r_gpu['avg_random_io_kb']:>12.1f} {r_cpu['avg_random_io_kb']:>12.1f} {'N/A':>12}")
            print(f"{'Avg query latency (s)':<25} {r_gpu['avg_random_latency']:>12.4f} {r_cpu['avg_random_latency']:>12.4f} {'N/A':>12}")
            print("=" * 65)
    else:
        run_test(cli_args.mode)
