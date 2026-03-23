# DUSTER

**D**edup-based **U**nified **St**orage and R**e**trieval for Point Clouds

DUSTER is a Python library that compresses sequential LiDAR point cloud data by exploiting cross-frame spatial redundancy. It identifies overlapping points across consecutive frames, deduplicates them into shared patches, and stores the result compactly — achieving ~13% better compression than per-frame methods like LASzip while maintaining high reconstruction fidelity.

## Key Features

- **Cross-frame deduplication** — Matches overlapping 3D points across consecutive frames via nearest-neighbor search, storing shared geometry once
- **CPU / GPU dual mode** — Seamlessly switches between sklearn KD-Tree (CPU) and faiss GpuIndexFlatL2 (GPU) for NN search
- **Hardware-aware scheduling** — Auto-detects CPU core count and GPU availability to set optimal parallelism (no manual tuning needed)
- **ILP + MAB segmentation** — Integer linear programming decides optimal segment lengths; multi-armed bandit with Gaussian smoothing assigns segments to processing blocks
- **LASzip integration** — Intensity values and remaining unmatched points compressed with LASzip for additional savings
- **Pip-installable** — `pip install -e .` with minimal dependencies; GPU packages are optional

## Installation

```bash
# CPU-only (no GPU required)
cd DUSTER/
pip install -e .

# With GPU acceleration (requires CUDA)
pip install -e ".[gpu]"

# With evaluation tools (requires PyTorch + OpenPCDet)
pip install -e ".[eval]"
```

### Dependencies

| Required | Optional |
|----------|----------|
| numpy, scipy, scikit-learn | pycuda, faiss-gpu (`[gpu]`) |
| laspy[laszip], pulp, joblib | torch, open3d (`[eval]`) |
| pymorton, bitarray, networkx | |

## Quick Start

```python
import numpy as np
from duster import Deduplicator, Segmentator, DiskManager, Indexer, SequenceIndexer
from duster.utils.scheduler import HardwareConfig
from duster.utils.logger import create_logger

# 1. Detect hardware and configure mode
hw = HardwareConfig(mode="auto")  # "auto" | "cpu" | "gpu"
print(hw.summary())
# Mode:        cpu
# CPUs:        20
# NN backend:  sklearn KD-Tree (n_jobs=-1)
# Block //:    10 workers (cpus=20)

# 2. Set up
logger = create_logger("duster.log")
velodyne_dir = "/path/to/sequences/00/velodyne"
pc_files = sorted(os.listdir(velodyne_dir))[:100]
# ... load poses, Tr from your dataset ...

# 3. Run segmentation + deduplication
segmentator = Segmentator(
    velodyne_dir, pc_files, queries, logger, poses, Tr,
    threshold=0.03, args=args, max_cluster_length=3, A=4,
    hw_config=hw,
)
segment_counts, seq_indexer = segmentator.segment_with_ip_mab(
    pc_files, time_budget=0, max_seg_length=3
)

# 4. Query reconstructed frames
pc_data, io_cost, latency = seq_indexer.random_search(frame_id=42)
pc_frames, io_cost, latency = seq_indexer.sequential_search([0, 10])
```

## How It Works

### 1. Segmentation

The input sequence of N frames is partitioned into segments of varying lengths. DUSTER uses a two-phase approach:

1. **ILP scoring** — For each candidate segment length (1, 2, 3, ...), an Integer Linear Programming solver estimates the compression benefit vs. processing cost. Longer segments have more cross-frame overlap but take longer to process.

2. **Multi-Armed Bandit (MAB) assignment** — Segments are assigned to processing blocks using a MAB strategy with Gaussian-smoothed scores. This balances load across parallel workers while maximizing total compression.

The result is a set of segments (e.g., `{length=1: 1 segment, length=3: 33 segments}` for 100 frames) that are then processed independently in parallel.

### 2. Deduplication

For each segment, DUSTER performs two-pass cross-frame matching:

**Pass 1 (Relative coordinates):** For each subset of frames ordered by the clustering algorithm:
- Select an anchor frame (middle element of the subset)
- For every other frame, build a KD-Tree (or faiss GPU index) on its points
- Query anchor points against each frame's index to find k=10 nearest neighbors
- Greedily match points satisfying: per-pair distance ≤ α × threshold AND running average distance ≤ threshold
- Matched points are stored as **patches** — shared (x,y,z) with per-frame intensity values
- Matched points are removed from their source frames

**Pass 2 (Absolute coordinates):** The remaining unmatched points are transformed to a global coordinate system using the provided poses, then the same matching process is repeated to catch additional overlaps.

**Result:** Each segment produces:
- Deduplicated patches (shared xyz + per-frame intensities) → compressed with LASzip
- Remaining unmatched points per frame → compressed with LASzip

### 3. Hardware Scheduling

DUSTER's `HardwareConfig` auto-detects available resources and configures parallelism:

| Parameter | CPU Mode | GPU Mode |
|-----------|----------|----------|
| NN backend | sklearn KD-Tree | faiss GpuIndexFlatL2 |
| NN threading | n_jobs=-1 (all cores) | Single-threaded (GPU handles it) |
| Block parallelism | cpu_count / 2 | gpu_count × 4 |
| Worker parallelism | cpu_count / 2 | gpu_count × 4 |
| GPU ID cycling | N/A | Round-robin across GPUs |

The mode can be set explicitly:
```python
hw = HardwareConfig(mode="cpu")   # Force CPU even if GPU available
hw = HardwareConfig(mode="gpu")   # Force GPU (raises error if unavailable)
hw = HardwareConfig(mode="auto")  # Auto-detect (default)
```

Child processes inherit the parent's mode via the `DUSTER_MODE` environment variable.

### 4. Why KD-Tree for CPU Mode?

DUSTER operates on 3D point clouds (~120K points per frame, k=10 neighbors). In this regime:
- **KD-Tree** is optimal: exact results, O(log N) query time, no tuning needed
- **ANN methods** (HNSW, NNDescent) are designed for high-dimensional data (50D+) and have unnecessary overhead in 3D
- **faiss GPU** is faster when GPUs are available, but KD-Tree on CPU is a strong alternative with zero GPU dependency

## Benchmark Results

100 frames from SemanticKITTI sequence 00 (2× RTX 2080 Ti, 20 CPU cores):

| Metric | GPU | CPU | LASzip |
|--------|-----|-----|--------|
| Size (MB) | 37.9 | 38.1 | 44.0 |
| % of raw | 20.5% | 20.6% | 23.8% |
| Build time (s) | 100.7 | 80.4 | 4.0 |
| Avg Chamfer dist | 0.0086 | 0.0082 | 0 (lossless) |
| Dedup % | 63.3% | 64.1% | N/A |
| Query latency (s) | 0.088 | 0.082 | N/A |

Key takeaways:
- DUSTER saves **~13-14%** over per-frame LASzip compression
- CPU mode (KD-Tree) is **faster** than GPU mode for 3D point cloud data due to O(log N) queries and higher process-level parallelism
- Reconstruction is near-lossless (Chamfer distance ~0.008, point count exactly preserved)

## Project Structure

```
DUSTER/
├── duster/                          # Pip-installable Python package
│   ├── __init__.py                  # Public API + __version__
│   ├── core/
│   │   ├── deduplicator.py          # Cross-frame point matching & deduplication
│   │   ├── disk_manager.py          # LASzip-based serialization to disk
│   │   └── indexer.py               # In-memory index for frame reconstruction
│   ├── segmentation/
│   │   └── segmentator.py           # ILP + MAB segment length optimization
│   ├── compression/
│   │   └── octree.py                # Octree-based spatial compression
│   ├── evaluation/
│   │   └── evaluator.py             # Detection accuracy evaluation (optional, needs PyTorch)
│   ├── query/
│   │   └── loader.py                # Query file loading utilities
│   ├── baselines/
│   │   ├── laz_index.py             # LASzip per-frame baseline
│   │   ├── raw_index.py             # Raw .bin file baseline
│   │   └── octree_index.py          # Octree compression baseline
│   └── utils/
│       ├── logger.py                # Logging setup
│       └── scheduler.py             # Hardware detection & parallelism config
├── configs/config.py                # CLI argument definitions
├── benchmark_exp.py                 # Full benchmark runner
├── test_e2e.py                      # End-to-end test script
├── tests/test_imports.py            # Import smoke test
├── pyproject.toml                   # Package metadata
└── README.md
```

## Datasets

- **SemanticKITTI**: http://semantic-kitti.org/dataset.html#format
- **ONCE**: https://once-for-auto-driving.github.io/download.html#downloads

## Running Benchmarks

```bash
# Run DUSTER on SemanticKITTI sequence 00
python benchmark_exp.py \
  --method duster \
  --data_path semantic_kitti \
  --sequence_id 00 \
  --distance_threshold 0.03 \
  --max_cluster_length 3 \
  --time_budget 0.25

# Run octree baseline
python benchmark_exp.py \
  --method octree \
  --data_path semantic_kitti \
  --sequence_id 00

# Run LASzip baseline
python benchmark_exp.py --method laz --data_path semantic_kitti --sequence_id 00

# End-to-end test with hardware mode selection
python test_e2e.py --mode auto
python test_e2e.py --mode cpu
python test_e2e.py --mode gpu
```

## License

See LICENSE file for details.
