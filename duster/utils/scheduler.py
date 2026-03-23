"""
Hardware-aware scheduler for DUSTER.
Auto-detects CPU/GPU resources and provides optimal parallelism parameters.
"""
import os


def _detect_gpu_info():
    """Detect GPU count and per-GPU free memory."""
    try:
        import pycuda.driver as cuda
        cuda.init()
        count = cuda.Device.count()
        gpus = []
        for i in range(count):
            dev = cuda.Device(i)
            ctx = dev.make_context()
            free, total = cuda.mem_get_info()
            ctx.pop()
            gpus.append({"id": i, "name": dev.name(), "free_mb": free / (1024 ** 2), "total_mb": total / (1024 ** 2)})
        return gpus
    except Exception:
        return []


def _detect_faiss_gpu():
    try:
        import faiss
        return hasattr(faiss, 'StandardGpuResources')
    except ImportError:
        return False


class HardwareConfig:
    """Immutable snapshot of detected hardware capabilities."""

    def __init__(self, mode="auto"):
        """
        Args:
            mode: "auto" | "cpu" | "gpu"
                - auto: use GPU if available, else CPU
                - cpu:  force CPU even if GPU exists
                - gpu:  force GPU (raises if unavailable)
        """
        self.cpu_count = os.cpu_count() or 4
        self.gpus = _detect_gpu_info()
        self.gpu_count = len(self.gpus)
        self.has_faiss_gpu = _detect_faiss_gpu()

        # Resolve mode
        if mode == "auto":
            self.use_gpu = self.gpu_count > 0 and self.has_faiss_gpu
        elif mode == "gpu":
            if self.gpu_count == 0 or not self.has_faiss_gpu:
                raise RuntimeError("GPU mode requested but no GPU/faiss-gpu found")
            self.use_gpu = True
        else:
            self.use_gpu = False

        self.mode = "gpu" if self.use_gpu else "cpu"

        # Set env var so child processes (ProcessPoolExecutor) inherit the mode
        os.environ["DUSTER_MODE"] = self.mode

    # ── Parallelism parameters ──
    # CPU mode: ProcessPoolExecutor workers limited, but each worker's KD-Tree
    # uses n_jobs=-1 internally (numpy/sklearn release GIL for C-level work,
    # so threads within a single process are fine).

    @property
    def nn_backend(self):
        """Which NN search backend to use: 'faiss_gpu' or 'kdtree'."""
        return "faiss_gpu" if self.use_gpu else "kdtree"

    @property
    def nn_n_jobs(self):
        """n_jobs for sklearn KD-Tree inside each worker subprocess."""
        if self.use_gpu:
            return 1
        return -1  # KD-Tree n_jobs=-1 is fine; GIL is released in C code

    @property
    def block_parallel(self):
        """How many blocks to evaluate in parallel during segmentation."""
        if self.use_gpu:
            return self.gpu_count * 4
        else:
            return max(4, self.cpu_count // 2)

    @property
    def worker_parallel(self):
        """Max workers for ProcessPoolExecutor (short segment processing)."""
        if self.use_gpu:
            return self.gpu_count * 4
        else:
            return max(4, self.cpu_count // 2)

    @property
    def gpu_ids(self):
        """List of GPU IDs to cycle through, or [None] for CPU mode."""
        if self.use_gpu:
            return [g["id"] for g in self.gpus]
        else:
            return [None]

    def summary(self):
        lines = [
            f"Mode:        {self.mode}",
            f"CPUs:        {self.cpu_count}",
            f"GPUs:        {self.gpu_count}",
        ]
        if self.use_gpu:
            for g in self.gpus:
                lines.append(f"  GPU {g['id']}: {g['name']} ({g['free_mb']:.0f}/{g['total_mb']:.0f} MB)")
            lines.append(f"NN backend:  faiss_gpu")
            lines.append(f"Block //:    {self.block_parallel} (GPU×6)")
            lines.append(f"Worker //:   {self.worker_parallel}")
        else:
            lines.append(f"NN backend:  sklearn KD-Tree (n_jobs=-1)")
            lines.append(f"Block //:    {self.block_parallel} workers (cpus={self.cpu_count})")
            lines.append(f"Worker //:   {self.worker_parallel}")
        return "\n".join(lines)


# Module-level singleton — lazy init
_default_config = None


def get_config(mode="auto", force_new=False):
    """Get or create the global HardwareConfig.
    In child processes, reads DUSTER_MODE env var to inherit parent's choice.
    """
    global _default_config
    if _default_config is None or force_new:
        env_mode = os.environ.get("DUSTER_MODE")
        if env_mode and mode == "auto":
            mode = env_mode
        _default_config = HardwareConfig(mode)
    return _default_config
