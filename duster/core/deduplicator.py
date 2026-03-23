'''
The function class for deduplication with input PC data.
Supports both GPU (faiss) and CPU (sklearn KD-Tree) backends via HardwareConfig.
'''
try:
    import pycuda.driver as cuda
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

try:
    import faiss
    HAS_FAISS_GPU = hasattr(faiss, 'StandardGpuResources')
except ImportError:
    HAS_FAISS_GPU = False

import numpy as np
import pulp
from sklearn.neighbors import NearestNeighbors
import time
import math
import bisect
import os


class Deduplicator:
    """Handles point cloud deduplication across overlapping LiDAR frames."""

    def __init__(self, args, hw_config=None):
        """Initialize with CLI args and optional hardware configuration."""
        self.args = args
        # Lazy import to avoid circular deps at module level
        if hw_config is None:
            from duster.utils.scheduler import get_config
            hw_config = get_config()
        self.hw = hw_config

    def euclidean_distance_3d(self, a, b):
        """
        Compute Euclidean distance in 3D for the first three coordinates of points a and b.
        Each point is assumed to be [x, y, z, intensity].
        """
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def deduplication(self, PC_data, poses, Tr, threshold, gpuid=None):
        """
        Main deduplication entry point. Returns a dictionary with:
          - 'matched_points': total number of matched points (across both passes),
          - 'deduplicated_patches': a list of "patches", each containing
              { 'xyz': (x,y,z), 'intensities': [i0, i1, ...] }
        """
        # Set GPU device if in GPU mode
        if self.hw.use_gpu and gpuid is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

        # 1) Build the clustering order of subsets + anchors
        clustering_order = self.generate_cluster_ordering(PC_data)

        # 2) Conduct the clustering (Pass 1: no transposition)
        deduplicated_patches = {}

        if self.args.relative_absolute == 'absolute':
            matched_points_pass1 = 0
        else:
            matched_points_pass1 = self.match_frames(
                PC_data, clustering_order, threshold, deduplicated_patches, poses, Tr, gpuid=gpuid
            )

        # 3) Apply poses to transpose the unmatched points
        self.transpose_PC_data(PC_data, poses, Tr)

        # 4) Conduct the clustering again (Pass 2: after transposition)
        if self.args.relative_absolute == 'relative':
            matched_points_pass2 = 0
        else:
            matched_points_pass2 = self.match_frames(
                PC_data, clustering_order, threshold, deduplicated_patches, poses, Tr, transposed=True, gpuid=gpuid
            )

        # Store remaining unmatched points for this pass.
        self.store_remaining_points(PC_data, deduplicated_patches, poses, Tr, transposed=True)

        return {
            "matched_points": matched_points_pass1 + matched_points_pass2,
            "deduplicated_patches": deduplicated_patches
        }

    def solve_nn_matching(self, anchor_points, candidate_points, k=10):
        """
        NN search: faiss GPU (squared L2 -> sqrt) or sklearn KD-Tree (euclidean).
        Returns euclidean distances and indices.
        """
        if self.hw.nn_backend == "faiss_gpu":
            dim = 3
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, dim)
            index.add(candidate_points[:, :3].astype(np.float32))
            distances, indices = index.search(anchor_points[:, :3].astype(np.float32), k)
            distances = np.sqrt(distances)  # faiss returns squared L2
            return distances, indices
        else:
            nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', metric='euclidean', n_jobs=self.hw.nn_n_jobs)
            nn.fit(candidate_points[:, :3].astype(np.float32))
            distances, indices = nn.kneighbors(anchor_points[:, :3].astype(np.float32))
            return distances, indices

    def solve_greedy_matching(self, anchor_points, candidate_points, threshold, alpha=2.0, k=10):
        """
        Greedy matching using NN search (GPU or CPU).
        """
        m = candidate_points.shape[0]

        st_time = time.time()

        if self.hw.nn_backend == "faiss_gpu":
            dim = 3
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, dim)
            index.add(anchor_points[:, :3].astype(np.float32))
            distances, indices = index.search(candidate_points[:, :3].astype(np.float32), k)
            distances = np.sqrt(distances)
        else:
            nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', metric='euclidean', n_jobs=self.hw.nn_n_jobs)
            nn.fit(anchor_points[:, :3].astype(np.float32))
            distances, indices = nn.kneighbors(candidate_points[:, :3].astype(np.float32))

        valid_matches = []

        # Collect all valid matches
        for i in range(m):
            for j_pos in range(k):
                j = indices[i][j_pos]
                valid_matches.append((i, j, distances[i][j_pos]))
        st_time = time.time()
        valid_matches.sort(key=lambda x: x[2])

        st_time = time.time()
        # Filter out matches with distance larger than alpha * threshold
        threshold_distance = alpha * threshold
        idx = bisect.bisect_right([match[2] for match in valid_matches], threshold_distance)
        valid_matches = valid_matches[:idx]

        used_anchors = set()
        used_candidates = set()
        total_dist = 0.0
        result = []

        st_time = time.time()

        for i, j, dist in valid_matches:
            if i in used_anchors or j in used_candidates:
                continue
            if len(result) > 0:
                if dist < threshold or (total_dist + dist) / (len(result) + 1) <= threshold:
                    result.append((i, j))
                    used_anchors.add(i)
                    used_candidates.add(j)
                    total_dist += dist
            else:
                result.append((i, j))
                used_anchors.add(i)
                used_candidates.add(j)
                total_dist += dist

        return [(j, i) for i, j in result]

    def solve_ilp_matching(self, anchor_points, candidate_points, threshold, alpha=2.0):
        """
        ILP matching with:
        - Hard constraint: no match has distance > alpha * threshold
        - Soft constraint: average distance <= threshold
        """

        n = anchor_points.shape[0]
        m = candidate_points.shape[0]

        # Step 1: collect valid pairs within alpha*threshold
        valid_pairs = []
        distances = {}
        for i in range(n):
            for j in range(m):
                dist = np.linalg.norm(anchor_points[i, :3] - candidate_points[j, :3])
                if dist <= alpha * threshold:
                    valid_pairs.append((i, j))
                    distances[(i, j)] = dist

        # Step 2: setup ILP
        prob = pulp.LpProblem("SoftHardThresholdMatching", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", valid_pairs, cat="Binary")

        # Objective: minimize total distance
        prob += pulp.lpSum(x[i, j] * distances[i, j] for (i, j) in valid_pairs)

        # Each anchor matched at most once
        for i in range(n):
            prob += pulp.lpSum(x[i, j] for j in range(m) if (i, j) in x) <= 1

        # Each candidate matched at most once
        for j in range(m):
            prob += pulp.lpSum(x[i, j] for i in range(n) if (i, j) in x) <= 1

        # Soft global constraint: total dist / #match <= threshold
        total_matches = pulp.lpSum(x[i, j] for (i, j) in valid_pairs)
        total_dist = pulp.lpSum(x[i, j] * distances[i, j] for (i, j) in valid_pairs)
        prob += total_dist <= threshold * total_matches

        # Solve
        prob.solve(pulp.PulpSolverDefault)

        # Extract matches
        result = [(i, j) for (i, j) in valid_pairs if pulp.value(x[i, j]) > 0.5]
        return result

    def generate_related_subsets(self, subset, anchor):
        """Return all contiguous sub-subsets of `subset` that contain `anchor`, largest first."""
        N = len(subset)
        related_subsets = []
        anchor_index = subset.index(anchor)

        for size in range(N - 1, 1, -1):
            for start in range(max(0, anchor_index - size + 1), min(N - size + 1, anchor_index + 1)):
                sub_subset = subset[start: start + size]
                if anchor in sub_subset:
                    related_subsets.append(list(sub_subset))

        return related_subsets

    def generate_anchor(self, related_subset):
        """
        Generate the anchor(s) for a given related_subset.
        """
        size = len(related_subset)
        if size == 0:
            raise ValueError("The related_subset cannot be empty.")

        if size % 2 == 1:
            return [related_subset[size // 2]]
        else:
            return [related_subset[(size // 2) - 1], related_subset[size // 2]]

    def match_frames(self, PC_data, clustering_order, threshold, deduplicated_patches, poses, Tr, transposed=False,
                     gpuid=None):
        """
        Optimized matching using NN search (GPU or CPU).
        For each subset (from clustering_order), the anchor frame is used as reference.
        """
        matched_points_count = 0
        matched_subset_mark = {tuple(subset): False for subset, _ in clustering_order}

        for subset, anchor_idx in clustering_order:
            if matched_subset_mark[tuple(subset)]:
                continue

            anchor_points = np.array(PC_data[anchor_idx])
            if anchor_points.shape[0] == 0:
                continue

            anchor_matches = {i: [] for i in range(anchor_points.shape[0])}

            # Step 1: Collect distances and indices for all candidate point clouds
            all_distances = []
            all_indices = []

            for frame_idx in subset:
                if frame_idx == anchor_idx:
                    continue
                candidate_points = np.array(PC_data[frame_idx])
                if candidate_points.shape[0] == 0:
                    continue

                distances, indices = self.solve_nn_matching(anchor_points, candidate_points, k=10)

                all_distances.append(distances)
                all_indices.append(indices)

            # Step 2: Compute the sum of top-1 distances for each anchor point
            distance_sums = []
            for anc_idx in range(anchor_points.shape[0]):
                top_1_sum = sum(distances[anc_idx, 0] for distances in all_distances)
                distance_sums.append((anc_idx, top_1_sum))

            # Step 3: Sort anchor points by the distance sums
            distance_sums.sort(key=lambda x: x[1])

            # Step 4: Generate anchor_matches based on the sorted anchor points
            used_anchors = set()
            used_candidates = set()
            anchor_frame_idx = subset.index(anchor_idx)

            total_dist = 0.0
            match_count = 0

            alpha = self.args.soft_distance_ratio
            for anc_idx, _ in distance_sums:
                for candidate_idx, distances in enumerate(all_distances):
                    if candidate_idx < anchor_frame_idx:
                        frame_idx = subset[candidate_idx]
                    else:
                        frame_idx = subset[candidate_idx + 1]

                    for rank in range(distances.shape[1]):
                        cand_idx = all_indices[candidate_idx][anc_idx, rank]
                        distance = distances[anc_idx, rank]

                        if (
                                distance <= alpha * threshold and
                                (candidate_idx, cand_idx) not in used_candidates and
                                (match_count == 0 or (total_dist + distance) / (match_count + 1) <= threshold)
                        ):
                            anchor_matches[anc_idx].append((frame_idx, cand_idx))
                            used_candidates.add((candidate_idx, cand_idx))
                            total_dist += distance
                            match_count += 1
                            break

            delete_indices = {frame_idx: [False for _ in range(len(PC_data[frame_idx]))] for frame_idx in
                              range(len(PC_data))}
            same_intensity_count = 0
            keys = []
            related_subsets = self.generate_related_subsets(subset, anchor_idx)

            deduplicated_related_subsets = []
            for anc_idx, matches in anchor_matches.items():
                if len(matches) == len(subset) - 1:
                    patch = {
                        "xyz": (anchor_points[anc_idx, :3]).tolist(),
                        "intensities": [None] * len(subset)
                    }
                    patch["intensities"][subset.index(anchor_idx)] = anchor_points[anc_idx, 3]

                    same_intensity = 0
                    for frame_idx, cand_idx in matches:
                        patch["intensities"][subset.index(frame_idx)] = PC_data[frame_idx][cand_idx][3]
                        if abs(PC_data[frame_idx][cand_idx][3] - anchor_points[
                            anc_idx, 3]) < self.args.intensity_threshold:
                            same_intensity += 1
                    if self.args.deduplicate_intensity:
                        if same_intensity == len(subset) - 1:
                            same_intensity_count += 1
                            key = (tuple(subset), anchor_idx, True, transposed)
                            if key not in keys:
                                keys.append(key)
                            if key not in deduplicated_patches:
                                deduplicated_patches[key] = []
                            patch["intensities"] = [anchor_points[anc_idx, 3]]
                            deduplicated_patches[key].append(patch)
                        else:
                            key = (tuple(subset), anchor_idx, False, transposed)
                            if key not in keys:
                                keys.append(key)
                            if key not in deduplicated_patches:
                                deduplicated_patches[key] = []
                            deduplicated_patches[key].append(patch)
                    else:
                        key = (tuple(subset), anchor_idx, False, transposed)
                        if key not in keys:
                            keys.append(key)
                        if key not in deduplicated_patches:
                            deduplicated_patches[key] = []
                        deduplicated_patches[key].append(patch)

                    matched_points_count += 1
                    for frame_idx, cand_idx in matches:
                        delete_indices[frame_idx][cand_idx] = True
                    delete_indices[anchor_idx][anc_idx] = True
                else:
                    continue
            for deduplicated_related_subset in deduplicated_related_subsets:
                matched_subset_mark[tuple(deduplicated_related_subset)] = True

            for key in keys:
                patches = {
                    "xyz": None,
                    "intensities": []
                }
                xyz_matrix = np.array([patch["xyz"] for patch in deduplicated_patches[key]])
                patches["xyz"] = xyz_matrix
                for i in range(len(deduplicated_patches[key][0]["intensities"])):
                    intensity = np.array([patch["intensities"][i] for patch in deduplicated_patches[key]])
                    patches["intensities"].append(intensity)
                deduplicated_patches[key] = patches

            if transposed and len(keys) > 0:
                inv_Tr = np.linalg.inv(Tr)
                dof = poses[anchor_idx]
                rotation = np.array([dof[0:3], dof[4:7], dof[8:11]])
                translation = np.array([dof[3], dof[7], dof[11]])
                Pose_camera = np.eye(4)
                Pose_camera[:3, :3] = rotation
                Pose_camera[:3, 3] = translation
                Pose_velodyne = inv_Tr @ Pose_camera @ Tr
                inv_Pose_camera = np.linalg.inv(Pose_velodyne)

                for key in keys:
                    xyz_matrix = deduplicated_patches[key]['xyz']
                    re_transposed_xyz_matrix = np.dot(xyz_matrix, inv_Pose_camera[:3, :3].T) + inv_Pose_camera[:3, 3]
                    deduplicated_patches[key]['xyz'] = re_transposed_xyz_matrix

            for frame_idx, indices in delete_indices.items():
                if indices:
                    points = np.array(PC_data[frame_idx])
                    mask = np.array(indices) == False
                    PC_data[frame_idx] = points[mask].tolist()

        return matched_points_count

    def store_remaining_points(self, PC_data, deduplicated_patches, poses, Tr, transposed):
        """
        After processing all subsets, store the remaining (unmatched) points for each frame.
        """
        for i in range(len(PC_data)):
            remaining_points = np.array(PC_data[i])

            if remaining_points.shape[0] == 0:
                continue

            inv_Tr = np.linalg.inv(Tr)

            dof = poses[i]
            rotation = np.array([dof[0:3], dof[4:7], dof[8:11]])
            translation = np.array([dof[3], dof[7], dof[11]])

            Pose_camera = np.eye(4)
            Pose_camera[:3, :3] = rotation
            Pose_camera[:3, 3] = translation
            Pose_velodyne = inv_Tr @ Pose_camera @ Tr
            inv_Pose_camera = np.linalg.inv(Pose_velodyne)

            if remaining_points.ndim == 1:
                print("Warning: remaining_points is 1D, shape =", remaining_points.shape)
                print("remaining_points contents:", remaining_points)

            re_transposed_xyz = np.dot(remaining_points[:, :3], inv_Pose_camera[:3, :3].T) + inv_Pose_camera[:3, 3]
            remaining_points[:, :3] = re_transposed_xyz

            if remaining_points.shape[0] > 0:
                key = ((i,), i, True, False)
                if key not in deduplicated_patches:
                    deduplicated_patches[key] = {
                        "xyz": None,
                        "intensities": []
                    }

                deduplicated_patches[key]['xyz'] = re_transposed_xyz
                deduplicated_patches[key]['intensities'].append(remaining_points[:, 3])

    def generate_cluster_ordering(self, PC_data):
        """
        Generate the clustering order of PC frames according to the deduplication algorithm.
        """
        N = len(PC_data)
        clustering_order = []

        for size in range(N, 1, -1):
            for start in range(N - size + 1):
                subset = range(start, start + size)
                if size % 2 == 1:
                    anchor = subset[size // 2]
                else:
                    anchor = subset[(size // 2) - 1]

                subset_list = list(subset)
                clustering_order.append((subset_list, anchor))

        return clustering_order

    def transpose_PC_data(self, PC_data, poses, Tr):
        """
        Transpose (rotate/translate) the unmatched points in each point cloud frame
        according to the provided 6-DoF poses.
        """
        Tr_inv = np.linalg.inv(Tr)

        for i, frame_points in enumerate(PC_data):
            dof = poses[i]

            rotation = np.array([
                dof[0:3],
                dof[4:7],
                dof[8:11]
            ])
            translation = np.array([dof[3], dof[7], dof[11]])

            Pose_camera = np.eye(4)
            Pose_camera[:3, :3] = rotation
            Pose_camera[:3, 3] = translation

            Pose_velodyne = np.dot(np.dot(Tr_inv, Pose_camera), Tr)

            points = np.array(frame_points)

            transformed_xyz = np.dot(points[:, :3], Pose_velodyne[:3, :3].T) + Pose_velodyne[:3, 3]

            transformed_points = np.hstack((transformed_xyz, points[:, 3:4]))

            PC_data[i] = transformed_points.tolist()
