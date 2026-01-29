'''
The function class for deduplication with input PC data
'''
import pycuda.driver as cuda
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import pulp
import faiss
from joblib import Parallel, delayed
import concurrent.futures
import time
import math
import joblib
import bisect
import gc
import os

# def get_free_memory(gpu_id):
#     cuda.init()
#     device = cuda.Device(gpu_id)
#     free_mem, total_mem = device.get_memory_info()
#     return free_mem

def get_free_memory(gpu_id):
    cuda.init()
    device = cuda.Device(gpu_id)
    context = device.make_context()
    free_mem, total_mem = cuda.mem_get_info()
    context.pop()
    return free_mem



class Deduplicator:
    def __init__(self, args):

        self.args = args

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

        if gpuid is not None:
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

    # def solve_faiss_greedy_matching(self, anchor_points, candidate_points, threshold, alpha=2.0, k=10, batch_size=2000, n_worker=5):
    #     """
    #     GPU-accelerated greedy matching using FAISS (x, y, z only).
    #     - Only consider top-k nearest neighbors within alpha * threshold.
    #     - Enforce per-pair constraint: dist <= alpha * threshold
    #     - Enforce global avg distance constraint: avg <= threshold
    #     """
    #
    #     n = anchor_points.shape[0]
    #     m = candidate_points.shape[0]
    #
    #     dim = 3
    #     res = faiss.StandardGpuResources()
    #     index = faiss.GpuIndexFlatL2(res, dim)
    #     index.add(candidate_points[:, :3].astype(np.float32))
    #
    #     distances, indices = index.search(anchor_points[:, :3].astype(np.float32), k)
    #
    #     valid_matches = []
    #     for i in range(n):
    #         for j_pos in range(k):
    #             j = indices[i][j_pos]
    #             dist = np.sqrt(distances[i][j_pos])  # FAISS returns squared L2
    #             if dist <= alpha * threshold:
    #                 valid_matches.append((i, j, dist))
    #
    #
    #     used_anchors = set()
    #     used_candidates = set()
    #     total_dist = 0.0
    #     result = []
    #
    #     for i, j, dist in valid_matches:
    #         if i in used_anchors or j in used_candidates:
    #             continue
    #         if len(result) > 0 and (total_dist + dist) / (len(result) + 1) > threshold:
    #             continue
    #         result.append((i, j))
    #         used_anchors.add(i)
    #         used_candidates.add(j)
    #         total_dist += dist
    #
    #     return result

    # def solve_faiss_greedy_matching(self, anchor_points, candidate_points, threshold, alpha=2.0, k=10, batch_size=2000,
    #                                 n_worker=5):
    #     """
    #     GPU-accelerated greedy matching using FAISS (x, y, z only).
    #     - Only consider top-k nearest neighbors within alpha * threshold.
    #     - Enforce per-pair constraint: dist <= alpha * threshold
    #     - Enforce global avg distance constraint: avg <= threshold
    #     """
    #
    #     # Check available memory on each GPU
    #     gpu_id = 0 if get_free_memory(0) > get_free_memory(1) else 1
    #
    #     # Set the GPU device
    #     cuda.Device(gpu_id).make_context()
    #
    #     n = anchor_points.shape[0]
    #     m = candidate_points.shape[0]
    #
    #     dim = 3
    #     res = faiss.StandardGpuResources()
    #     # res.setDefaultStream(gpu_id)
    #     index = faiss.GpuIndexFlatL2(res, dim)
    #     index.add(candidate_points[:, :3].astype(np.float32))
    #
    #     # st_time = time.time()
    #     distances, indices = index.search(anchor_points[:, :3].astype(np.float32), k)
    #     # end_time = time.time()
    #     # print("time compute with index: {}".format(end_time - st_time))
    #
    #     # st_time = time.time()
    #     # valid_matches = []
    #     # for i in range(n):
    #     #     for j_pos in range(k):
    #     #         j = indices[i][j_pos]
    #     #         dist = np.sqrt(distances[i][j_pos])  # FAISS returns squared L2
    #     #         if dist <= alpha * threshold:
    #     #             valid_matches.append((i, j, dist))
    #     # end_time = time.time()
    #     # print("time finding matching: {}, len: {}".format(end_time - st_time, len(valid_matches)))
    #
    #     '''Here starts the fast binary search version'''
    #     st_time = time.time()
    #     valid_matches = []
    #     distances = np.sqrt(distances)
    #
    #     # Collect all valid matches
    #     for i in range(n):
    #         for j_pos in range(k):
    #             j = indices[i][j_pos]
    #             valid_matches.append((i, j, distances[i][j_pos]))
    #     valid_matches.sort(key=lambda x: x[2])
    #     # end_time = time.time()
    #     # print("time before chech distance: {}".format(end_time - st_time))
    #     # Filter out matches with distance larger than alpha * threshold
    #     threshold_distance = alpha * threshold
    #     index = bisect.bisect_right([match[2] for match in valid_matches], threshold_distance)
    #     valid_matches = valid_matches[:index]
    #     '''Here end the binary search version'''
    #
    #     # valid_matches = [match for match in valid_matches if match[2] <= alpha * threshold]
    #     # end_time = time.time()
    #     # print("time finding matching: {}, len: {}".format(end_time - st_time, len(valid_matches)))
    #
    #     # st_time = time.time()
    #     # used_anchors = set()
    #     # used_candidates = set()
    #     # total_dist = 0.0
    #     # result = []
    #     #
    #     # for i, j, dist in valid_matches:
    #     #     if i in used_anchors or j in used_candidates:
    #     #         continue
    #     #     if len(result) > 0 and (total_dist + dist) / (len(result) + 1) > threshold:
    #     #         continue
    #     #     result.append((i, j))
    #     #     used_anchors.add(i)
    #     #     used_candidates.add(j)
    #     #     total_dist += dist
    #     # end_time = time.time()
    #     # print("without sort: {}, time: {}".format(len(result), end_time - st_time))
    #
    #     # st_time = time.time()
    #     # Sort valid_matches by distance in ascending order
    #     valid_matches.sort(key=lambda x: x[2])
    #
    #     used_anchors = set()
    #     used_candidates = set()
    #     total_dist = 0.0
    #     result = []
    #
    #     for i, j, dist in valid_matches:
    #         if i in used_anchors or j in used_candidates:
    #             continue
    #         if len(result) > 0:
    #             if dist < threshold or (total_dist + dist) / (len(result) + 1) <= threshold:
    #                 result.append((i, j))
    #                 used_anchors.add(i)
    #                 used_candidates.add(j)
    #                 total_dist += dist
    #             else:
    #                 break
    #         else:
    #             result.append((i, j))
    #             used_anchors.add(i)
    #             used_candidates.add(j)
    #             total_dist += dist
    #         #     and (dist < threshold or (total_dist + dist) / (len(result) + 1) > threshold):
    #         #     continue
    #         # result.append((i, j))
    #         # used_anchors.add(i)
    #         # used_candidates.add(j)
    #         # total_dist += dist
    #     # end_time = time.time()
    #     # print("with sort: {}, time: {}".format(len(result), end_time - st_time))
    #
    #     return result

    def solve_faiss_greedy_matching(self, index, candidate_points, threshold, alpha=2.0, k=10):
        """
        GPU-accelerated greedy matching using FAISS (x, y, z only).
        - Only consider top-k nearest neighbors within alpha * threshold.
        - Enforce per-pair constraint: dist <= alpha * threshold
        - Enforce global avg distance constraint: avg <= threshold
        """

        # gpu_id = 0 if get_free_memory(0) > get_free_memory(1) else 1
        #
        # # Set the GPU device
        # device = cuda.Device(gpu_id)
        #
        # context = device.make_context()
        #
        # # free_memory_before = get_free_memory(gpu_id)
        #
        # # Build the FAISS index with anchor points
        # dim = 3
        # res = faiss.StandardGpuResources()
        # index = faiss.GpuIndexFlatL2(res, dim)
        # index.add(anchor_points[:, :3].astype(np.float32))

        # n = index.ntotal
        m = candidate_points.shape[0]

        st_time = time.time()

        # Search for top-k nearest neighbors
        distances, indices = index.search(candidate_points[:, :3].astype(np.float32), k)
        # distances, indices = index.search(candidate_points[:, :3].astype(np.float16), k)
        # end_time = time.time()
        # print("time for matching: ", end_time - st_time)

        valid_matches = []
        distances = np.sqrt(distances)



        # Collect all valid matches
        for i in range(m):
            for j_pos in range(k):
                j = indices[i][j_pos]
                valid_matches.append((i, j, distances[i][j_pos]))
        st_time = time.time()
        valid_matches.sort(key=lambda x: x[2])

        # end_time = time.time()
        # print("time for sort: ", end_time - st_time)

        st_time = time.time()
        # Filter out matches with distance larger than alpha * threshold
        threshold_distance = alpha * threshold
        index = bisect.bisect_right([match[2] for match in valid_matches], threshold_distance)
        valid_matches = valid_matches[:index]

        # end_time = time.time()
        # print("time for bi searching: ", end_time - st_time)

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

        # end_time = time.time()
        # print("time for final matching: ", end_time - st_time)

        return [(j, i) for i, j in result]

    def solve_faiss_greedy_matching_2(self, index, candidate_points, k=10):
        """
        Perform FAISS index search to find the top-k nearest neighbors for each candidate point.
        Returns distances and indices of the nearest neighbors.
        """
        distances, indices = index.search(candidate_points[:, :3].astype(np.float32), k)
        distances = np.sqrt(distances)
        return distances, indices

    def solve_ilp_matching(self, anchor_points, candidate_points, threshold, alpha=2.0):
        """
        ILP matching with:
        - Hard constraint: no match has distance > alpha * threshold
        - Soft constraint: average distance ≤ threshold
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

        # Soft global constraint: total dist / #match ≤ threshold
        total_matches = pulp.lpSum(x[i, j] for (i, j) in valid_pairs)
        total_dist = pulp.lpSum(x[i, j] * distances[i, j] for (i, j) in valid_pairs)
        prob += total_dist <= threshold * total_matches

        # Solve
        prob.solve(pulp.PulpSolverDefault)

        # Extract matches
        result = [(i, j) for (i, j) in valid_pairs if pulp.value(x[i, j]) > 0.5]
        return result

    def generate_related_subsets(self, subset, anchor):
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
        If the size is odd, return a list containing the middle element.
        If the size is even, return a list containing both middle elements.

        Args:
            related_subset (list): The related subset of elements.

        Returns:
            list: A list containing the anchor element(s) of the subset.
        """
        size = len(related_subset)
        if size == 0:
            raise ValueError("The related_subset cannot be empty.")

        # Determine the anchor(s) based on the size
        if size % 2 == 1:  # Odd size
            return [related_subset[size // 2]]
        else:  # Even size
            return [related_subset[(size // 2) - 1], related_subset[size // 2]]

    def match_frames(self, PC_data, clustering_order, threshold, deduplicated_patches, poses, Tr, transposed=False,
                     gpuid=None):
        """
        Optimized matching using cKDTree.
        For each subset (from clustering_order), the anchor frame is used as reference.
        For every other frame in the subset, we query its points against the anchor using cKDTree,
        based solely on the 3D distance (first 3 dimensions). For each matched candidate,
        the candidate's intensity is stored (while the anchor’s intensity is stored only once).
        After matching, the matched candidate points are removed from their frames,
        and matched anchor points are removed from the anchor frame.
        """
        matched_points_count = 0
        matched_subset_mark = {tuple(subset): False for subset, _ in clustering_order}

        for subset, anchor_idx in clustering_order:
            if matched_subset_mark[tuple(subset)]:
                continue

            anchor_points = np.array(PC_data[anchor_idx])
            if anchor_points.shape[0] == 0:
                continue

            # anchor_matches = {i: [] for i in range(anchor_points.shape[0])}
            #
            # dim = 3
            # res = faiss.StandardGpuResources()
            # index = faiss.GpuIndexFlatL2(res, dim)
            # index.add(anchor_points[:, :3].astype(np.float32))
            #
            # for frame_idx in subset:
            #     if frame_idx == anchor_idx:
            #         continue
            #     candidate_points = np.array(PC_data[frame_idx])
            #     if candidate_points.shape[0] == 0:
            #         continue
            #
            #     result = self.solve_faiss_greedy_matching(index, candidate_points, threshold, alpha=2.0, k=10)
            #     for anc_idx, cand_idx in result:
            #         anchor_matches[anc_idx].append((frame_idx, cand_idx))

            anchor_matches = {i: [] for i in range(anchor_points.shape[0])}

            # dim = 3
            # res = faiss.StandardGpuResources()
            # index = faiss.GpuIndexFlatL2(res, dim)
            # index.add(anchor_points[:, :3].astype(np.float32))

            # Step 1: Collect distances and indices for all candidate point clouds
            all_distances = []
            all_indices = []

            for frame_idx in subset:
                if frame_idx == anchor_idx:
                    continue
                candidate_points = np.array(PC_data[frame_idx])
                if candidate_points.shape[0] == 0:
                    continue

                # distances, indices = self.solve_faiss_greedy_matching_2(index, candidate_points, k=10)

                # Construct the FAISS index for candidate_points
                dim = 3
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatL2(res, dim)
                index.add(candidate_points[:, :3].astype(np.float32))

                # Query the anchor_points against the candidate_points index
                distances, indices = self.solve_faiss_greedy_matching_2(index, anchor_points, k=10)

                all_distances.append(distances)
                all_indices.append(indices)

            # Step 2: Compute the sum of top-1 distances for each anchor point
            distance_sums = []
            # print(anchor_points.shape[0])
            # print(len(all_distances), len(all_distances[0]))
            for anc_idx in range(anchor_points.shape[0]):
                top_1_sum = sum(distances[anc_idx, 0] for distances in all_distances)
                distance_sums.append((anc_idx, top_1_sum))

            # Step 3: Sort anchor points by the distance sums
            distance_sums.sort(key=lambda x: x[1])

            # Step 4: Generate anchor_matches based on the sorted anchor points
            used_anchors = set()
            used_candidates = set()

            anchor_frame_idx = subset.index(anchor_idx)

            # Initialize variables to track total distance and match count
            total_dist = 0.0
            match_count = 0

            alpha = self.args.soft_distance_ratio

            for anc_idx, _ in distance_sums:
                for candidate_idx, distances in enumerate(all_distances):
                    # Determine the correct frame index based on the position of candidate_idx
                    # print(subset, candidate_idx, anchor_frame_idx)
                    # if anchor_frame_idx in subset:
                    #     pass
                    # else:
                    #     raise ValueError(f"anchor_frame_idx ({anchor_frame_idx}) is not in subset: {subset}")

                    if candidate_idx < anchor_frame_idx:
                        frame_idx = subset[candidate_idx]
                    else:
                        frame_idx = subset[candidate_idx + 1]

                    for rank in range(distances.shape[1]):  # Iterate over top-k candidates
                        cand_idx = all_indices[candidate_idx][anc_idx, rank]  # Top-k index
                        distance = distances[anc_idx, rank]  # Top-k distance

                        # Check if the match satisfies the constraints
                        if (
                                distance <= alpha * threshold and
                                # cand_idx not in used_candidates and
                                (candidate_idx, cand_idx) not in used_candidates and
                                (match_count == 0 or (total_dist + distance) / (match_count + 1) <= threshold)
                        ):
                            anchor_matches[anc_idx].append((frame_idx, cand_idx))
                            # used_candidates.add(cand_idx)
                            used_candidates.add((candidate_idx, cand_idx))
                            total_dist += distance
                            match_count += 1
                            break  # Stop checking further candidates for this anchor point


            delete_indices = {frame_idx: [False for _ in range(len(PC_data[frame_idx]))] for frame_idx in
                              range(len(PC_data))}
            same_intensity_count = 0
            keys = []
            related_subsets = self.generate_related_subsets(subset, anchor_idx)

            deduplicated_related_subsets = []

            # We set a threshold here to prevent construct patch that with a small number
#            satisfied_count = 0
#            for anc_idx, matches in anchor_matches.items():
#                if len(matches) == len(subset) - 1:
#                    satisfied_count += 1
#            if satisfied_count < 2000:
#                continue

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
                    # print(matches)
                    match_frame_ids = [id for id, _ in matches]
                    match_frame_ids.append(anchor_idx)
                    match_frame_ids.sort()
                    current_subset = match_frame_ids

                    for related_subset in related_subsets:
                        if matched_subset_mark[tuple(related_subset)]:
                            continue

                        # if all(item in current_subset for item in related_subset):
                        if set(current_subset) == set(related_subset) and anchor_idx in self.generate_anchor(related_subset):
                            patch = {
                                "xyz": (anchor_points[anc_idx, :3]).tolist(),
                                "intensities": [None] * len(related_subset)
                            }
                            patch["intensities"][related_subset.index(anchor_idx)] = anchor_points[anc_idx, 3]

                            same_intensity = 0
                            for frame_idx, cand_idx in matches:
                                if frame_idx in related_subset:
                                    patch["intensities"][related_subset.index(frame_idx)] = \
                                    PC_data[frame_idx][cand_idx][3]
                                    if abs(PC_data[frame_idx][cand_idx][3] - anchor_points[
                                        anc_idx, 3]) < self.args.intensity_threshold:
                                        same_intensity += 1

                            if self.args.deduplicate_intensity:
                                if same_intensity == len(related_subset) - 1:
                                    same_intensity_count += 1
                                    key = (tuple(related_subset), anchor_idx, True, transposed)
                                    if key not in keys:
                                        keys.append(key)
                                    if key not in deduplicated_patches:
                                        deduplicated_patches[key] = []
                                    patch["intensities"] = [anchor_points[anc_idx, 3]]
                                    deduplicated_patches[key].append(patch)
                                else:
                                    key = (tuple(related_subset), anchor_idx, False, transposed)
                                    matched_subset_mark[tuple(related_subset)] = True
                                    if key not in keys:
                                        keys.append(key)
                                    if key not in deduplicated_patches:
                                        deduplicated_patches[key] = []
                                    deduplicated_patches[key].append(patch)
                            else:
                                key = (tuple(related_subset), anchor_idx, False, transposed)
                                deduplicated_related_subsets.append(related_subset)
                                # matched_subset_mark[tuple(related_subset)] = True
                                if key not in keys:
                                    keys.append(key)
                                if key not in deduplicated_patches:
                                    deduplicated_patches[key] = []
                                deduplicated_patches[key].append(patch)

                            matched_points_count += 1
                            for frame_idx, cand_idx in matches:
                                if frame_idx in related_subset:
                                    delete_indices[frame_idx][cand_idx] = True
                            delete_indices[anchor_idx][anc_idx] = True

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
        Each remaining point is stored as a patch with the same structure as matched patches.
        The key is of the form ((i,), transposed) so that the original PC frame can be reconstructed.
        """
        for i in range(len(PC_data)):
            remaining_points = np.array(PC_data[i])

            
            if remaining_points.shape[0] == 0:
                continue

            '''Here align the points back to relative coordinate system'''
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
            # inv_Pose_camera = np.linalg.inv(poses[anchor_idx])
            re_transposed_xyz = np.dot(remaining_points[:, :3], inv_Pose_camera[:3, :3].T) + inv_Pose_camera[:3,
                                                                                             3]
            # re_transposed_xyz = np.dot(re_transposed_xyz, inv_Tr[:3, :3].T) + inv_Tr[:3, 3]
            remaining_points[:, :3] = re_transposed_xyz  # Directly update the anchor_points

            # mean_xyz = np.mean(remaining_points[:, :3], axis=0)
            #
            # # Subtract the mean from the xyz coordinates to centralize the points
            # remaining_points[:, :3] = re_transposed_xyz - mean_xyz  # Directly update the anchor_points

            if remaining_points.shape[0] > 0:
                key = ((i,), i, True, False)
                if key not in deduplicated_patches:
                    # deduplicated_patches[key] = []
                    deduplicated_patches[key] =  {
                    "xyz": None,
                    "intensities": []
                    }

                # Store each unmatched point as a patch with the same structure.
                # for j in range(remaining_points.shape[0]):
                #     patch = {
                #         "xyz": remaining_points[j, :3].tolist(),
                #         "intensities": [remaining_points[j, 3]]
                #     }
                #     deduplicated_patches[key].append(patch)
                deduplicated_patches[key]['xyz'] = re_transposed_xyz
                deduplicated_patches[key]['intensities'].append(remaining_points[:, 3])

    def generate_cluster_ordering(self, PC_data):
        """
        Generate the clustering order of PC frames according to the deduplication algorithm.
        We enumerate subsets of frames from size N down to size 2. For each subset:
          - If the subset size is odd, select the center element in the subset as anchor.
          - If the subset size is even, select the left of the two center elements as anchor.

        Returns:
            clustering_order (list): A list of tuples (subset, anchor_index), where
                'subset' is a list of frame indices, and 'anchor_index' is the chosen anchor
                from that subset.
        """
        N = len(PC_data)
        clustering_order = []

        # For each subset size from N down to 2
        # for size in range(N, 1, -1):
        #     # Generate all subsets (combinations) of the frame indices of length 'size'
        #     for subset in combinations(range(N), size):
        #         # subset is a tuple of frame indices in ascending order
        #         # Decide anchor frame according to odd/even size
        #         if size % 2 == 1:
        #             # Odd number of frames: pick the exact center
        #             anchor = subset[size // 2]
        #         else:
        #             # Even number of frames: pick the left of the two centers
        #             anchor = subset[(size // 2) - 1]
        #
        #         # Convert subset to a list (optional, for consistency)
        #         subset_list = list(subset)
        #         clustering_order.append((subset_list, anchor))

        for size in range(N, 1, -1):
            # Generate all continuous subsets of the frame indices of length 'size'
            for start in range(N - size + 1):
                subset = range(start, start + size)
                # Decide anchor frame according to odd/even size
                if size % 2 == 1:
                    # Odd number of frames: pick the exact center
                    anchor = subset[size // 2]
                else:
                    # Even number of frames: pick the left of the two centers
                    anchor = subset[(size // 2) - 1]

                # Convert subset to a list (optional, for consistency)
                subset_list = list(subset)
                clustering_order.append((subset_list, anchor))

        # # Add a subset with only the start and end frames, using the start frame as the anchor
        # start_end_subset = [0, N-1]
        # if (start_end_subset, start_end_subset[0]) not in clustering_order:
        #     clustering_order.append((start_end_subset, start_end_subset[0]))

        # for i in range(N - 1):
        #     subset = [i, i + 1]
        #     anchor = subset[0]
        #     clustering_order.append((subset, anchor))

        return clustering_order

    def transpose_PC_data(self, PC_data, poses, Tr):
        """
        Transpose (rotate/translate) the unmatched points in each point cloud frame
        according to the provided 6-DoF poses.

        Parameters:
          - PC_data: a list of point cloud frames. Each frame is a list (or array)
                     of points, where each point is [x, y, z, intensity].
          - poses: a list of 6-DoF parameters (one per frame). Each pose is a list
                   of 12 values arranged as:
                     [r11, r12, r13, t1,  r21, r22, r23, t2,  r31, r32, r33, t3]

        The transformation is applied in two steps:
          1. Construct the camera pose matrix (4x4) from the 6-DoF parameters.
          2. Convert to Velodyne coordinates using:
                Pose_velodyne = Tr_inv * Pose_camera * Tr
             where Tr and Tr_inv are provided.
          3. Apply Pose_velodyne to the x,y,z coordinates of each point (intensity remains unchanged).
        """
        # Define the transformation matrices from camera to Velodyne coordinates.

        Tr_inv = np.linalg.inv(Tr)

        # Process each frame in PC_data
        for i, frame_points in enumerate(PC_data):
            # Retrieve the 6-DoF parameters for frame i.
            # Expected format: [r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3]
            dof = poses[i]

            # Construct the rotation matrix.
            rotation = np.array([
                dof[0:3],
                dof[4:7],
                dof[8:11]
            ])
            # Construct the translation vector.
            translation = np.array([dof[3], dof[7], dof[11]])

            # Build the 4x4 camera pose matrix.
            Pose_camera = np.eye(4)
            Pose_camera[:3, :3] = rotation
            Pose_camera[:3, 3] = translation

            # Convert to Velodyne coordinate system.
            Pose_velodyne = np.dot(np.dot(Tr_inv, Pose_camera), Tr)

            # Convert the frame points to a numpy array (if not already)
            points = np.array(frame_points)  # Expected shape: (n, 4)

            # Apply the transformation to the x,y,z coordinates.
            transformed_xyz = np.dot(points[:, :3], Pose_velodyne[:3, :3].T) + Pose_velodyne[:3, 3]

            # Concatenate the intensity dimension (unchanged)
            transformed_points = np.hstack((transformed_xyz, points[:, 3:4]))

            # Update PC_data[i] with the transformed points (as a list, if desired)
            PC_data[i] = transformed_points.tolist()


if __name__ == "__main__":
    import os

    # Load three consecutive point cloud frames
    file1 = "pc_instance/cont_pc/000007.bin"
    file2 = "pc_instance/cont_pc/000008.bin"
    file3 = "pc_instance/cont_pc/000009.bin"

    if not (os.path.exists(file1) and os.path.exists(file2) and os.path.exists(file3)):
        print("Please ensure the .bin files exist in the specified paths.")
    else:
        points_1 = np.fromfile(file1, dtype=np.float32).reshape(-1, 4)
        points_2 = np.fromfile(file2, dtype=np.float32).reshape(-1, 4)
        points_3 = np.fromfile(file3, dtype=np.float32).reshape(-1, 4)
        print("points number: ", len(points_1), len(points_2), len(points_3))

        # Combine them into a list
        PC_data = [points_1.tolist(), points_2.tolist(), points_3.tolist()]

        Tr = np.array([
            [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02],
            [-7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02],
            [9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01],
            [0, 0, 0, 1]
        ])

        dof1 = [0.999573, 0.00254257, -0.0291203, -0.110765, -0.00272044, 0.999978, -0.00607018, -0.105792, 0.0291042,
                0.00614681, 0.999558, 5.08088]
        dof2 = [0.999438, 0.00434114, -0.0332473, -0.142767, -0.00450281, 0.999978, -0.00478932, -0.11819, 0.0332258,
                0.00493634, 0.999436, 5.87572]
        dof3 = [0.999253, 0.00411408, -0.0384275, -0.178263, -0.00420248, 0.999989, -0.00221999, - 0.130267, 0.0384179,
                0.00237982, 0.999259, 6.68287]

        # For each frame, you might have a pose (rotation/translation).
        # Here, we just use placeholders.
        poses = [dof1, dof2, dof3]

        # Example threshold for matching (tune as needed)
        threshold = 0.02

        deduplicator = Deduplicator()
        results = deduplicator.deduplication(PC_data, poses, Tr, threshold)

        print("Matched points (total):", results["matched_points"])
        print("Number of deduplicated patches:", len(results["deduplicated_patches"]))
