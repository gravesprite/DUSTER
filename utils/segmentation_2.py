import pulp
import random
import numpy as np
from math import gcd, sqrt, log

from utils.deduplicatation_2 import Deduplicator
from utils.disk_manage_las import DiskManager
from utils.index import Indexer, SequenceIndexer

import os
from io import BytesIO
import numpy as np
import laspy
from laspy.compression import LazBackend
import concurrent.futures
import math

import pycuda.driver as cuda
import itertools

import concurrent.futures
# from multiprocessing import Manager
import multiprocessing

import time
import copy


# def evaluate_block_wrapper(args):
#     """Wrapper function to call evaluate_block."""
#     instance, block, block_size, long_seg, short_seg, deduplicator, diskmanager = args
#     return instance.evaluate_block(block, block_size, long_seg, short_seg, deduplicator, diskmanager)

def evaluate_block_wrapper(args):

    """Wrapper function to call evaluate_block with GPU assignment."""
    instance, block, block_size, long_seg, short_seg, deduplicator, diskmanager, gpuid = args

    if gpuid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

    cuda.init()

    return instance.evaluate_block(block, block_size, long_seg, short_seg, deduplicator, diskmanager, gpuid)


def process_short_segment(args):
    iterator, short_seg, pc_files, poses, deduplicator, diskmanager, score_segment, gpuid = args
    # Set the GPU ID for the current process
    if gpuid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

    cuda.init()
    # gpu_resources = faiss.StandardGpuResources()

    """Process a short segment and return the result."""

    # print(gpuid)
    segment = pc_files[iterator:iterator + short_seg]
    segment_poses = poses[iterator:iterator + short_seg]



    score = score_segment(segment, deduplicator, diskmanager, segment_poses, gpuid=gpuid, if_store=True)

    return (iterator, iterator + short_seg), score

def compute_segment_score_wrapper(args):
    """
    Standalone function to compute segment score.
    """
    segment, frames, deduplicator, diskmanager, poses, score_segment, gpuid, if_store = args
    if gpuid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

    cuda.init()

    segment_files = frames[segment[0]:segment[1]]
    segment_poses = poses[segment[0]:segment[1]]
    # Perform the computation

    score = score_segment(segment_files, deduplicator, diskmanager, segment_poses, gpuid=gpuid, if_store=if_store)
    return score

class Segmentator:

    def __init__(self, velodyne_dir, pc_files, queries, logger, poses, Tr, threshold, args, max_length, A,
                 max_processes=4, exploration_param=1.0):
        """
        Parameters:
          max_length: Maximum allowed length of a segment.
          A: A multiplier that defines the block size (block_size = A * max_length).
          exploration_param: Parameter for exploration bonus in UCB.
        """

        self.velodyne_dir = velodyne_dir
        self.pc_files = pc_files
        self.queries = queries
        self.logger = logger
        self.poses = poses
        self.Tr = Tr
        self.threshold = threshold
        self.args = args

        self.max_length = max_length
        self.A = A
        self.block_size = A * max_length
        self.exploration_param = exploration_param

        # manager = Manager()
        # self.semaphore = manager.Semaphore(max_processes)

        # Score matrix for segments in current block.
        # Will store values for valid segments (i,j) with i < j and j-i <= max_length.
        self.score_matrix = None

        # MAB storage: dictionary with keys (i,j) and values: {"count": int, "value": float}
        self.mab_stats = {}

        # Flag to indicate if we need to recompute via DP (if policy proposal was worse than baseline)
        self.needs_recompute = False

        # manager = Manager()
        # self.semaphore = manager.Semaphore(max_processes)


    # ... [rest of your existing class methods] ...

    def compute_time_and_scores(self, pc_files, poses, deduplicator, diskmanager, score_segment, max_length):
        """
        Compute time costs and compression scores for segment lengths from 1 to max_length.

        Args:
            pc_files (list): List of point cloud files.
            poses (list): List of poses corresponding to the point cloud files.
            deduplicator (Deduplicator): Deduplication utility.
            diskmanager (DiskManager): Disk management utility.
            score_segment (function): Function to compute the score of a segment.
            max_length (int): Maximum segment length to consider.

        Returns:
            dict, dict: Time costs and compression scores for each segment length.
        """
        time_costs = {}
        compression_scores = {}

        for segment_length in range(1, max_length + 1):
            # Define the segment
            segment = pc_files[:segment_length]
            segment_poses = poses[:segment_length]

            # Measure time cost
            start_time = time.time()
            score = score_segment(segment, deduplicator, diskmanager, segment_poses)
            end_time = time.time()

            # Record time cost and compression score
            time_costs[segment_length] = end_time - start_time
            compression_scores[segment_length] = score

        return time_costs, compression_scores

    def segment_with_ip_mab(self, pc_files, time_budget, max_seg_length):
        """
        Segment the sequence using integer programming and MAB.
        """
        total_frames = len(pc_files)

        multiprocessing.set_start_method("spawn", force=True)

        # Assume you provide these dicts externally or define them
        time_costs = {l: 0.2 * l for l in range(1, max_seg_length + 1)}
        compression_scores = {l: 100 / l + random.uniform(-5, 5) for l in range(1, max_seg_length + 1)}  # Dummy

        time_costs = {1: 0.1 * 1, 2: 1.5 * 2, 3: 2.0 * 3, 4: 2.5 * 4}
        compression_scores = {1: 0.2, 2: 0.4, 3: 0.54, 4: 0.7}  # Corresponding to 1, 0.2*2, 0.18*3, 0.175*4

        time_costs = {1: 0.1914682388305664, 2: 6.162039518356323, 3: 15.435150146484375, 4: 34.22509431838989}
        compression_scores = {1: 461.71875, 2: 842.2587890625, 3: 1210.3564453125, 4: 1598.0810546875}

        # deduplicator = Deduplicator(self.args)
        # diskmanager = DiskManager(self.args)
        #
        # time_costs, compression_scores = self.compute_time_and_scores(pc_files, self.poses, deduplicator, diskmanager,
        #                                                          self.score_segment, max_seg_length)

        # Assuming self.args.max_cluster_length is defined
        max_cluster_length = self.args.max_cluster_length

        # Filter dictionaries to remove keys larger than max_cluster_length
        time_costs = {k: v for k, v in time_costs.items() if k <= max_cluster_length}
        compression_scores = {k: v for k, v in compression_scores.items() if k <= max_cluster_length}

        self.logger.info("time_costs: {}\n compression_scores: {}".format(time_costs, compression_scores))
        # return
        
        if self.args.same_segment:
            time_budget = time_costs[self.args.max_cluster_length] * total_frames / self.args.max_cluster_length
        else:
            time_budget = ((time_costs[self.args.max_cluster_length - 1] / (self.args.max_cluster_length - 1)) *(1 - self.args.time_budget) + (time_costs[self.args.max_cluster_length] / self.args.max_cluster_length) * (self.args.time_budget))  * total_frames

        # Step 1: Solve Integer Programming Problem
        segment_counts = self.solve_general_integer_program(
            time_costs, compression_scores, total_frames, time_budget, max_seg_length
        )
        print(segment_counts)

        cloned_segment_counts = copy.deepcopy(segment_counts)

        self.logger.info("Segment counts: {}".format(segment_counts))

        if len(segment_counts) == 1:
            segments = self.direct_assign_segments(pc_files, segment_counts)
        else:
            # Step 2: Assign segments using smoothed probability update
            segments = self.assign_segments_with_gaussian_smoothing(pc_files, segment_counts)

        segmentation_results = []
        for segment in segments:
            proposal = [[0, segment[1] - segment[0]]]
            segmentation_results.append((segment[0], segment[1], proposal))



        # Construct the SequenceIndexer object
        sequence_indexer = SequenceIndexer(
            segmentation_results=segmentation_results,
            frames=pc_files,
            root_store_path=self.args.root_store_path_wa if self.args.workload_aware else self.args.root_store_path,
            Tr=self.Tr,
            diskmanager=DiskManager(self.args),
            args=self.args
        )


        return cloned_segment_counts, sequence_indexer

    def direct_assign_segments(self, pc_files, segment_counts):
        total_frames = len(pc_files)
        long_seg = max(segment_counts.keys())
        num_long = segment_counts.get(long_seg, 0)

        # Create segments based on segment counts
        segments = []
        iterator = 0

        deduplicator = Deduplicator(self.args)
        diskmanager = DiskManager(self.args)

        segments  = []

        # Assign long segments and process in batches
        T = 6  # Batch size for processing
        batch_segments = []
        gpu_ids = itertools.cycle([0, 1])  # Cycle between GPU 0 and GPU 1



        for idx in range(num_long):
            if iterator + long_seg <= total_frames:
                batch_segments.append((iterator, iterator + long_seg))
                segments.append((iterator, iterator + long_seg))
                iterator += long_seg

            # Process the batch every T assignments or at the end of all assignments
            if len(batch_segments) == T or idx == num_long - 1:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    args = [
                        (segment[0], long_seg, pc_files, self.poses, deduplicator, diskmanager, self.score_segment,
                         next(gpu_ids))
                        for segment in batch_segments
                    ]
                    batch_scores = list(executor.map(process_short_segment, args))

                print(batch_scores)
                batch_segments = []  # Clear the batch for the next set of assignments

        return segments


    def solve_general_integer_program(self, time_costs, scores, total_frames, time_budget, max_seg_length):
        prob = pulp.LpProblem("GeneralSegmentationOptimization", pulp.LpMinimize)
        variables = {}

        for l in range(1, max_seg_length + 1):
            variables[l] = pulp.LpVariable(f"num_{l}", lowBound=0, cat='Integer')

        # Total frames must match
        prob += pulp.lpSum([variables[l] * l for l in variables]) == total_frames
        # Time budget constraint
        prob += pulp.lpSum([variables[l] * time_costs[l] for l in variables]) <= time_budget
        # Objective: minimize total compressed size
        prob += pulp.lpSum([variables[l] * scores[l] for l in variables])

        prob.solve()

        if pulp.LpStatus[prob.status] != 'Optimal':
            raise RuntimeError("No feasible solution found")

        return {l: int(variables[l].varValue) for l in variables if int(variables[l].varValue) > 0}

    def assign_segments_with_gaussian_smoothing(self, pc_files, segment_counts):
        total_frames = len(pc_files)
        # short_seg = min(segment_counts.keys())
        long_seg = max(segment_counts.keys())
        short_seg = long_seg - 1
        num_short = segment_counts.get(short_seg, 0)
        num_long = segment_counts.get(long_seg, 0)

        def lcm(a, b):
            return a * b // gcd(a, b)

        block_size = lcm(short_seg, long_seg)

        # num_blocks = total_frames // block_size

        num_blocks = (num_long * long_seg) // block_size + (num_short * short_seg) // block_size
        self.logger.info("Number of blocks: {}".format(num_blocks))


        remainder_start = num_blocks * block_size

        scores = np.ones(num_blocks)
        selected_mask = np.zeros(num_blocks, dtype=bool)
        assignments = []

        # eta = 50
        # sigma = 1.5

        eta = 20

        sigma = 2

        mode = "weight"
        # mode = 'random'
        print("mode: ", mode)

        deduplicator = Deduplicator(self.args)
        diskmanager = DiskManager(self.args)

        total_performance_gain = 0


        assign_count = 0

        T = 8  # Number of blocks to process in parallel

        while num_long > 0 and not all(selected_mask):
            self.logger.info("num_long: {}".format(num_long))
            if num_long < block_size // long_seg:
                break

            masked_scores = np.where(selected_mask, 0, scores)
            total_score = np.sum(masked_scores)
            if total_score == 0:
                break

            probs = masked_scores / total_score
            # print(probs)

            # Select T blocks based on probabilities
            if num_long < ((T * block_size) // long_seg):
                self.logger.info("Here num_long: {}".format(num_long))
                selected_blocks = np.random.choice(num_blocks, size= ((num_long * long_seg) // block_size), p=probs, replace=False)
            else:
                selected_blocks = np.random.choice(num_blocks, size=T, p=probs, replace=False)
            self.logger.info("Selected blocks: {}".format(selected_blocks))

            block_starts = [selected * block_size for selected in selected_blocks]

            # # Prepare arguments for parallel evaluation
            # args = [
            #     ((block_start, block_start + block_size), block_size, long_seg, short_seg, deduplicator, diskmanager)
            #     for block_start in block_starts
            # ]
            #
            # # Parallel evaluation of blocks
            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     results = list(executor.map(lambda arg: self.evaluate_block(*arg), args))

            # # Prepare arguments for parallel evaluation
            # args = [
            #     (
            #     self, (block_start, block_start + block_size), block_size, long_seg, short_seg, Deduplicator(self.args),
            #     DiskManager(self.args))
            #     for block_start in block_starts
            # ]
            #
            # # Parallel evaluation of blocks
            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     results = list(executor.map(evaluate_block_wrapper, args))

            # Parallel evaluation of blocks with GPU assignment
            gpu_ids = itertools.cycle([0, 1])  # Cycle between GPU 0 and GPU 1
            args_with_gpu = [
                (
                self, (block_start, block_start + block_size), block_size, long_seg, short_seg, Deduplicator(self.args),
                DiskManager(self.args), next(gpu_ids))
                for block_start in block_starts
            ]

            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(evaluate_block_wrapper, args_with_gpu))



            # Process results
            for (segment_result, performance_gain), block_start in zip(results, block_starts):
                self.logger.info("{} {}".format(segment_result, performance_gain))

                # Update total performance gain
                total_performance_gain += performance_gain

                for segment in segment_result:
                    assignments.append(segment)

                avg_performance_gain = total_performance_gain / (assign_count + 1) if assign_count > 0 else 1.0

                reward = (performance_gain - avg_performance_gain) / avg_performance_gain

                # Update scores of neighbors using Gaussian smoothing
                for offset in range(-4, 5):
                    j = block_start // block_size + offset
                    if 0 <= j < num_blocks and not selected_mask[j]:
                        influence = np.exp(-(offset ** 2) / (2 * sigma ** 2))
                        scores[j] += eta * reward * influence
                        scores[j] = max(scores[j], math.exp(-6))  # Ensure non-negative scores

                selected_mask[block_start // block_size] = True

                num_long -= block_size // long_seg
                assign_count += 1



        # Fill remaining blocks with short segments
        used = set(idx for seg in assignments for idx in range(seg[0], seg[1]))

        iterator = 0

        # while iterator < remainder_start and num_short > 0:
        #     if iterator not in used and iterator + short_seg <= remainder_start and all(
        #             x not in used for x in range(iterator, iterator + short_seg)):
        #         assignments.append((iterator, iterator + short_seg))
        #
        #         segment = self.pc_files[iterator:iterator + short_seg]
        #         poses = self.poses[iterator:iterator + short_seg]
        #         score = self.score_segment(segment, deduplicator, diskmanager, poses, gpuid=None, if_store=True)
        #
        #         num_short -= 1
        #         iterator += short_seg
        #     else:
        #         iterator += 1

        tasks = []
        gpu_ids = itertools.cycle([0, 1])  # Cycle between GPU 0 and GPU 1
        MAX_WORKERS = 8

        while iterator < remainder_start and num_short > 0:
            if iterator not in used and iterator + short_seg <= remainder_start and all(
                    x not in used for x in range(iterator, iterator + short_seg)):
                tasks.append((
                    iterator, short_seg, self.pc_files, self.poses, deduplicator, diskmanager, self.score_segment,
                    next(gpu_ids)
                ))
                iterator += short_seg
                num_short -= 1
            else:
                iterator += 1

        # Process tasks in parallel with a limit on the number of workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(process_short_segment, tasks))
            print(results)

        for segment, score in results:
            assignments.append(segment)
            total_performance_gain += score

        # for i in range(0, remainder_start):
        #     if i not in used and i + short_seg <= remainder_start and all(
        #             x not in used for x in range(i, i + short_seg)):
        #         assignments.append((i, i + short_seg))
        #         num_short -= 1

        # Update segment_counts with remaining num_short and num_long
        segment_counts[short_seg] = num_short
        segment_counts[long_seg] = num_long

        iterate = remainder_start
        while iterate < total_frames and any(count > 0 for count in segment_counts.values()):
            print(segment_counts)
            self.logger.info("  Current segment counts: {}".format(segment_counts))
            available_segments = [seg for seg, count in segment_counts.items() if count > 0]
            if not available_segments:
                break
            selected_seg = random.choice(available_segments)
            end = iterate + selected_seg
            self.logger.info("  Selected segment: {}".format([iterate, end]))

            if end <= total_frames and all(x not in used for x in range(iterate, end)):
                self.logger.info("  assign the segment here")
                assignments.append((iterate, end))
                used.update(range(iterate, end))
                segment_counts[selected_seg] -= 1

                segment = self.pc_files[iterate:end]
                poses = self.poses[iterate:end]
                score = self.score_segment(segment, deduplicator, diskmanager, poses, gpuid=None, if_store=True)

            iterate += selected_seg

        if iterate != total_frames:
            raise RuntimeError("Segment assignment did not cover all frames")

        assignments.sort(key=lambda x: x[0])
        print(assignments)
        print("total performance gain:", total_performance_gain)
        self.logger.info("Total performance gain: {}".format(total_performance_gain))
        return assignments

    def load_data(self, segment):
        """
        Dummy method to load data from the given directory.
        In a real scenario, implement this to load your point cloud or other data.
        For this example, we simply return a list of numbers.
        """
        # For demonstration, we return a list of points (or indices).
        # Replace this with your actual data-loading logic.

        pc_data = []

        for file in segment:
            file_path = os.path.join(self.velodyne_dir, file)
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            pc_data.append(points.tolist())

        return pc_data

    def wrapper_compute_segment_score(self, segment, frames, deduplicator, diskmanager, poses, score_segment, gpuid=None, if_store=False):
        with self.semaphore:
            i, j = segment
            seg = frames[i:j]
            seg_poses = poses[i:j]
            return score_segment(seg, deduplicator, diskmanager, seg_poses, gpuid, if_store)

    def compute_total_score(self, segments, frames, deduplicator, diskmanager, poses, if_store=False):
        gpu_ids = itertools.cycle([0, 1])  # Adjust the list of GPU IDs as needed


        # Prepare the arguments with GPU assignment
        args_with_gpu = [
            (segment, frames, deduplicator, diskmanager, poses, self.score_segment, next(gpu_ids), if_store)
            for segment in segments
        ]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            scores = list(executor.map(self._compute_segment_score_wrapper, args_with_gpu))

        total = sum(scores)
        return total

    def _compute_segment_score_wrapper(self, args):
        segment, frames, deduplicator, diskmanager, poses, score_segment, gpuid, if_store = args
        return self.wrapper_compute_segment_score(segment, frames, deduplicator, diskmanager, poses, score_segment, gpuid, if_store)


    def score_segment(self, segment, deduplicator, diskmanager, poses, gpuid=None, if_store=False):
        """
        Computes the score for a given segment.
        Here, we use the segment length as a proxy for the "result size".
        In a real application, replace this with the actual scoring logic.
        """
        # pc_data = []
        #
        # for file in segment:
        #     file_path = os.path.join(self.velodyne_dir, file)
        #     points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        #     pc_data.append(points.tolist())
        # print("if_store", if_store)
        if len(segment) == 1:
            pc_data = self.load_data(segment)
            points = np.array(pc_data[0])
            # print(points.shape)
            header = laspy.LasHeader(point_format=3, version="1.2")  # Adjust point_format as needed

            # Create a LasData object
            las = laspy.LasData(header)
            las.header.x_scale = 0.0001
            las.header.y_scale = 0.0001
            las.header.z_scale = 0.0001

            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]
            las.intensity = (points[:, 3] * self.args.intensity_scale).astype(np.uint16)

            buffer = BytesIO()
            las.write(buffer, do_compress=True, laz_backend=LazBackend.Laszip)
            laz_binary = buffer.getvalue()

            if if_store:
                file = os.path.splitext(segment[0])[0]
                file_name = f"{file}.laz"
                if self.args.workload_aware:
                    file_path = os.path.join(self.args.root_store_path_wa, file_name)
                else:
                    file_path = os.path.join(self.args.root_store_path, file_name)  # Construct the full path
                with open(file_path, "wb") as f:
                    f.write(laz_binary)  # Write the laz_binary to the file
                pass

            return len(laz_binary) / 1024

        pc_data = self.load_data(segment)

        current_poses = poses

        # print("compute score of segment: {}, gpuid: {}".format(segment, gpuid))
        results = deduplicator.deduplication(pc_data, current_poses, self.Tr, self.threshold, gpuid)

        index = Indexer()
        index.create_index(results["deduplicated_patches"], current_poses, self.Tr, self.args)

        # compact_file_name = f"{i + 1}.npz"
        # compact_file_name = "data/" + compact_file_name

        if if_store:
            file_0 = os.path.splitext(segment[0])[0]
            file_1 = os.path.splitext(segment[-1])[0]
            file_name = f"{file_0}-{file_1}.bin"
            if self.args.workload_aware:
                file_path = os.path.join(self.args.root_store_path_wa, file_name)
            else:
                file_path = os.path.join(self.args.root_store_path, file_name)  # Construct the full path
            # file_path = os.path.join(self.args.root_store_path, file_name)
            size = diskmanager.save_2(index, path=file_path)
            pass
        else:
            size = diskmanager.save_2(index, path=None)

        return size

    def evaluate_block(self, block, block_size, long_seg, short_seg, deduplicator, diskmanager, gpu_id=0):
        """Stub reward function for evaluating a block. Replace with your real deduplication + compression logic."""

        block_start = block[0]
        block_end = block[1]

        assignments = []

        score_long_seg = 0
        for i in range((block_end - block_start) // long_seg):
            s = block_start + i * long_seg
            e = s + long_seg
            assignments.append((s, e))
            segment = self.pc_files[s:e]
            poses = self.poses[s:e]
            score = self.score_segment(segment, deduplicator, diskmanager, poses, gpuid=gpu_id, if_store=True)
            score_long_seg += score

        short_seg = 1

        score_short_seg = 0
        for i in range((block_end - block_start) // short_seg):
            s = block_start + i * short_seg
            e = s + short_seg

            segment = self.pc_files[s:e]
            poses = self.poses[s:e]
            score = self.score_segment(segment, deduplicator, diskmanager, poses, gpuid=gpu_id, if_store=False)
            score_short_seg += score

        performance_gain = score_short_seg - score_long_seg

        return assignments, performance_gain

    def evaluate_segment(self, segment):
        """Stub reward function. Replace with your real deduplication + compression logic."""
        return random.uniform(0.5, 1.0)  # Dummy reward


if __name__ == "__main__":
    segmentator = Segmentator()
    time_costs = {1: 0.1 * 1, 2: 1.5 * 2, 3: 2.0 * 3, 4: 2.5 * 4}
    scores = {1: 1, 2: 0.4, 3: 0.54, 4: 0.7}  # Corresponding to 1, 0.2*2, 0.18*3, 0.175*4
    total_frames = 51
    time_budget = 100
    max_seg_length = 4

    result = segmentator.solve_general_integer_program(time_costs, scores, total_frames, time_budget, max_seg_length)
    print("ILP Solution:", result)
