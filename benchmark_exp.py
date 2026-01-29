import os
import numpy as np
# from utils.deduplicatation import Deduplicator
from utils.deduplicatation_2 import Deduplicator

from utils.index import Indexer, SequenceIndexer
from utils.disk_manage_las import DiskManager
from utils.logger import create_logger

from utils.segmentation import  RLSegmentator

from utils.segmentation_2 import Segmentator

from configs.config import parse_config

from utils.retrieve_eva import Evaluator
from query.load_query import load_queries


import laspy
from laspy.compression import LazBackend
import shutil

from baselines.laz_index import LazIndex
from baselines.raw_index import RawIndexer
from baselines.octree_index import OctreeIndexer
import json
import time
from scipy.spatial.transform import Rotation as R

def get_total_memory_used(directory):
    total_size = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path) / 1024
    return total_size

def get_transformation_matrix(calib_file):
    """
    Reads the calibration file and extracts the Tr matrix as a 4x4 numpy array.

    Args:
        calib_file (str): Path to the calibration file.

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith("Tr:"):
                # Extract the values after "Tr:"
                values = list(map(float, line.split("Tr:")[1].strip().split()))
                # Reshape into a 3x4 matrix
                tr_matrix = np.array(values).reshape(3, 4)
                # Add the last row [0, 0, 0, 1] to make it 4x4
                tr_matrix = np.vstack((tr_matrix, [0, 0, 0, 1]))
                return tr_matrix
    raise ValueError("Tr matrix not found in the calibration file.")

def once_pose_to_dof(pose):
    """
    Convert ONCE 7D pose (quaternion + translation) to a 12D flattened SE(3) array.
    Format: [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz]
    """
    quat = pose[:4]  # [x, y, z, w]
    trans = pose[4:]  # [tx, ty, tz]

    # Convert quaternion to 3x3 rotation matrix
    rot_mat = R.from_quat(quat).as_matrix()  # shape (3, 3)

    # Flatten into SemanticKITTI-style 12 DoF
    dof = [
        rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], trans[0],
        rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], trans[1],
        rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2], trans[2]
    ]
    return dof

def extract_and_sort_poses(json_data):
    """
    Extracts poses from the JSON data and sorts them by frame_id.

    Args:
        json_data (dict): The JSON data containing frames and poses.

    Returns:
        list: A list of tuples containing frame_id and pose, sorted by frame_id.
    """
    frames = json_data.get("frames", [])
    # Extract frame_id and pose, then sort by frame_id
    # sorted_poses = sorted(
    #     [(frame["frame_id"], frame["pose"]) for frame in frames],
    #     key=lambda x: int(x[0])  # Convert frame_id to integer for sorting
    # )
    sorted_poses = sorted(
        [(frame["frame_id"], frame["pose"]) for frame in frames],
        key=lambda x: int(x[0])  # Convert frame_id to integer for sorting
    )
    poses = []
    for frame in sorted_poses:
        poses.append(once_pose_to_dof(frame[1]))
    return poses

def benchmark(method='duster'):
    args = parse_config()

    if args.data_path == "semantic_kitti":
        base_path = "/home/user1/jiangneng/pcdb/OpenPCDet/data/semantic_kitti/dataset/sequences/{}".format(args.sequence_id)
        poses_file = os.path.join(base_path, "poses.txt")
        calib_file = os.path.join(base_path, "calib.txt")
        velodyne_dir = os.path.join(base_path, "velodyne")

        # Load poses
        with open(poses_file, 'r') as f:
            poses = []
            for line in f:
                dof = list(map(float, line.strip().split()))
                poses.append(dof)

        Tr = get_transformation_matrix(calib_file)

    elif args.data_path == "once":
        base_path = "/mnt/sda4/data/once/data/{}".format(args.sequence_id)

        velodyne_dir = os.path.join(base_path, "lidar_roof")
        pose_file = os.path.join(base_path, "{}.json".format(args.sequence_id))
        with open(pose_file, 'r') as file:
            data = json.load(file)
        poses = extract_and_sort_poses(data)
        Tr = np.eye(4)  # 4x4 identity matrix
    # base_path = "/home/user1/jiangneng/pcdb/OpenPCDet/data/semantic_kitti/dataset/sequences/00"
    else:
        raise ValueError("Invalid data path specified.")


    # store_path = args.root_store_path
    # if os.path.exists(store_path):
    #     shutil.rmtree(store_path)  # Deletes the directory and its contents
    # if not os.path.exists(store_path):
    #     os.makedirs(store_path)

    # Create logger
    logger = create_logger(args.log_file_dir)

    queries = load_queries(args)
    # queries = {
    #     "random": [0,1,2,3,4,5,6,7,8,9,10],
    #     "sequential": [[0,10]]
    # }


    # Load point cloud files
    pc_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    logger.info("**Frame Number**: {}".format(len(pc_files)))

    # pc_files = pc_files[0:300]
    # pc_data = []
    # for pc_file in pc_files:
    #     file_path = os.path.join(velodyne_dir, pc_file)
    #     points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    #     pc_data.append(points.tolist())


    # # Transformation matrix
    # Tr = np.array([
    #     [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02],
    #     [-7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02],
    #     [9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01],
    #     [0, 0, 0, 1]
    # ])


    # result_path = "result"

    result_path = "result_test"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

#    if args.workload_aware:
#        result_file = "{}_{}_{}_wa.json".format(args.data_path, args.sequence_id, args.method)
#    else:
#        result_file = "{}_{}_{}.json".format(args.data_path, args.sequence_id, args.method)

    if args.same_segment:
        logger.info("Start same segmentation with {} each segment".format(args.max_cluster_length))
        result_file = "{}_{}_{}_ss.json".format(args.data_path, args.sequence_id, args.method, args.max_cluster_length)
    else:
        result_file = "{}_{}_{}.json".format(args.data_path, args.sequence_id, args.method)

    result_file_path = os.path.join(result_path, result_file)

    logger.info("start {} method, data: {}, sequence: {}".format(args.method, args.data_path, args.sequence_id))

    if args.method == "duster":
        if args.workload_aware:
            data_store_path = args.root_store_path_wa
        else:
            data_store_path = args.root_store_path

        if os.path.exists(data_store_path):
            shutil.rmtree(data_store_path)  # Deletes the directory and its contents
        if not os.path.exists(data_store_path):
            os.makedirs(data_store_path)

        threshold = args.distance_threshold
        max_length = args.max_cluster_length

        # if args.workload_aware:
        #     result_file = "{}_{}_{}_{}_{}_{}_wa.json".format(args.data_path, args.sequence_id, args.method, args.distance_threshold, args.max_cluster_length, args.relative_absolute)
        # else:
        #     result_file = "{}_{}_{}_{}_{}_{}.json".format(args.data_path, args.sequence_id, args.method, args.distance_threshold, args.max_cluster_length, args.relative_absolute)

        if args.same_segment:
            result_file = "{}_{}_{}_{}_{}_{}_{}_ss.json".format(args.data_path, args.sequence_id, args.method, args.distance_threshold, args.max_cluster_length, args.relative_absolute, args.soft_distance_ratio)
        else:
            result_file = "{}_{}_{}_{}_{}_{}_{}_{}.json".format(args.data_path, args.sequence_id, args.method, args.distance_threshold, args.max_cluster_length, args.relative_absolute, args.time_budget, args.soft_distance_ratio)


        result_file_path = os.path.join(result_path, result_file)

        start_time = time.time()

        segmentator = Segmentator(velodyne_dir, pc_files, queries, logger, poses, Tr, threshold, args, max_length, 4,
                                    max_processes=6, exploration_param=0.1)
        segment_counts, sequence_indexer = segmentator.segment_with_ip_mab(pc_files, 0, args.max_cluster_length)
        # return
        #
        # segmentator = RLSegmentator(velodyne_dir, pc_files, queries, logger, poses, Tr, threshold, args, max_length, 2,
        #                             max_processes=6, exploration_param=0.1)
        # sequence_indexer = segmentator.segment_rl_mcts(pc_files, if_store=True)

        # sequence_indexer = SequenceIndexer([], pc_files, data_store_path, Tr, DiskManager(args), args)
        # sequence_indexer.initialize_from_store(data_store_path, pc_files)

        end_time = time.time()

        construct_time = end_time - start_time

        deduplicated_percentage = sequence_indexer.calculate_deduplicated_percentage(velodyne_dir, data_store_path)

        disk_consume = get_total_memory_used(data_store_path)

        evaluator = Evaluator(queries, logger, velodyne_dir, args)

        metrics = evaluator.evaluate(sequence_indexer, method='opus')

        metrics['segment_counts'] = segment_counts

        metrics['disk_consume'] = disk_consume
        metrics['deduplicated_percentage'] = deduplicated_percentage
        metrics['construct_time'] = construct_time

        logger.info(metrics)

        with open(result_file_path, 'w') as f:
            json.dump(metrics, f, indent=4, separators=(',', ': '))

        pass
    elif args.method == "octree":
        data_store_path = args.root_store_path_octree

        if os.path.exists(data_store_path):
            shutil.rmtree(data_store_path)  # Deletes the directory and its contents
        if not os.path.exists(data_store_path):
            os.makedirs(data_store_path)

        start_time = time.time()

        octree_indexer = OctreeIndexer(velodyne_dir, data_store_path, args)

        end_time = time.time()

        construct_time = end_time - start_time

        disk_consume = get_total_memory_used(data_store_path)

        evaluator = Evaluator(queries, logger, velodyne_dir, args)

        metrics = evaluator.evaluate(octree_indexer)

        metrics['disk_consume'] = disk_consume
        metrics['construct_time'] = construct_time

        logger.info(metrics)

        with open(result_file_path, 'w') as f:
            json.dump(metrics, f, indent=4, separators=(',', ': '))

        pass
    elif args.method == "laz":
        data_store_path = args.root_store_path_laz

        if os.path.exists(data_store_path):
            shutil.rmtree(data_store_path)  # Deletes the directory and its contents
        if not os.path.exists(data_store_path):
            os.makedirs(data_store_path)

        start_time = time.time()

        laz_indexer = LazIndex(velodyne_dir, data_store_path, args, Tr)

        end_time = time.time()

        construct_time = end_time - start_time

        disk_consume = get_total_memory_used(data_store_path)
        evaluator = Evaluator(queries, logger, velodyne_dir, args)
        metrics = evaluator.evaluate(laz_indexer)

        metrics['disk_consume'] = disk_consume
        metrics['construct_time'] = construct_time

        logger.info(metrics)

        with open(result_file_path, 'w') as f:
            json.dump(metrics, f, indent=4, separators=(',', ': '))

        pass
    elif args.method == "raw":

        raw_indexer = RawIndexer(pc_files, velodyne_dir, args)

        evaluator = Evaluator(queries, logger, velodyne_dir, args)
        metrics = evaluator.evaluate(raw_indexer)

        disk_consume = get_total_memory_used(velodyne_dir)
        metrics['disk_consume'] = disk_consume

        logger.info(metrics)

        with open(result_file_path, 'w') as f:
            json.dump(metrics, f, indent=4, separators=(',', ': '))

        pass


    return

    # if not os.path.exists(args.root_store_path_octree):
    #     os.makedirs(args.root_store_path_octree)
    #
    # laz_indexer = LazIndex(velodyne_dir, "data_laz", args, Tr)
    #
    # # raw_indexer = RawIndexer(pc_files, velodyne_dir, args)
    #
    # octree_indexer = OctreeIndexer(velodyne_dir, args.root_store_path_octree, args)
    #
    # queries = load_queries(args)
    #
    # evaluator = Evaluator(queries, logger)
    # disk_consume = get_total_memory_used(args.root_store_path_laz)
    # metrics = evaluator.evaluate(octree_indexer)
    # metrics['disk_consume'] = disk_consume
    #
    # # print(metrics)
    # logger.info(metrics)
    # return

    threshold = 0.03
    max_length = 4

    # segmentator = Segmentator(velodyne_dir, pc_files, queries, poses, Tr, threshold, args)
    #
    # seg_result = segmentator.segment_dp(pc_files, max_length)
    # print(seg_result)

    if args.workload_aware:
        logger.info("Start workload aware situation")
        if not os.path.exists(args.root_store_path_wa):
            os.makedirs(args.root_store_path_wa)

    # segmentator =  RLSegmentator(velodyne_dir, pc_files, queries, logger, poses, Tr, threshold, args, max_length, 3, exploration_param=0.1)
    #
    # # seg_result = segmentator.segment_rl_mcts(pc_files[0:50])
    # # sequence_indexer = segmentator.segment_rl_mcts(pc_files[0:8], if_store=True)
    # sequence_indexer = segmentator.segment_rl_mcts(pc_files, if_store=True)

    # sequence_indexer = SequenceIndexer([], pc_files, args.root_store_path, Tr, DiskManager(args), args)
    # sequence_indexer.initialize_from_store(args.root_store_path, pc_files)

    sequence_indexer = SequenceIndexer([], pc_files, args.root_store_path_wa, Tr, DiskManager(args), args)
    sequence_indexer.initialize_from_store(args.root_store_path_wa, pc_files)

    disk_consume = get_total_memory_used(args.root_store_path)


    evaluator = Evaluator(queries, logger, args)

    metrics = evaluator.evaluate(sequence_indexer, method='opus')

    metrics['disk_consume'] = disk_consume

    # print(metrics)
    logger.info(metrics)

    return


    deduplicator = Deduplicator()
    diskmanager = DiskManager(args)
    frame_records = []

    # for i in range(0, len(pc_data) - 2):
    #     current_pc_data = pc_data[i:i + 3]
    #     current_poses = poses[i:i + 3]
    #
    #     results = deduplicator.deduplication(current_pc_data, current_poses, Tr, threshold)
    #     index = Indexer()
    #     index.create_index(results["deduplicated_patches"], current_poses, Tr)
    #
    #     compact_file_name = f"{i + 1}.npz"
    #     diskmanager.save(index, compact_file_name)
    #     frame_records.append((pc_files[i], compact_file_name))
    #
    # # Save frame records
    # with open("frame_records.txt", "w") as f:
    #     for record in frame_records:
    #         f.write(f"{record[0]} {record[1]}\n")

    if method == 'opus-d':
        # for i in range(0, len(pc_files) - 2):
        i = 0
        while i < int(len(pc_files) / 3) * 3:
            current_pc_data = []
            original_size = 0
            for j in range(3):
                file_path = os.path.join(velodyne_dir, pc_files[i + j])
                original_size += os.path.getsize(file_path)
                points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
                current_pc_data.append(points.tolist())
            print("original size: ", original_size)

            current_poses = poses[i:i + 3]
            # print(len(current_pc_data))
            # print(len(current_poses))
            # print(current_poses)
            results = deduplicator.deduplication(current_pc_data, current_poses, Tr, threshold)
            index = Indexer()
            index.create_index(results["deduplicated_patches"], current_poses, Tr, args)

            compact_file_name = f"{i + 1}.npz"
            compact_file_name = "data/" + compact_file_name
            diskmanager.save_2(index, compact_file_name)
            compressed_size = os.path.getsize(compact_file_name)
            print("compressed size: ", compressed_size)
            print("compression optimization: {} %".format( compressed_size / original_size * 100))
            frame_records.append((pc_files[i], compact_file_name))

            i += 3
            # break

            # Save frame records
        with open("frame_records.txt", "w") as f:
            for record in frame_records:
                f.write(f"{record[0]} {record[1]}\n")
    elif method == 'laz':
        for i in range(int(len(pc_files) / 3) * 3):
            file_path = os.path.join(velodyne_dir, pc_files[i])
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            # print(len(points))
            header = laspy.LasHeader(point_format=3, version="1.2")  # Adjust point_format as needed

            # Create a LasData object
            las = laspy.LasData(header)
            las.header.x_scale = 0.0001
            las.header.y_scale = 0.0001
            las.header.z_scale = 0.0001

            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]
            las.intensity = (points[:, 3] * 65535).astype(np.uint16)

            store_file = os.path.join("data_laz", pc_files[i].replace('.bin', '.laz'))
            # store_file = os.path.join("data_laz", pc_files[i])
            with laspy.open(store_file, mode="w", header=las.header, laz_backend=LazBackend.Laszip) as writer:
                writer.write_points(las.points)
            # break
        pass

    return



if __name__ == '__main__':

    data_path = ""


    benchmark('laz')
