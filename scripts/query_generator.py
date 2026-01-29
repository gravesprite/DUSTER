import os
import random
import json

def generate_queries(pc_frames_dir, sequence, random_query_num, sequential_query_num, sequential_length):
    """
    Generate random and sequential queries for point cloud frames.

    Parameters:
        pc_frames_dir (str): Path to the directory containing point cloud frames.
        random_query_num (int): Number of random queries to generate.
        sequential_query_num (int): Number of sequential queries to generate.
        sequential_length (int): Length of each sequential query.
    """
    # Ensure the query directory exists
    query_dir = "../query"
    os.makedirs(query_dir, exist_ok=True)

    # Get the list of point cloud frame files
    pc_frames = sorted(os.listdir(pc_frames_dir))
    total_frames = len(pc_frames)

    if total_frames == 0:
        raise ValueError("No point cloud frames found in the specified directory.")

    # Generate random queries
    random_queries = random.sample(range(total_frames), min(random_query_num, total_frames))

    # Generate sequential queries
    sequential_queries = []
    for _ in range(sequential_query_num):
        start_index = random.randint(0, max(0, total_frames - sequential_length))
        sequential_queries.append([start_index, min(start_index + sequential_length, total_frames)])
        # sequential_query = list(range(start_index, min(start_index + sequential_length, total_frames)))
        # sequential_queries.append(sequential_query)

    # Save the queries to JSON files
    random_query_path = os.path.join(query_dir, "random_queries_{}.json".format(sequence))
    sequential_query_path = os.path.join(query_dir, "sequential_queries_{}.json".format(sequence))

    with open(random_query_path, "w") as random_file:
        json.dump(random_queries, random_file, indent=4)

    with open(sequential_query_path, "w") as sequential_file:
        json.dump(sequential_queries, sequential_file, indent=4)

    print(f"Random queries saved to: {random_query_path}")
    print(f"Sequential queries saved to: {sequential_query_path}")

# generate_queries("/home/user1/jiangneng/pcdb/OpenPCDet/data/semantic_kitti/dataset/sequences/13/velodyne", '13', random_query_num=100, sequential_query_num=10, sequential_length=30)
generate_queries("/mnt/sda4/data/once/data/000062/lidar_roof", '000062', random_query_num=100, sequential_query_num=10, sequential_length=30)