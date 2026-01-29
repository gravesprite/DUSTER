import numpy as np
import pickle
import math
from bitarray import bitarray
from collections import deque
from concurrent.futures import ThreadPoolExecutor

class OctreeNode:
    def __init__(self, depth, is_leaf=False):
        self.is_leaf = is_leaf
        self.depth = depth
        self.children = [0]  # 8-bit child existence bitmask
        self.point_cloud_bits = 0 # (#PC-1)-bit length denoting if there are new child nodes generated
        self.children_nodes = [None] * 8  # References to child nodes
        self.point_count = []  # Stores point counts for each point cloud (if leaf)


class Octree:
    def __init__(self):
        self.root = None
        self.root_bound = None
        self.point_cloud_num = 0

    def construct(self, point_clouds, max_depth=15):
        self.point_cloud_num += len(point_clouds)

        # Calculate the union of the root bounds
        min_corner = np.min([np.min(pc, axis=0) for pc in point_clouds], axis=0)
        max_corner = np.max([np.max(pc, axis=0) for pc in point_clouds], axis=0)
        self.root_bound = (min_corner, max_corner)

        # Generate the initial octree with the union root bounds
        root = generate_initial_octree(point_clouds[0], self.root_bound, max_depth=max_depth)

        # Add the remaining point clouds to the shared octree
        for i in range(1, len(point_clouds)):
            root = add_point_cloud_to_shared_octree(root, point_clouds[i], i, self.root_bound, max_depth=max_depth)

        self.root = root

    def save_to_file(self, filename):
        serialized_data, leaf_data, bits_needed = serialize_octree(self.root, self.point_cloud_num)
        print("octree size:", get_size_in_mb(serialized_data), "MB ", "data size: ", get_size_in_mb(leaf_data), "MB ")
        with open(filename, 'wb') as f:
            # Write the length of the serialized octree data
            f.write(len(serialized_data).to_bytes(4, 'little'))
            # Write the length of the leaf data
            f.write(len(leaf_data).to_bytes(4, 'little'))
            # Write the serialized octree data
            f.write(serialized_data)
            # Write the leaf data
            f.write(leaf_data)
            # Write the number of bits needed
            f.write(bits_needed.to_bytes(1, 'little'))
            # Write the point cloud number
            f.write(self.point_cloud_num.to_bytes(4, 'little'))
            # Write the root bound
            for value in self.root_bound:
                f.write(np.float32(value).tobytes())

    @staticmethod
    def load_from_file(filename):
        with open(filename, 'rb') as f:
            # Read the length of the serialized octree data
            octree_length = int.from_bytes(f.read(4), 'little')
            # Read the length of the leaf data
            leaf_length = int.from_bytes(f.read(4), 'little')
            # Read the serialized octree data
            serialized_data = f.read(octree_length)
            # Read the leaf data
            leaf_data = f.read(leaf_length)
            # Read the number of bits needed
            bits_needed = int.from_bytes(f.read(1), 'little')
            # Read the point cloud number
            point_cloud_num = int.from_bytes(f.read(4), 'little')
            # Read the root bound
            root_bound = []
            for _ in range(2):
                bound = []
                for _ in range(3):
                    bound.append(np.frombuffer(f.read(4), dtype=np.float32)[0])
                root_bound.append(tuple(bound))
            root_bound = tuple(root_bound)

        root = deserialize_octree(serialized_data, leaf_data, bits_needed, point_cloud_num)
        octree = Octree()
        octree.root = root
        octree.root_bound = root_bound
        octree.point_cloud_num = point_cloud_num
        return octree

    def regenerate_point_cloud(self, pc_id):
        points = []

        def traverse(node, bounds):
            if node.is_leaf:
                if node.point_cloud_bits & (1 << pc_id):
                    points.extend(generate_points_within_bounds(bounds, node.point_count[pc_id]))
            else:
                for i in range(8):
                    if node.children[0] & (1 << i):
                        child_bounds = compute_child_bounds(bounds, i)
                        traverse(node.children_nodes[i], child_bounds)

        traverse(self.root, self.root_bound)
        return np.array(points)


def compute_child_bounds(bounds, index):
    """Computes the bounding box of a child based on its index within the parent node."""
    min_corner, max_corner = bounds
    center = [(min_corner[i] + max_corner[i]) / 2 for i in range(3)]
    new_min = list(min_corner)
    new_max = list(max_corner)
    for i in range(3):
        if index & (1 << i):
            new_min[i] = center[i]
        else:
            new_max[i] = center[i]
    return (tuple(new_min), tuple(new_max))


def generate_initial_octree(point_cloud, bounds, max_depth, depth=0):
    """
    Generate the octree for the first point cloud.
    """
    # point_cloud = np.asarray(point_cloud)

    # if depth >= max_depth or len(point_cloud) <= 1:
    if depth >= max_depth or len(point_cloud) < 1:
        # Create a leaf node
        leaf_node = OctreeNode(depth=depth, is_leaf=True)
        leaf_node.point_count = [len(point_cloud)]  # Store point count for this cloud
        leaf_node.point_cloud_bits = 1
        return leaf_node

    # Initialize node
    node = OctreeNode(depth=depth, is_leaf=False)

    node.point_cloud_bits = 1  # Set bit for first point cloud

    # Divide points among children
    center = [(bounds[0][i] + bounds[1][i]) / 2 for i in range(3)]

    child_points = [[] for _ in range(8)]

    for point in point_cloud:
        index = 0
        if point[0] > center[0]: index |= 1
        if point[1] > center[1]: index |= 2
        if point[2] > center[2]: index |= 4
        child_points[index].append(point)

    # Generate child nodes recursively
    for i in range(8):
        if len(child_points[i]) > 0:
            child_bounds = compute_child_bounds(bounds, i)
            node.children[0] |= (1 << i)  # Set child bitmask
            node.children_nodes[i] = generate_initial_octree(child_points[i], child_bounds, max_depth, depth + 1)

    return node

def add_point_cloud_to_shared_octree(root, point_cloud, pc_id, bounds, max_depth, depth=0):
    """
    Merge a new point cloud into the existing shared octree.
    """
    # if depth >= max_depth or len(point_cloud) <= 1:
    if depth >= max_depth or len(point_cloud) < 1:
        if root.is_leaf:
            # Update point count for this point cloud
            root.point_count.append(len(point_cloud))

            root.point_cloud_bits |= (1 << pc_id)

        # else:
        #     print("there is actually situation into this")
        #     # Create a new leaf node if none exists
        #     leaf_node = OctreeNode(depth=depth, is_leaf=True)
        #
        #     # only need to set 0 to the first point cloud, the other is appended, for future change
        #     leaf_node.point_count = [0] + [len(point_cloud)]
        #     # change the bit noting that this point cloud has points
        #     leaf_node.point_cloud_bits |= (1 << pc_id)
        #     return leaf_node
        return root

    # if root.is_leaf:
    #     print("there is actually situation into this")
    #     # If the existing node is a leaf, split it
    #     new_node = OctreeNode(depth=depth, is_leaf=False)
    #     new_node.point_count = root.point_count + [0]  # Expand for new point cloud
    #     return new_node

    # Divide points among children
    center = [(bounds[0][i] + bounds[1][i]) / 2 for i in range(3)]
    child_points = [[] for _ in range(8)]

    for point in point_cloud:
        index = 0
        if point[0] > center[0]: index |= 1
        if point[1] > center[1]: index |= 2
        if point[2] > center[2]: index |= 4
        child_points[index].append(point)

    # create an 8-bit bitmap for the added point cloud
    new_children_bits = 0
    new_node_flag = 0

    # Traverse or create child nodes
    for i in range(8):
        if len(child_points[i]) > 0:
            # Update the bitmap
            new_children_bits |= (1 << i)

            child_bounds = compute_child_bounds(bounds, i)
            if root.children[0] & (1 << i):
                # Child exists, merge recursively
                root.children_nodes[i] = add_point_cloud_to_shared_octree(
                    root.children_nodes[i], child_points[i], pc_id, child_bounds, max_depth, depth + 1
                )
            else:
                new_node_flag = 1
                # Create new child node
                root.children_nodes.append(
                    generate_initial_octree(
                        child_points[i], child_bounds, max_depth, depth + 1
                    )
                )

    # Create a new bitmap
    # if new_node_flag:
    root.children.append(new_children_bits)
    root.point_cloud_bits |= (1 << pc_id)

    return root

def calculate_bits_needed(max_count):
    """
    Calculate the number of bits needed to store the maximum count.
    Args:
        max_count: The maximum count of points in any leaf node.
    Returns:
        The number of bits needed to store the maximum count.
    """
    return max(1, math.ceil(math.log2(max_count + 1)))


def serialize_octree(root, total_point_clouds):
    """
    Serializes the octree into a compact format.
    Args:
        root: Root of the shared octree.
        total_point_clouds: Total number of point clouds.
    Returns:
        A byte array representing the serialized octree and leaf data.
    """
    serialized_data = bitarray()
    leaf_data = bitarray()
    queue = deque([root])

    # Determine the maximum point count
    max_count = 0
    node_count = 0
    while queue:
        node_count+=1
        node = queue.popleft()
        if node.is_leaf:
            max_count = max(max_count, *node.point_count)
        else:
            for child in node.children_nodes:
                if child:
                    queue.append(child)
    print("octree node count: ", node_count)
    bits_needed = calculate_bits_needed(max_count)

    queue.append(root)

    while queue:
        node = queue.popleft()

        # Serialize node type (1 bit: 0 for leaf, 1 for non-leaf)
        serialized_data.append(1 if not node.is_leaf else 0)

        # Serialize point cloud bitmask
        pc_bits = bin(node.point_cloud_bits)[2:].zfill(total_point_clouds)
        serialized_data.extend(pc_bits)

        # Serialize children bitmask (8 bits for non-leaf nodes)
        if not node.is_leaf:
            if (pc_bits == '01' and len(node.children) == 1) or (pc_bits == '11' and len(node.children) == 2):
                pass
            else:
                print(pc_bits, len(node.children))

            for child in node.children:
                # serialized_data.frombytes(bytes(child))
                serialized_data.frombytes((child).to_bytes(1, byteorder='big'))

        if node.is_leaf:
            if (pc_bits == '01' and len(node.point_count) == 1) or (pc_bits == '11' and len(node.point_count) == 2):
                pass
            else:
                print(pc_bits, len(node.point_count))
            # Append point counts to separate data for leaf nodes
            for count in node.point_count:
                leaf_data.extend(bin(count)[2:].zfill(bits_needed))  # Store each count using the minimum bits needed
        else:
            # Add child nodes to the queue for BFS traversal
            for child in node.children_nodes:
                if child:
                    queue.append(child)

    return serialized_data.tobytes(), leaf_data.tobytes(), bits_needed

# def deserialize_octree(serialized_data, leaf_data, bits_needed, total_point_clouds):
#     """
#     Deserializes the octree from serialized data.
#     Args:
#         serialized_data: Byte array containing the serialized octree.
#         leaf_data: Byte array containing point counts for leaf nodes.
#         bits_needed: Number of bits used to store each point count.
#         total_point_clouds: Total number of point clouds.
#     Returns:
#         The root of the deserialized octree.
#     """
#     bits = bitarray()
#     bits.frombytes(serialized_data)
#     leaf_bits = bitarray()
#     leaf_bits.frombytes(leaf_data)
#     bit_index = 0
#     leaf_idx = 0
#
#     root = OctreeNode(depth=0)
#     queue = deque([(root, 0)])  # (node, depth)
#
#     while queue:
#         node, depth = queue.popleft()
#
#         # Read node type (1 bit)
#         is_leaf = bits[bit_index] == 0
#         bit_index += 1
#         node.is_leaf = is_leaf
#
#         if is_leaf:
#             # Read point counts for all point clouds based on pc_bits
#             node.point_count = []
#             pc_bits = bits[bit_index:bit_index + total_point_clouds]
#             bit_index += total_point_clouds
#             node.point_cloud_bits = int(pc_bits.to01(), 2)
#             for i in range(total_point_clouds):
#                 if pc_bits[i] == 1:
#                     count = int(leaf_bits[leaf_idx:leaf_idx + bits_needed].to01(), 2)
#                     node.point_count.append(count)
#                     leaf_idx += bits_needed
#                 else:
#                     node.point_count.append(0)
#         else:
#             # Non-leaf node: Read children bitmask (8 bits)
#             children_mask = int(bits[bit_index:bit_index + 8].to01(), 2)
#             bit_index += 8
#             node.children[0] = children_mask
#
#             # Read point cloud bitmask
#             pc_bits = bits[bit_index:bit_index + total_point_clouds]
#             node.point_cloud_bits = int(pc_bits.to01(), 2)
#             bit_index += total_point_clouds
#
#             # Create child nodes and enqueue them for BFS traversal
#             for i in range(8):
#                 if children_mask & (1 << i):
#                     child = OctreeNode(depth=depth + 1)
#                     node.children_nodes[i] = child
#                     queue.append((child, depth + 1))
#
#     return root

def deserialize_octree(serialized_data, leaf_data, bits_needed, total_point_clouds):
    """
    Deserializes the octree from serialized data.
    Args:
        serialized_data: Byte array containing the serialized octree.
        leaf_data: Byte array containing point counts for leaf nodes.
        bits_needed: Number of bits used to store each point count.
        total_point_clouds: Total number of point clouds.
    Returns:
        The root of the deserialized octree.
    """
    bits = bitarray()
    bits.frombytes(serialized_data)
    leaf_bits = bitarray()
    leaf_bits.frombytes(leaf_data)
    bit_index = 0
    leaf_idx = 0

    root = OctreeNode(depth=0)
    queue = deque([(root, 0)])  # (node, depth)

    while queue:
        node, depth = queue.popleft()
        # print(len(queue), depth)
        # Read is_leaf bit
        is_leaf = bits[bit_index] == 0
        bit_index += 1
        node.is_leaf = is_leaf

        # Read point cloud bitmask
        pc_bits = bits[bit_index:bit_index + total_point_clouds]
        bit_index += total_point_clouds
        node.point_cloud_bits = int(pc_bits.to01(), 2)

        # Check if there are new child nodes
        has_new_child_nodes = node.point_cloud_bits & (1 << (total_point_clouds - 1))

        if is_leaf:
            # Leaf node: Read point counts for all point clouds
            node.point_count = []
            if pc_bits[0] == 1:
                count = int(leaf_bits[leaf_idx:leaf_idx + bits_needed].to01(), 2)
                node.point_count.append(count)
                leaf_idx += bits_needed
                count = int(leaf_bits[leaf_idx:leaf_idx + bits_needed].to01(), 2)
                node.point_count.append(count)
                leaf_idx += bits_needed
            else:
                count = int(leaf_bits[leaf_idx:leaf_idx + bits_needed].to01(), 2)
                node.point_count.append(count)
                leaf_idx += bits_needed
            # for i in range(total_point_clouds):
            #     if pc_bits[i] == 1:
            #         count = int(leaf_bits[leaf_idx:leaf_idx + bits_needed].to01(), 2)
            #         node.point_count.append(count)
            #         leaf_idx += bits_needed
            #     else:
            #         node.point_count.append(0)
        else:
            # Non-leaf node: Load child information
            if has_new_child_nodes:
                # Read two children bitmasks (8 bits each)
                children_mask_0 = int(bits[bit_index:bit_index + 8].to01(), 2)
                bit_index += 8
                node.children[0] = children_mask_0

                children_mask_1 = int(bits[bit_index:bit_index + 8].to01(), 2)
                bit_index += 8
                node.children.append(children_mask_1)
                # Create child nodes and enqueue them for BFS traversal
                for i in range(8):
                    if node.children[0] & (1 << i):
                        child = OctreeNode(depth=depth + 1)
                        node.children_nodes[i] = child
                        queue.append((child, depth + 1))
                for i in range(8):
                    if (not node.children[0] & (1 << i)) and node.children[1] & (1 << i):
                        child = OctreeNode(depth=depth + 1)
                        node.children_nodes.append(child)
                        # node.children_nodes[i] = child
                        queue.append((child, depth + 1))
            else:
                # No new child nodes, skip processing
                children_mask = int(bits[bit_index:bit_index + 8].to01(), 2)
                bit_index += 8
                node.children[0] = children_mask
                for i in range(8):
                    if node.children[0] & (1 << i):
                        child = OctreeNode(depth=depth + 1)
                        node.children_nodes[i] = child
                        queue.append((child, depth + 1))
                pass

            # # Non-leaf node: Load child information
            # if has_new_child_nodes:
            #     # Load two child bits
            #     child_bits = bits[bit_index:bit_index + 2]
            #     bit_index += 2
            #     child_mask = int(child_bits.to01(), 2)
            # else:
            #     # Load one child bit
            #     child_mask = bits[bit_index]
            #     bit_index += 1
            #
            # if child_mask:
            #     # Read children bitmask (8 bits)
            #     children_mask = int(bits[bit_index:bit_index + 8].to01(), 2)
            #     bit_index += 8
            #     node.children[0] = children_mask
            #
            #     # Create child nodes and enqueue them for BFS traversal
            #     for i in range(8):
            #         if children_mask & (1 << i):
            #             child = OctreeNode(depth=depth + 1)
            #             node.children_nodes[i] = child
            #             queue.append((child, depth + 1))

    return root

def get_size_in_mb(data):
    """
    Returns the size of the given data in megabytes (MB).
    Args:
        data: The binary data to measure.
    Returns:
        The size of the data in MB.
    """
    size_in_bytes = len(data)
    size_in_mb = size_in_bytes / (1024 * 1024)
    return size_in_mb

def save_octree_to_file(root, pc_num, filename):
    """
    Saves the serialized octree and leaf data to a binary file.
    Args:
        root: Root of the shared octree.
        filename: Name of the file to save the octree.
    """
    serialized_data, leaf_data, bits_needed = serialize_octree(root, pc_num)
    print("octree size:", get_size_in_mb(serialized_data), "MB ", "data size: ", get_size_in_mb(leaf_data), "MB ")
    with open(filename, 'wb') as f:
        # Write the length of the serialized octree data
        f.write(len(serialized_data).to_bytes(4, 'little'))
        # Write the length of the leaf data
        f.write(len(leaf_data).to_bytes(4, 'little'))
        # Write the serialized octree data
        f.write(serialized_data)
        # Write the leaf data
        f.write(leaf_data)
        # Write the number of bits needed
        f.write(bits_needed.to_bytes(1, 'little'))

def load_octree_from_file(filename, pc_num):
    """
    Loads the octree and leaf data from a binary file.
    Args:
        filename: Name of the file to load the octree from.
    Returns:
        The root of the deserialized octree.
    """
    with open(filename, 'rb') as f:
        # Read the length of the serialized octree data
        octree_length = int.from_bytes(f.read(4), 'little')
        # Read the length of the leaf data
        leaf_length = int.from_bytes(f.read(4), 'little')
        # Read the serialized octree data
        serialized_data = f.read(octree_length)
        # Read the leaf data
        leaf_data = f.read(leaf_length)
        # Read the number of bits needed
        bits_needed = int.from_bytes(f.read(1), 'little')
        # print(bits_needed)

    return deserialize_octree(serialized_data, leaf_data, bits_needed, pc_num)


def generate_points_within_bounds(bounds, point_count):
    """Generates points randomly within the given bounding box."""
    min_corner, max_corner = bounds
    center = [(min_corner[i] + max_corner[i]) / 2 for i in range(3)]
    return np.tile(center, (point_count, 1)).astype(np.float32)
    # return np.random.uniform(min_corner, max_corner, (point_count, 3)).astype(np.float32)


def regenerate_point_cloud(root, root_bounds, pc_id):
    """
    Regenerates a point cloud based on the octree and pc_id.
    Args:
        root: Root of the shared octree.
        root_bounds: Bounding box of the root node.
        pc_id: ID of the point cloud to regenerate.
    Returns:
        A numpy array representing the regenerated point cloud.
    """
    points = []

    def traverse(node, bounds, pc_id):
        if node.is_leaf:
            # print(node.point_cloud_bits)
            if node.point_cloud_bits & (1 << pc_id):
                # Generate points within this leaf node's bounds
                points.extend(generate_points_within_bounds(bounds, node.point_count[pc_id]))
        else:
            # # Calculate and recurse for each child node
            # for i in range(8):
            #     if node.children[0] & (1 << i):  # If this child exists
            #         child_bounds = compute_child_bounds(bounds, i)
            #         traverse(node.children_nodes[i], child_bounds)

            if pc_id == 0:
                for i in range(8):
                    if node.children[0] & (1 << i):  # If this child exists
                        child_bounds = compute_child_bounds(bounds, i)
                        traverse(node.children_nodes[i], child_bounds, pc_id)
            else:
                ind = 0
                for i in range(8):
                    if node.children[1] & (1 << i):  # If this child exists
                        if node.children[0] & (1 << i):
                            child_bounds = compute_child_bounds(bounds, i)
                            traverse(node.children_nodes[i], child_bounds, pc_id)
                        else:
                            child_bounds = compute_child_bounds(bounds, i)
                            # print(len(node.children_nodes))
                            # print(8 + i)
                            traverse(node.children_nodes[8 + ind], child_bounds, pc_id=0)
                            ind += 1
                pass

    traverse(root, root_bounds, pc_id)
    return np.array(points)


def points_to_bin(points, file_name):
    try:
        # Load the .xyz file
        # points = np.loadtxt(xyz_filename, dtype=np.float32)

        # Ensure it has the correct shape (N, 3)
        if points.shape[1] != 3:
            points.tofile(file_name)
            return

        points = np.hstack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))

        # Create the output filename by changing the extension to .bin
        bin_filename = file_name# + '.bin'

        # Save the points as a binary file
        points.tofile(bin_filename)

        retmessage = 'Saved: ' + bin_filename
    except Exception as e:
        retmessage = 'Error: ' + str(e)

    return retmessage