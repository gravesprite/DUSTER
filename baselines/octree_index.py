import os
import numpy as np
from pc_compress_multi import (generate_initial_octree, add_point_cloud_to_shared_octree, save_octree_to_file, load_octree_from_file, regenerate_point_cloud)

import laspy
from laspy.compression import LazBackend
from io import BytesIO
from collections import deque
import time
import multiprocessing


def process_file_pair(args):
    """
    Helper function to process a single pair of files in parallel.

    Args:
        args (tuple): Contains input_dir, output_dir, file1, file2, and other required data.

    Returns:
        tuple: Processed file information (e.g., output file paths).
    """
    input_dir, output_dir, file1, file2, args_obj = args

    input_file1 = os.path.join(input_dir, file1)
    points1 = np.fromfile(input_file1, dtype=np.float32).reshape(-1, 4)
    xyz_points1 = points1[:, :3]
    intensity1 = points1[:, 3]

    if file2:  # Pair two point clouds
        input_file2 = os.path.join(input_dir, file2)
        points2 = np.fromfile(input_file2, dtype=np.float32).reshape(-1, 4)
        xyz_points2 = points2[:, :3]
        intensity2 = points2[:, 3]

        # Compute root bounds
        min_corner = np.min(np.vstack((xyz_points1, xyz_points2)), axis=0)
        max_corner = np.max(np.vstack((xyz_points1, xyz_points2)), axis=0)
        root_bounds = (min_corner, max_corner)

        # Generate octree
        root = generate_initial_octree(xyz_points1, root_bounds, max_depth=15)
        root = add_point_cloud_to_shared_octree(root, xyz_points2, 1, root_bounds, max_depth=15)

        # Save octree
        output_file = f"{file1[:-4]}-{file2[:-4]}.npz"
        output_file_path = os.path.join(output_dir, output_file)
        save_octree_to_file(root, 2, output_file_path)

        # # Save intensity data
        # intensity_file_1 = file1.replace(".bin", "_intensity.laz")
        # intensity_file_2 = file2.replace(".bin", "_intensity.laz")
        # output_intensity_file_1 = os.path.join(output_dir, intensity_file_1)
        # output_intensity_file_2 = os.path.join(output_dir, intensity_file_2)

        # Store intensity data
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)
        las.intensity = (intensity1 * args_obj.intensity_scale).astype(np.uint16)

        buffer = BytesIO()
        las.write(buffer, do_compress=True, laz_backend=LazBackend.Laszip)
        laz_bin_int = buffer.getvalue()

        # Save intensity binary alongside octree
        intensity_file_1 = file1.replace(".bin", "_intensity.laz")
        output_intensity_file_1 = os.path.join(output_dir, intensity_file_1)
        with open(output_intensity_file_1, "wb") as f:
            f.write(laz_bin_int)

        # Store intensity data
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)
        las.intensity = (intensity2 * args_obj.intensity_scale).astype(np.uint16)

        buffer = BytesIO()
        las.write(buffer, do_compress=True, laz_backend=LazBackend.Laszip)
        laz_bin_int = buffer.getvalue()

        # Save intensity binary alongside octree
        intensity_file_2 = file2.replace(".bin", "_intensity.laz")
        output_intensity_file_2 = os.path.join(output_dir, intensity_file_2)
        with open(output_intensity_file_2, "wb") as f:
            f.write(laz_bin_int)

        return output_file, (intensity_file_1, intensity_file_2)
    else:  # Single point cloud
        # Compute root bounds
        min_corner = np.min(xyz_points1, axis=0)
        max_corner = np.max(xyz_points1, axis=0)
        root_bounds = (min_corner, max_corner)

        # Generate octree
        root = generate_initial_octree(xyz_points1, root_bounds, max_depth=15)

        # Save octree
        output_file = f"{file1[:-4]}.npz"
        output_file_path = os.path.join(output_dir, output_file)
        save_octree_to_file(root, 1, output_file_path)

        # # Save intensity data
        # intensity_file = file1.replace(".bin", "_intensity.laz")
        # output_intensity_file = os.path.join(output_dir, intensity_file)

        # Store intensity data
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)
        las.intensity = (intensity1 * args_obj.intensity_scale).astype(np.uint16)

        buffer = BytesIO()
        las.write(buffer, do_compress=True, laz_backend=LazBackend.Laszip)
        laz_bin_int = buffer.getvalue()

        # Save intensity binary alongside octree
        intensity_file = file1.replace(".npz", "_intensity.laz")
        output_intensity_file = os.path.join(output_dir, intensity_file)
        with open(output_intensity_file, "wb") as f:
            f.write(laz_bin_int)

        return output_file, (intensity_file,)

class OctreeIndexer:
    def __init__(self, input_dir, output_dir, args):
        """
        Initialize the OctreeIndexer by compressing point cloud files into octree structures.

        Args:
            input_dir (str): Directory containing input point cloud files.
            output_dir (str): Directory to store compressed octree files.
            args: Additional arguments (e.g., intensity scale).
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.args = args
        self.frames = []  # List of stored file names
        self.root_bounds = []

        if os.path.exists(self.output_dir) and os.listdir(self.output_dir):
            self.load_from_compressed_files()
        else:
            self._compress_files()

    def load_from_compressed_files(self):
        print("load from compressed files.")
        files = sorted([f for f in os.listdir(self.input_dir) if f.endswith(".bin")])
        num_files = len(files)

        for i in range(0, num_files, 2):
            # if i > 15:
            #     break
            file1 = files[i]
            input_file1 = os.path.join(self.input_dir, file1)
            points1 = np.fromfile(input_file1, dtype=np.float32).reshape(-1, 4)
            xyz_points1 = points1[:, :3]
            intensity1 = points1[:, 3]

            if i + 1 < num_files:  # Pair two point clouds
                file2 = files[i + 1]
                input_file2 = os.path.join(self.input_dir, file2)
                points2 = np.fromfile(input_file2, dtype=np.float32).reshape(-1, 4)
                xyz_points2 = points2[:, :3]
                intensity2 = points2[:, 3]

                # Compute root bounds
                min_corner = np.min(np.vstack((xyz_points1, xyz_points2)), axis=0)
                max_corner = np.max(np.vstack((xyz_points1, xyz_points2)), axis=0)
                root_bounds = (min_corner, max_corner)

                self.root_bounds.append(root_bounds)
                self.root_bounds.append(root_bounds)  # input second time

                # # Generate octree
                # root = generate_initial_octree(xyz_points1, root_bounds, max_depth=15)
                # root = add_point_cloud_to_shared_octree(root, xyz_points2, 1, root_bounds, max_depth=15)

                # Save octree
                output_file = f"{file1[:-4]}-{file2[:-4]}.npz"
                output_file_path = os.path.join(self.output_dir, f"{file1[:-4]}-{file2[:-4]}.npz")

                # save_octree_to_file(root, 2, output_file_path)
                #
                # # Store intensity data
                # header = laspy.LasHeader(point_format=3, version="1.2")
                # las = laspy.LasData(header)
                # las.intensity = (intensity1 * self.args.intensity_scale).astype(np.uint16)
                #
                # buffer = BytesIO()
                # las.write(buffer, do_compress=True, laz_backend=LazBackend.Laszip)
                # laz_bin_int = buffer.getvalue()

                # Save intensity binary alongside octree
                intensity_file_1 = file1.replace(".bin", "_intensity.laz")
                output_intensity_file_1 = os.path.join(self.output_dir, intensity_file_1)
                # with open(output_intensity_file_1, "wb") as f:
                #     f.write(laz_bin_int)

                # # Store intensity data
                # header = laspy.LasHeader(point_format=3, version="1.2")
                # las = laspy.LasData(header)
                # las.intensity = (intensity2 * self.args.intensity_scale).astype(np.uint16)
                #
                # buffer = BytesIO()
                # las.write(buffer, do_compress=True, laz_backend=LazBackend.Laszip)
                # laz_bin_int = buffer.getvalue()

                # Save intensity binary alongside octree
                intensity_file_2 = file2.replace(".bin", "_intensity.laz")
                output_intensity_file_2 = os.path.join(self.output_dir, intensity_file_2)
                # with open(output_intensity_file_2, "wb") as f:
                #     f.write(laz_bin_int)

                self.frames.append((output_file, (intensity_file_1, intensity_file_2)))
            else:  # Single point cloud
                # Compute root bounds
                min_corner = np.min(xyz_points1, axis=0)
                max_corner = np.max(xyz_points1, axis=0)
                root_bounds = (min_corner, max_corner)

                self.root_bounds.append(root_bounds)

                # # Generate octree
                # root = generate_initial_octree(xyz_points1, root_bounds, max_depth=15)

                # Save octree
                output_file = f"{file1[:-4]}.npz"
                output_file_path = os.path.join(self.output_dir, f"{file1[:-4]}.npz")
                # save_octree_to_file(root, 1, output_file_path)

                # # Store intensity data
                # header = laspy.LasHeader(point_format=3, version="1.2")
                # las = laspy.LasData(header)
                # las.intensity = (intensity1 * self.args.intensity_scale).astype(np.uint16)
                #
                # buffer = BytesIO()
                # las.write(buffer, do_compress=True, laz_backend=LazBackend.Laszip)
                # laz_bin_int = buffer.getvalue()

                # Save intensity binary alongside octree
                intensity_file = file1.replace(".npz", "_intensity.laz")
                # with open(intensity_file, "wb") as f:
                #     f.write(laz_bin_int)

                self.frames.append((output_file, (intensity_file)))
        pass

    def _compress_files(self):
        """
        Compress point cloud files into octree structures using multiprocessing.
        """
        files = sorted([f for f in os.listdir(self.input_dir) if f.endswith(".bin")])
        num_files = len(files)

        # Prepare arguments for multiprocessing
        args_list = []
        for i in range(0, num_files, 2):
            file1 = files[i]
            file2 = files[i + 1] if i + 1 < num_files else None
            args_list.append((self.input_dir, self.output_dir, file1, file2, self.args))

        # Use multiprocessing to process files in parallel
        # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        with multiprocessing.Pool(processes=12) as pool:
            results = pool.map(process_file_pair, args_list)

        # Collect results
        for result in results:
            self.frames.append(result)

        for i in range(0, num_files, 2):
            # if i > 15:
            #     break
            file1 = files[i]
            input_file1 = os.path.join(self.input_dir, file1)
            points1 = np.fromfile(input_file1, dtype=np.float32).reshape(-1, 4)
            xyz_points1 = points1[:, :3]
            intensity1 = points1[:, 3]

            if i + 1 < num_files:  # Pair two point clouds
                file2 = files[i + 1]
                input_file2 = os.path.join(self.input_dir, file2)
                points2 = np.fromfile(input_file2, dtype=np.float32).reshape(-1, 4)
                xyz_points2 = points2[:, :3]
                intensity2 = points2[:, 3]

                # Compute root bounds
                min_corner = np.min(np.vstack((xyz_points1, xyz_points2)), axis=0)
                max_corner = np.max(np.vstack((xyz_points1, xyz_points2)), axis=0)
                root_bounds = (min_corner, max_corner)

                self.root_bounds.append(root_bounds)
                self.root_bounds.append(root_bounds)  # input second time
            else:  # Single point cloud
                # Compute root bounds
                min_corner = np.min(xyz_points1, axis=0)
                max_corner = np.max(xyz_points1, axis=0)
                root_bounds = (min_corner, max_corner)

                self.root_bounds.append(root_bounds)
        return

    # def _compress_files(self):
    #     """
    #     Compress point cloud files into octree structures and store them with intensity data.
    #     """
    #     files = sorted([f for f in os.listdir(self.input_dir) if f.endswith(".bin")])
    #     num_files = len(files)
    #
    #     for i in range(0, num_files, 2):
    #         # if i > 15:
    #         #     break
    #         file1 = files[i]
    #         input_file1 = os.path.join(self.input_dir, file1)
    #         points1 = np.fromfile(input_file1, dtype=np.float32).reshape(-1, 4)
    #         xyz_points1 = points1[:, :3]
    #         intensity1 = points1[:, 3]
    #
    #         if i + 1 < num_files:  # Pair two point clouds
    #             file2 = files[i + 1]
    #             input_file2 = os.path.join(self.input_dir, file2)
    #             points2 = np.fromfile(input_file2, dtype=np.float32).reshape(-1, 4)
    #             xyz_points2 = points2[:, :3]
    #             intensity2 = points2[:, 3]
    #
    #             # Compute root bounds
    #             min_corner = np.min(np.vstack((xyz_points1, xyz_points2)), axis=0)
    #             max_corner = np.max(np.vstack((xyz_points1, xyz_points2)), axis=0)
    #             root_bounds = (min_corner, max_corner)
    #
    #             self.root_bounds.append(root_bounds)
    #             self.root_bounds.append(root_bounds) # input second time
    #
    #             # Generate octree
    #             root = generate_initial_octree(xyz_points1, root_bounds, max_depth=15)
    #             root = add_point_cloud_to_shared_octree(root, xyz_points2, 1, root_bounds, max_depth=15)
    #
    #             # Save octree
    #             output_file = f"{file1[:-4]}-{file2[:-4]}.npz"
    #             output_file_path = os.path.join(self.output_dir, f"{file1[:-4]}-{file2[:-4]}.npz")
    #             save_octree_to_file(root, 2, output_file_path)
    #
    #             # Store intensity data
    #             header = laspy.LasHeader(point_format=3, version="1.2")
    #             las = laspy.LasData(header)
    #             las.intensity = (intensity1 * self.args.intensity_scale).astype(np.uint16)
    #
    #             buffer = BytesIO()
    #             las.write(buffer, do_compress=True, laz_backend=LazBackend.Laszip)
    #             laz_bin_int = buffer.getvalue()
    #
    #             # Save intensity binary alongside octree
    #             intensity_file_1 = file1.replace(".bin", "_intensity.laz")
    #             output_intensity_file_1 = os.path.join(self.output_dir, intensity_file_1)
    #             with open(output_intensity_file_1, "wb") as f:
    #                 f.write(laz_bin_int)
    #
    #             # Store intensity data
    #             header = laspy.LasHeader(point_format=3, version="1.2")
    #             las = laspy.LasData(header)
    #             las.intensity = (intensity2 * self.args.intensity_scale).astype(np.uint16)
    #
    #             buffer = BytesIO()
    #             las.write(buffer, do_compress=True, laz_backend=LazBackend.Laszip)
    #             laz_bin_int = buffer.getvalue()
    #
    #             # Save intensity binary alongside octree
    #             intensity_file_2 = file2.replace(".bin", "_intensity.laz")
    #             output_intensity_file_2 = os.path.join(self.output_dir, intensity_file_2)
    #             with open(output_intensity_file_2, "wb") as f:
    #                 f.write(laz_bin_int)
    #
    #             self.frames.append((output_file, (intensity_file_1, intensity_file_2)))
    #         else:  # Single point cloud
    #             # Compute root bounds
    #             min_corner = np.min(xyz_points1, axis=0)
    #             max_corner = np.max(xyz_points1, axis=0)
    #             root_bounds = (min_corner, max_corner)
    #
    #             self.root_bounds.append(root_bounds)
    #
    #             # Generate octree
    #             root = generate_initial_octree(xyz_points1, root_bounds, max_depth=15)
    #
    #             # Save octree
    #             output_file = f"{file1[:-4]}.npz"
    #             output_file_path = os.path.join(self.output_dir, f"{file1[:-4]}.npz")
    #             save_octree_to_file(root, 1, output_file_path)
    #
    #             # Store intensity data
    #             header = laspy.LasHeader(point_format=3, version="1.2")
    #             las = laspy.LasData(header)
    #             las.intensity = (intensity1 * self.args.intensity_scale).astype(np.uint16)
    #
    #             buffer = BytesIO()
    #             las.write(buffer, do_compress=True, laz_backend=LazBackend.Laszip)
    #             laz_bin_int = buffer.getvalue()
    #
    #             # Save intensity binary alongside octree
    #             intensity_file = file1.replace(".npz", "_intensity.laz")
    #             with open(intensity_file, "wb") as f:
    #                 f.write(laz_bin_int)
    #
    #             self.frames.append((output_file, (intensity_file)))

    def random_search(self, query_id):
        """
        Perform a random search for a given frame index.

        Args:
            query_id (int): Frame index to search.

        Returns:
            tuple: (pc_data, io_cost, load_time)
        """
        # if query_id < 0 or query_id >= len(self.frames):
        #     raise ValueError(f"Frame index {query_id} is out of bounds.")
        #
        # file_path = self.frames[query_id]
        # start_time = time.time()

        file_index = query_id // 2
        if file_index < 0 or file_index >= len(self.frames):
            raise ValueError(f"Frame index {query_id} is out of bounds.")

        # Get the file path and determine if it's the first or second point cloud
        file_paths = self.frames[file_index]
        is_first_pc = (query_id % 2 == 0)

        octree_file= file_paths[0]
        intensities = file_paths[1]

        start_time = time.time()
        io_cost = 0

        # Load octree and regenerate point cloud
        if "-" in os.path.basename(octree_file):  # Paired point clouds
            octree_file_path = os.path.join(self.output_dir, octree_file)
            root = load_octree_from_file(octree_file_path, 2)
            io_cost += os.path.getsize(octree_file_path) / 1024  # I/O cost in KB

            root_bounds = self.root_bounds[query_id]
            if is_first_pc:
                pc_data = regenerate_point_cloud(root, root_bounds, 0)
                intensity_file = intensities[0]
                intensity_file_path = os.path.join(self.output_dir, intensity_file)
                io_cost += os.path.getsize(intensity_file_path) / 1024  # I/O cost in KB
                with laspy.open(intensity_file_path, mode="r", laz_backend=LazBackend.Laszip) as las_file:
                    las_data = las_file.read()
                    intensity = las_data.intensity / self.args.intensity_scale
                pc_data = np.column_stack((pc_data, intensity))
            else:
                pc_data = regenerate_point_cloud(root, root_bounds, 1)
                intensity_file = intensities[1]
                intensity_file_path = os.path.join(self.output_dir, intensity_file)
                io_cost += os.path.getsize(intensity_file_path) / 1024  # I/O cost in KB
                with laspy.open(intensity_file_path, mode="r", laz_backend=LazBackend.Laszip) as las_file:
                    las_data = las_file.read()
                    intensity = las_data.intensity / self.args.intensity_scale
                pc_data = np.column_stack((pc_data, intensity))
            # pc_data = np.vstack((
            #     regenerate_point_cloud(root, root_bounds, 0),
            #     regenerate_point_cloud(root, root_bounds, 1)
            # ))

        else:  # Single point cloud
            octree_file_path = os.path.join(self.output_dir, octree_file)
            root = load_octree_from_file(octree_file_path, 1)
            io_cost += os.path.getsize(octree_file_path) / 1024  # I/O cost in KB
            root_bounds = self.root_bounds[query_id]
            # root_bounds = self._get_root_bounds(root)
            pc_data = regenerate_point_cloud(root, root_bounds, 0)
            intensity_file = intensities[0]
            intensity_file_path = os.path.join(self.output_dir, intensity_file)
            io_cost += os.path.getsize(intensity_file_path) / 1024  # I/O cost in KB
            with laspy.open(intensity_file_path, mode="r", laz_backend=LazBackend.Laszip) as las_file:
                las_data = las_file.read()
                intensity = las_data.intensity / self.args.intensity_scale
            pc_data = np.column_stack((pc_data, intensity))
        load_time = time.time() - start_time
        # io_cost = os.path.getsize(file_path) / 1024  # I/O cost in KB

        return pc_data, io_cost, load_time

    def sequential_search(self, query_range):
        """
        Perform a sequential search for two point cloud frames stored in the same octree.

        Args:
            query_range (list): List containing start and end indices [start, end].

        Returns:
            tuple: (pc_frames, total_io_cost, total_load_time)
        """
        start_id, end_id = query_range
        start_file_index, end_file_index = start_id // 2, (end_id - 1) // 2
        if start_file_index < 0 or end_file_index >= len(self.frames):
            raise ValueError(f"Frame range {query_range} is out of bounds.")

        start_time = time.time()

        pc_frames = []

        total_io_cost = 0

        query_id = start_id
        while query_id < end_id:
        # for query_id in range(start_id, end_id):
            if query_id % 2 != 0 or query_id == end_id - 1:
                pc_data, io_cost, load_time = self.random_search(query_id)
                total_io_cost += io_cost
                pc_frames.append(pc_data)
                query_id += 1
                continue
            file_index = query_id // 2
            file_paths = self.frames[file_index]
            octree_file = file_paths[0]
            intensities = file_paths[1]
            octree_file_path = os.path.join(self.output_dir, octree_file)
            root = load_octree_from_file(octree_file_path, 2)
            total_io_cost += os.path.getsize(octree_file_path) / 1024  # I/O cost in KB

            root_bounds = self.root_bounds[query_id]
            for i in range(2):
                pc_data = regenerate_point_cloud(root, root_bounds, i)
                intensity_file = intensities[i]
                intensity_file_path = os.path.join(self.output_dir, intensity_file)
                total_io_cost += os.path.getsize(intensity_file_path) / 1024  # I/O cost in KB
                with laspy.open(intensity_file_path, mode="r", laz_backend=LazBackend.Laszip) as las_file:
                    las_data = las_file.read()
                    intensity = las_data.intensity / self.args.intensity_scale
                pc_data = np.column_stack((pc_data, intensity))
                pc_frames.append(pc_data)
            query_id += 2

        load_time = time.time() - start_time
        print(end_id - start_id, len(pc_frames))
        return pc_frames, total_io_cost, load_time

