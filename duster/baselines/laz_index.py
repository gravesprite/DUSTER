import os
import numpy as np
import laspy
from laspy.compression import LazBackend
import time
import shutil
class LazIndex:
    def __init__(self, input_dir, output_dir, args, Tr):
        """
        Initialize the LazIndex by compressing all files in the input directory.

        Args:
            input_dir (str): Directory containing the input point cloud files.
            output_dir (str): Directory to store the compressed LAZ files.
            Tr (np.ndarray): Transformation matrix for coordinate transformations.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.args = args
        self.Tr = Tr
        self.segmentations = []  # Stores all segments
        self.frames = []  # List of frame file names

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self._compress_files()

    def _compress_files(self):
        """
        Compress all point cloud files in the input directory into LAZ format.
        """

        for file_name in sorted(os.listdir(self.input_dir)):

            if file_name.endswith(".bin"):
                input_file = os.path.join(self.input_dir, file_name)
                compressed_file = os.path.join(self.output_dir, file_name.replace(".bin", ".laz"))
                self.frames.append(compressed_file)

                # Read binary file
                point_data = np.fromfile(input_file, dtype=np.float32).reshape(-1, 4)

                # Create LAS header
                header = laspy.LasHeader(point_format=3, version="1.2")
                header.x_scale, header.y_scale, header.z_scale = 0.0001, 0.0001, 0.0001

                # Create LAS data
                las = laspy.LasData(header)
                las.x, las.y, las.z = point_data[:, 0], point_data[:, 1], point_data[:, 2]
                las.intensity = (point_data[:, 3] * self.args.intensity_scale).astype(np.uint16)

                # Write to LAZ file
                with laspy.open(compressed_file, mode="w", header=las.header, laz_backend=LazBackend.Laszip) as writer:
                    writer.write_points(las.points)

                # Add segment information
                self.segmentations.append([len(self.frames) - 1, len(self.frames)])

    def random_search(self, query_id):
        """
        Perform a random search for a given frame index.

        Args:
            query_id (int): Frame index to search.

        Returns:
            tuple: (pc_data, io_cost, load_time)
        """
        if query_id < 0 or query_id >= len(self.frames):
            raise ValueError(f"Frame index {query_id} is out of bounds.")

        file_path = self.frames[query_id]
        start_time = time.time()

        # Read the LAZ file
        with laspy.open(file_path, mode="r", laz_backend=LazBackend.Laszip) as las_file:
            las_data = las_file.read()
            intensity = las_data.intensity / self.args.intensity_scale
            pc_data = np.column_stack((las_data.x, las_data.y, las_data.z, intensity))
        print(pc_data)
        load_time = time.time() - start_time
        io_cost = os.path.getsize(file_path) / 1024  # I/O cost in KB

        return pc_data, io_cost, load_time

    def sequential_search(self, query_range):
        """
        Perform a sequential search for a range of frame indices.

        Args:
            query_range (list): List of frame indices to search, e.g., [start, end].

        Returns:
            tuple: (pc_frames, io_cost, load_time)
        """
        start_id, end_id = query_range
        if start_id < 0 or end_id > len(self.frames):
            raise ValueError(f"Frame range {query_range} is out of bounds.")

        pc_frames = []
        total_io_cost = 0
        total_load_time = 0

        for query_id in range(start_id, end_id):
            pc_data, io_cost, load_time = self.random_search(query_id)
            pc_frames.append(pc_data)
            total_io_cost += io_cost
            total_load_time += load_time

        return pc_frames, total_io_cost, total_load_time