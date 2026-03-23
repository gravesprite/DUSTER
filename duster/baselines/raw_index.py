import os
import numpy as np
import laspy
from laspy.compression import LazBackend
import time

class RawIndexer:
    def __init__(self, frames, input_dir, args):
        """
        Initialize the RawIndexer with frame file names and input directory.

        Args:
            frames (list): List of frame file names.
            input_dir (str): Path to the directory containing the frame files.
            intensity_scale (float): Scale factor for intensity values.
        """
        self.frames = frames
        self.input_dir = input_dir
        self.intensity_scale = args.intensity_scale

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

        file_name = self.frames[query_id]
        file_path = os.path.join(self.input_dir, file_name)

        start_time = time.time()
        if file_name.endswith('.laz'):
            with laspy.open(file_path, mode="r", laz_backend=LazBackend.Laszip) as las_file:
                las_data = las_file.read()
                intensity = las_data.intensity / self.intensity_scale
                pc_data = np.column_stack((las_data.x, las_data.y, las_data.z, intensity))
                io_cost = os.path.getsize(file_path)  # File size in bytes
        else:
            pc_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            io_cost = os.path.getsize(file_path)  # File size in bytes

        load_time = time.time() - start_time
        io_cost = io_cost / 1024  # Convert to KB

        return pc_data, io_cost, load_time

    def sequential_search(self, query_range):
        """
        Perform a sequential search for a range of frame indices.

        Args:
            query_range (list): List containing start and end indices [start, end].

        Returns:
            tuple: (pc_frames, total_io_cost, total_load_time)
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