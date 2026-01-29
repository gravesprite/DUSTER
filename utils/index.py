'''
This is the class of functions that will be used to index the deduplicated PC frames.
'''
import numpy as np
import time
import os

import laspy
from laspy.compression import LazBackend

class SequenceIndexer:
    def __init__(self, segmentation_results, frames, root_store_path, Tr, diskmanager, args):
        """
        Initialize the SequenceIndexer with segmentation results, frames, and root storage path.

        Args:
            segmentation_results (list): List of tuples (start_index, end_index, proposals).
            frames (list): List of frame file names.
            root_store_path (str): Path to the directory where files are stored.
        """
        self.segmentations = segmentation_results  # Stores all segments
        self.frames = frames
        self.root_store_path = root_store_path
        self.Tr = Tr
        self.diskmanager = diskmanager
        self.args = args

        results = []
        # Process segmentation results
        for start_index, end_index, proposals in segmentation_results:
            for segment in proposals:
                segment_start, segment_end = segment
                segment_start, segment_end = segment_start + start_index, segment_end  + start_index
                results.append([segment_start, segment_end])
        self.segmentations = results

    def initialize_from_store(self, root_store_path, pc_files):
        """
        Initialize the SequenceIndexer from the store by loading and sorting file names.

        Args:
            root_store_path (str): Path to the root store directory.
            pc_files (list): List of point cloud files.
            Tr (numpy.ndarray): Transformation matrix.
            diskmanager (DiskManager): Disk manager instance.

        Returns:
            SequenceIndexer: An instance of the SequenceIndexer class.
        """
        # Load all file names from the root store path
        file_names = [f for f in os.listdir(root_store_path) if os.path.isfile(os.path.join(root_store_path, f))]

        # Sort the file names
        sorted_files = sorted(file_names)

        # Generate segmentations based on sorted files
        segmentations = []
        for file in sorted_files:
            if '-' in file:  # Check if the file name contains a range
                # Extract start and end indices from the file name
                start_file, end_file = file.split('.')[0].split('-')
                start_file += '.bin'
                end_file += '.bin'

                # Find the indices of the start and end files in the sorted list
                start_index = pc_files.index(start_file)
                end_index = pc_files.index(end_file) + 1  # Exclusive range

                # Append the segmentation
                segmentations.append([start_index, end_index])
            else:
                # Ensure the file ends with .laz
                if not file.endswith('.laz'):
                    raise ValueError(f"File '{file}' does not end with '.laz'")

                # Change the file extension from .laz to .bin
                bin_file = file.replace('.laz', '.bin')

                if bin_file in pc_files:
                    index = pc_files.index(bin_file)
                    segmentations.append([index, index + 1])
                else:
                    raise ValueError(f"File '{bin_file}' does not end in bin files")
        self.segmentations = segmentations
        return

    # Function to calculate the deduplicated percentage in SequenceIndexer
    def calculate_deduplicated_percentage(self, velodyne_dir, output_path):
        """
        Calculate the deduplicated percentage by comparing the total points in the indexers
        with the total points in all frames.

        Returns:
            float: Deduplicated percentage (all points in indexers / all points in frames).
        """
        total_indexer_points = 0
        total_frame_points = 0

        for segment in self.segmentations:  # Assuming self.segments contains the list of segments
            segment_file_name = self.get_segment_file_name(segment)  # Generate the file name for the segment
            file_path = os.path.join(output_path, segment_file_name)  # Construct the full file path
            if '-' in segment_file_name:
                indexer, _ = self.diskmanager.load_2(file_path, Tr=self.Tr)  # Load the indexer
                total_indexer_points += indexer.get_total_points()  # Sum up the points
            else:
                with laspy.open(file_path, mode="r", laz_backend=LazBackend.Laszip) as las_file:
                    las_data = las_file.read()
                    pc_data = np.column_stack(
                        (las_data.x, las_data.y, las_data.z, las_data.intensity / self.args.intensity_scale))
                    total_indexer_points += pc_data.shape[0]  # Sum up the points

        # Load all frames and sum up their points
        for frame_file in self.frames:  # Assuming self.frame_files contains paths to all frame files
            frame_file_path = os.path.join(velodyne_dir, frame_file)
            frame_data = np.fromfile(frame_file_path, dtype=np.float32).reshape(-1, 4)
            total_frame_points += frame_data.shape[0]

        # Calculate deduplicated percentage
        deduplicated_percentage = total_indexer_points / total_frame_points if total_frame_points > 0 else 0

        return deduplicated_percentage

    def get_segment_file_name(self, segment):
        """
        Get the file name for a given segment.

        Args:
            segment (list): A list containing the start and end indices of the segment [start_index, end_index].

        Returns:
            str: The file name corresponding to the segment.
        """
        segment_start, segment_end = segment
        if segment_start < 0 or segment_end > len(self.frames):
            raise ValueError("Segment indices are out of bounds.")

        # Get the start and end frame names
        start_frame = os.path.splitext(self.frames[segment_start])[0]
        end_frame = os.path.splitext(self.frames[segment_end - 1])[0]

        # Construct the file name
        # file_name = f"{start_frame}-{end_frame}.bin"
        # return file_name
        file_name = f"{start_frame}.laz" if segment_start == segment_end - 1 else f"{start_frame}-{end_frame}.bin"
        return file_name

    def random_search(self, query_id):
        # from utils.disk_manage_las import DiskManager
        """
        Perform a random search for a given frame index.

        Args:
            query_id (int): Frame index to search.

        Returns:
            tuple: (pc_data, io_cost, load_time)
        """
        if query_id < 0 or query_id >= len(self.frames):
            raise ValueError(f"Frame index {query_id} is out of bounds.")

        segment = None
        for seg in self.segmentations:
            if seg[0] <= query_id < seg[1]:
                segment = seg
                break

        if segment is None:
            raise ValueError(f"Frame index {query_id} not found in any segment.")

        segment_file_name = self.get_segment_file_name(segment)

        # Get the segment file name for the given query_id
        # segment_file_name = self.get_segment_file_name([query_id, query_id + 1])

        file_path = os.path.join(self.root_store_path, segment_file_name)

        # Load the point cloud data using DiskManager
        start_time = time.time()

        segment_start, segment_end = segment[0], segment[1]

        relative_frame_id = query_id - segment_start
        # print(relative_frame_id, file_path)
        if segment_end - segment_start > 1:
            disk_manager = self.diskmanager
            indexer, io_cost = disk_manager.load_2(file_path, Tr=self.Tr, frame_id=[relative_frame_id])

            pc_data = indexer.search(relative_frame_id)
        else:
            with laspy.open(file_path, mode="r", laz_backend=LazBackend.Laszip) as las_file:
                las_data = las_file.read()
                pc_data = np.column_stack((las_data.x, las_data.y, las_data.z, las_data.intensity / self.args.intensity_scale))
                io_cost = os.path.getsize(file_path)  # Get the file size in bytes
        load_time = time.time() - start_time

        # Calculate I/O cost
        io_cost = io_cost / 1024  # I/O cost in KB

        return pc_data, io_cost, load_time

    def sequential_search(self, sequential_query):
        # from utils.disk_manage_las import DiskManager
        """
        Perform a sequential search for a list of frame ranges.

        Args:
            sequential_query (list): List of frame ranges to search, e.g., [[start1, end1], [start2, end2]].

        Returns:
            tuple: (pc_frames, io_cost, load_time)
        """
        if not sequential_query:
            raise ValueError("Sequential query list is empty.")

        pc_frames = []
        total_io_cost = 0
        total_load_time = 0

        query_range = sequential_query

        start_id, end_id = query_range
        if start_id < 0 or end_id > len(self.frames):
            raise ValueError(f"Frame range {query_range} is out of bounds.")

        # Find all segments overlapping with the query range
        relevant_segments = [
            seg for seg in self.segmentations if seg[0] < end_id and seg[1] > start_id
        ]

        for segment in relevant_segments:
            # Get the segment file name
            segment_file_name = self.get_segment_file_name(segment)
            file_path = os.path.join(self.root_store_path, segment_file_name)

            segment_start, segment_end = segment[0], segment[1]
            # Load the point cloud data using DiskManager
            start_time = time.time()

            if segment_end - segment_start > 1:

                # # Determine the frame IDs to load for this segment
                # frame_ids_to_load = list(range(max(segment[0], start_id), min(segment[1], end_id)))

                frame_ids_to_load = [frame_id - segment[0] for frame_id in
                                     range(max(segment[0], start_id), min(segment[1], end_id))]

                disk_manager = self.diskmanager
                # indexer = disk_manager.load_2(file_path, Tr=self.Tr)
                indexer, io_cost = disk_manager.load_2(file_path, Tr=self.Tr, frame_id=frame_ids_to_load)
                for frame_id in frame_ids_to_load:
                    pc_data = indexer.search(frame_id)
                    pc_frames.append(pc_data)
            else:
                with laspy.open(file_path, mode="r", laz_backend=LazBackend.Laszip) as las_file:
                    las_data = las_file.read()
                    pc_data = np.column_stack((las_data.x, las_data.y, las_data.z, las_data.intensity / self.args.intensity_scale))
                    pc_frames.append(pc_data)
                    io_cost = os.path.getsize(file_path)  # Get the file size in bytes
                    pass
            load_time = time.time() - start_time

            # Accumulate I/O cost and load time
            total_io_cost += io_cost / 1024  # I/O cost in KB
            total_load_time += load_time

        return pc_frames, total_io_cost, total_load_time

class Indexer:
    def __init__(self):
        """
        Parameters:
          segmented_data: a dictionary whose keys are (subset, transposed)
          poses: list of 6-DoF parameters
          Tr: 4x4 calibration matrix
        """
        # self.segmented_data = None # segmented_data
        self.poses = None # poses
        self.Tr = None # Tr
        self.Tr_inv = None # np.linalg.inv(Tr)
        self.args = None

        # Store patch data only once in patch_store, keyed by a unique patch ID.
        self.patch_store = {}
        self.patch_transpose = {}
        self.patch_anchor = {}
        self.patch_subset = {}
        self.intensity_signal = {}  # New attribute to store intensity signal

        self.index = {}

    def create_index(self, segmented_data, poses, Tr, args):
        """
        Build an index linking each frame_id to its corresponding patches.
        Each patch belongs to **all** frames in its subset.
        """

        # self.segmented_data = segmented_data
        self.poses = poses
        self.Tr = Tr
        self.Tr_inv = np.linalg.inv(Tr)
        self.args = args

        patch_id_counter = 0  # Patch ID counter
        unique_patches = {}  # Dictionary to ensure unique patch storage

        for key, patches in segmented_data.items():
            subset, anchor, intensity_signal, transposed = key

            # If key (subset, transposed) is already assigned a patch ID, reuse it
            if key not in unique_patches:
                unique_patches[key] = patch_id_counter
                self.patch_store[patch_id_counter] = patches  # Store the patch
                self.patch_transpose[patch_id_counter] = transposed  # Store the patch
                self.intensity_signal[patch_id_counter] = intensity_signal  # Store the intensity signal
                self.patch_subset[patch_id_counter] = subset  # Store the subset

                # if len(subset) % 2 == 1:
                #     # Odd number of frames: pick the exact center
                #     anchor = subset[len(subset) // 2]
                # else:
                #     # Even number of frames: pick the left of the two centers
                #     anchor = subset[(len(subset) // 2) - 1]
                self.patch_anchor[patch_id_counter] = anchor

                patch_id_counter += 1

            patch_id = unique_patches[key]

            # Assign this patch to **all frames in subset**
            for frame_id in subset:
                if frame_id not in self.index:
                    self.index[frame_id] = []

                # Ensure no duplicate entries
                if patch_id not in self.index[frame_id]:
                    self.index[frame_id].append(patch_id)

    def retranspose_point(self, xyz_array, anchor_id, frame_id):
        """
        Re-transform point from transposed patch back to the reference frame.
        """
        # st_time =time.time()

        anchor_dof = self.poses[anchor_id]
        frame_dof = self.poses[frame_id]

        anchor_rotation = np.array([anchor_dof[0:3], anchor_dof[4:7], anchor_dof[8:11]])
        anchor_translation = np.array([anchor_dof[3], anchor_dof[7], anchor_dof[11]])

        frame_rotation = np.array([frame_dof[0:3], frame_dof[4:7], frame_dof[8:11]])
        frame_translation = np.array([frame_dof[3], frame_dof[7], frame_dof[11]])

        Pose_anchor = np.eye(4)
        Pose_anchor[:3, :3] = anchor_rotation
        Pose_anchor[:3, 3] = anchor_translation

        Pose_frame = np.eye(4)
        Pose_frame[:3, :3] = frame_rotation
        Pose_frame[:3, 3] = frame_translation

        Pose_velodyne_anchor = self.Tr_inv @ Pose_anchor @ self.Tr
        Pose_velodyne_frame = self.Tr_inv @ Pose_frame @ self.Tr

        M = np.linalg.inv(Pose_velodyne_frame) @ Pose_velodyne_anchor

        # xyz = np.array(xyz)
        # new_xyz = M[:3, :3] @ xyz + M[:3, 3]

        # new_xyz = M[:3, :3] @ xyz_array + M[:3, 3]
        new_xyz = (M[:3, :3] @ xyz_array.T).T + M[:3, 3]

        # end_time =time.time()
        # print("finish matrix computation time cost", end_time - st_time)

        # return new_xyz.tolist()
        return new_xyz

    # def retranspose_point(self, xyz, anchor_id, frame_id):
    #     """
    #     Re-transform point from transposed patch back to the reference frame.
    #     """
    #     dof = self.poses[frame_id]
    #     rotation = np.array([dof[0:3], dof[4:7], dof[8:11]])
    #     translation = np.array([dof[3], dof[7], dof[11]])
    #
    #     Pose_camera = np.eye(4)
    #     Pose_camera[:3, :3] = rotation
    #     Pose_camera[:3, 3] = translation
    #
    #     Pose_velodyne = self.Tr_inv @ Pose_camera @ self.Tr
    #     M = np.linalg.inv(Pose_velodyne)
    #
    #     xyz = np.array(xyz)
    #     new_xyz = M[:3, :3] @ xyz + M[:3, 3]
    #     return new_xyz.tolist()

    # Function to calculate the total number of points in the Indexer
    def get_total_points(self):
        """
        Calculate the total number of points in the indexer.

        Returns:
            int: Total number of points across all patches in the indexer.
        """
        total_points = 0
        for patch in self.patch_store.values():
            total_points += patch['xyz'].shape[0]  # Sum up the number of points in each patch
        return total_points

    def search(self, frame_id):
        """
        Reconstruct the PC frame for the given frame_id.
        Instead of transposing points one by one, we apply NumPy-based batch transformations.
        """
        if frame_id not in self.index:
            return np.empty((0, 4))  # Empty array if no patches found

        # print("self.index[frame_id]",self.index[frame_id])

        points_list = []
        for patch_id in self.index[frame_id]:
            patches = self.patch_store[patch_id]  # Get the patch list
            transposed = self.patch_transpose[patch_id]
            anchor_id = self.patch_anchor[patch_id]  # Get the anchor frame ID
            intensity_signal = self.intensity_signal[patch_id]
            subset = self.patch_subset[patch_id]

            frame_index = subset.index(frame_id)

            # st_time = time.time()
            # Convert patches to NumPy array for efficient processing
            # xyz_array = np.array([patch["xyz"] for patch in patches], dtype=np.float32)  # Shape (N, 3)
            xyz_array = patches['xyz']

            # intensity_list = [patch["intensities"] for patch in patches]  # List of intensity lists
            intensity_list = patches['intensities']

            # end_time = time.time()
            # print("time taken to convert patches to numpy array", end_time - st_time)
            # print("xyz_array.shape", xyz_array.shape)

            # Apply transformation only once for all points if transposed
            if transposed:
                if frame_id == anchor_id:
                    # continue
                    pass
                else:
                    # st_time = time.time()
                    # xyz_array = np.apply_along_axis(lambda xyz: self.retranspose_point(xyz, anchor_id, frame_id), 1, xyz_array)
                    xyz_array = self.retranspose_point(xyz_array, anchor_id, frame_id)
                    # end_time = time.time()
                    # print("transpose time cost", end_time - st_time)
            # print("xyz_array.shape_2", xyz_array.shape)
            if intensity_signal:
                # intensities_flattened = np.array([patch["intensities"][0] for patch in patches], dtype=np.float32)
                intensities_flattened = intensity_list[0]
            else:
                # intensities_flattened = np.array([patch["intensities"][frame_index] for patch in patches], dtype=np.float32)
                intensities_flattened = intensity_list[frame_index]

            intensities_flattened = intensities_flattened / self.args.intensity_scale

            frame_points = np.column_stack((xyz_array, intensities_flattened))

            # # Flatten intensities while repeating XYZ coordinates
            # xyz_repeated = np.repeat(xyz_array, [len(intensities) for intensities in intensity_list], axis=0)
            # intensities_flattened = np.concatenate(intensity_list)
            #
            # # Stack XYZ and intensities into final (N, 4) array
            # frame_points = np.column_stack((xyz_repeated, intensities_flattened))

            points_list.append(frame_points)

        return np.vstack(points_list, dtype=np.float32) if points_list else np.empty((0, 4))

    def window_search(self, start_frame_id, end_frame_id):
        """
        Returns reconstructed point clouds for frames in [start_frame_id, end_frame_id].
        """
        return {frame_id: self.search(frame_id) for frame_id in range(start_frame_id, end_frame_id + 1)}

    def range_search(self, frame_id, range):
        """Placeholder for a spatial range search relative to frame_id."""
        pass

    def range_window_search(self, start_frame_id, end_frame_id, range):
        """Placeholder for spatial range search over a window of frames."""
        pass
