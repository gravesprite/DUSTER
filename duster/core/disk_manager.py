import laspy
from laspy.compression import LazBackend
import numpy as np
import struct
import pickle
from io import BytesIO
import os
import pymorton
import time

from duster.core.indexer import Indexer


class DiskManager:
    def __init__(self, args):
        self.args = args
        pass

    def patch_to_laz_binary(self, patch_list, pid=0, scale=[0.0001, 0.0001, 0.0001]):
        """
        Convert a list of patches into LAZ binary blocks in memory.

        Each patch in the list is a dictionary with:
          "xyz": a list of three floats (e.g. [x, y, z])
          "intensities": a list of intensity values.

        Steps:
          1. Combine all xyz and intensity values from the patches.
          2. Create a LAS header using the minimum xyz (for offset) and the provided scale.
          3. Write the geometry (x, y, z) only to a LAZ binary block (with intensity set to 0).
          4. For each intensity channel, create a LAS data object where the intensity
             field is set from that channel and the x, y, z fields are set to 0.
          5. Return a dictionary containing:
                { "laz_xyz": <binary geometry data>,
                  "laz_intensities": [ <binary intensity data for channel 1>, <... for channel 2>, ... ] }
        """
        num_intensity_channels = len(patch_list["intensities"])
        xyz = patch_list['xyz']
        intensities = np.array(patch_list["intensities"], dtype=np.float64).reshape(-1, num_intensity_channels)

        # Repeat xyz for each point.
        n_points = intensities.shape[0]

        size = 0

        combined = np.hstack((xyz, intensities))

        # Create LAS header.
        header = laspy.LasHeader(point_format=3, version="1.2")

        # ----- Geometry-only LAZ binary -----
        las = laspy.LasData(header)

        las.header.x_scale = scale[0]
        las.header.y_scale = scale[1]
        las.header.z_scale = scale[2]

        las.x = combined[:, 0]
        las.y = combined[:, 1]
        las.z = combined[:, 2]

        buffer = BytesIO()
        las.write(buffer, do_compress=True, laz_backend=LazBackend.Laszip)
        laz_binary_xyz = buffer.getvalue()

        size += len(laz_binary_xyz) / 1024

        # ----- Intensity channels LAZ binary -----
        laz_binary_intensities = []
        for i in range(num_intensity_channels):
            las = laspy.LasData(header)
            # Set intensity from the (3 + i)-th column.
            intens_channel = (combined[:, 3 + i] * self.args.intensity_scale).astype(np.uint16)

            las.intensity = intens_channel

            buffer = BytesIO()
            las.write(buffer, do_compress=True, laz_backend=LazBackend.Laszip)
            laz_bin_int = buffer.getvalue()
            laz_binary_intensities.append(laz_bin_int)
            size += len(laz_bin_int) / 1024

        return size, {"laz_xyz": laz_binary_xyz, "laz_intensities": laz_binary_intensities}

    def save(self, indexer, path="index_with_patches.bin"):
        """
        Save the deduplicated patches and meta index to disk.

        For each patch in the indexer's patch_store, convert the patch into a LAZ binary block
        using the method above. The meta information (index) is serialized with struct.pack.

        The final data is stored in a dictionary with keys:
          - "meta_data": binary meta information mapping frame_id -> list of (patch_id, transposed)
          - "patch_laz": dict mapping patch_id -> LAZ binary block (bytes)
          - "patch_transpose": dict mapping patch_id -> transposed flag (True/False)
        """
        meta_index = indexer.index
        patch_store = indexer.patch_store
        patch_transpose = indexer.patch_transpose

        # --- Serialize meta information ---
        meta_data = bytearray()
        frame_ids = list(meta_index.keys())
        meta_data += struct.pack("I", len(frame_ids))
        for fid in frame_ids:
            patches = meta_index[fid]
            meta_data += struct.pack("I", fid)
            meta_data += struct.pack("I", len(patches))
            for (patch_id, transposed) in patches:
                meta_data += struct.pack("I", patch_id)
                meta_data += struct.pack("B", 1 if transposed else 0)

        # For each patch, convert it into a LAZ binary.
        patch_laz = {}
        for pid, patch_list in patch_store.items():
            laz_bin = self.patch_to_laz_binary(patch_list, pid)
            patch_laz[pid] = laz_bin

        data_to_save = {
            "meta_data": bytes(meta_data),
            "patch_laz": patch_laz,
            "patch_transpose": patch_transpose
        }

        with open(path, "wb") as f:
            pickle.dump(data_to_save, f)

    def load(self, path, frame_id=None):
        """
        Load the indexed data from disk.
        Returns a dictionary with keys:
          - "meta_index": dict mapping frame_id -> list of (patch_id, transposed)
          - "patch_laz": dict mapping patch_id -> LAZ binary block (bytes)
          - "patch_transpose": dict mapping patch_id -> transposed flag
        If frame_id is provided, only the meta information for that frame is returned.
        """
        with open(path, "rb") as f:
            data_loaded = pickle.load(f)

        meta_data = data_loaded["meta_data"]
        patch_laz = data_loaded["patch_laz"]
        patch_transpose = data_loaded["patch_transpose"]

        meta_index = {}
        offset = 0
        num_frames = struct.unpack_from("I", meta_data, offset)[0]
        offset += 4
        for _ in range(num_frames):
            fid = struct.unpack_from("I", meta_data, offset)[0]
            offset += 4
            num_patches = struct.unpack_from("I", meta_data, offset)[0]
            offset += 4
            meta_index[fid] = []
            for _ in range(num_patches):
                pid = struct.unpack_from("I", meta_data, offset)[0]
                offset += 4
                flag = struct.unpack_from("B", meta_data, offset)[0]
                offset += 1
                meta_index[fid].append((pid, bool(flag)))

        full_index = {
            "meta_index": meta_index,
            "patch_laz": patch_laz,
            "patch_transpose": patch_transpose
        }
        if frame_id is not None:
            if frame_id in meta_index:
                return {
                    "meta_index": {frame_id: meta_index[frame_id]},
                    "patch_laz": patch_laz,
                    "patch_transpose": patch_transpose
                }
            else:
                return {"meta_index": {}, "patch_laz": {}, "patch_transpose": {}}
        else:
            return full_index

    def decompress_patch(self, laz_binary):
        """
        Decompress a LAZ binary block and return the point cloud data.
        mode=xyz or int
        """
        buffer = BytesIO(laz_binary)
        with laspy.open(buffer, mode="r", laz_backend=LazBackend.Laszip) as reader:
            las = reader.read()
            points = np.vstack([las.x, las.y, las.z, las.intensity]).T
        return points

    def reconstruct_patches(self, patch_laz):
        """
        Reconstruct the patches from the LAZ binary data.
        """
        reconstructed_patches = {}

        laz_bin_xyz = patch_laz["laz_xyz"]

        points = self.decompress_patch(laz_bin_xyz)[:, :3]
        laz_binary_intensities = patch_laz["laz_intensities"]

        intensities = [self.decompress_patch(intensity)[:, 3] for intensity in laz_binary_intensities]

        patches = {
            "xyz": points,
            "intensities": intensities
        }

        return patches

    def reconstruct_patches(self, xyz_data, intensity_data):
        """
        Reconstruct the patches from the given xyz_data and intensity_data.
        If an intensity item is None, assign the corresponding intensity to 0.
        :param xyz_data:
        :param intensity_data:
        :return:  patches
        """
        # Decompress xyz data
        points = self.decompress_patch(xyz_data)[:, :3]

        # Process intensity data
        intensities = [
            self.decompress_patch(intensity)[:, 3] if intensity is not None else np.zeros(points.shape[0],
                                                                                          dtype=np.uint16)
            for intensity in intensity_data
        ]

        # Reconstruct patches
        patches = {
            "xyz": points,
            "intensities": intensities
        }
        return patches

    def save_2(self, indexer, path="index_with_patches.bin"):
        """
        Save the deduplicated patches and meta index to disk in a single file with offsets.
        """
        meta_index = indexer.index
        patch_store = indexer.patch_store
        patch_transpose = indexer.patch_transpose
        patch_anchor = indexer.patch_anchor
        patch_subset = indexer.patch_subset
        intensity_signal = indexer.intensity_signal
        poses = indexer.poses

        # --- Serialize meta information ---
        meta_data = bytearray()
        frame_ids = list(meta_index.keys())
        meta_data += struct.pack("I", len(frame_ids))
        for fid in frame_ids:
            patches = meta_index[fid]
            meta_data += struct.pack("I", fid)
            meta_data += struct.pack("I", len(patches))
            for patch_id in patches:
                meta_data += struct.pack("I", patch_id)

        if path is None:
            total_size = 0
            meta_data_size = (len(meta_data) + 4) / 1024
            total_size += meta_data_size

            for pid, patch_list in patch_store.items():
                size, laz_bin = self.patch_to_laz_binary(patch_list, pid)

                total_size += size

            return total_size

        # For each patch, convert it into a LAZ binary and store its offset and size.
        patch_offsets = {}

        with open(path, "wb") as f:
            total_size = 0

            # Write meta data
            f.write(struct.pack("I", len(meta_data)))
            f.write(meta_data)
            offset = len(meta_data) + 4

            total_size += offset / 1024

            for pid, patch_list in patch_store.items():
                size, laz_bin = self.patch_to_laz_binary(patch_list, pid)
                xyz_data = laz_bin["laz_xyz"]
                intensity_data = laz_bin["laz_intensities"]

                total_size += size

                # Write xyz_data
                xyz_offset = offset
                xyz_size = len(xyz_data)
                f.write(xyz_data)
                offset += xyz_size

                # Write intensity_data one by one
                intensity_offsets = []
                intensity_sizes = []
                for intensity in intensity_data:
                    intensity_offset = offset
                    intensity_size = len(intensity)
                    f.write(intensity)
                    offset += intensity_size

                    intensity_offsets.append(intensity_offset)
                    intensity_sizes.append(intensity_size)

                # Store offsets and sizes for this patch
                patch_offsets[pid] = {
                    "xyz_offset": xyz_offset,
                    "xyz_size": xyz_size,
                    "intensity_offsets": intensity_offsets,
                    "intensity_sizes": intensity_sizes,
                }

            # Write patch offsets and transpose information at the end
            offsets_data = pickle.dumps(patch_offsets)
            transpose_data = pickle.dumps(patch_transpose)
            anchor_data = pickle.dumps(patch_anchor)
            subset_data = pickle.dumps(patch_subset)
            intensity_data = pickle.dumps(intensity_signal)
            poses_data = pickle.dumps(poses)
            f.write(offsets_data)
            f.write(transpose_data)
            f.write(anchor_data)
            f.write(subset_data)
            f.write(intensity_data)
            f.write(poses_data)
            f.write(struct.pack("I", len(offsets_data)))
            f.write(struct.pack("I", len(transpose_data)))
            f.write(struct.pack("I", len(anchor_data)))
            f.write(struct.pack("I", len(subset_data)))
            f.write(struct.pack("I", len(intensity_data)))
            f.write(struct.pack("I", len(poses_data)))
            return total_size

    def load_2(self, path, Tr, frame_id=None):
        """
        Load the indexed data from disk.
        Returns a dictionary with keys:
          - "meta_index": dict mapping frame_id -> list of (patch_id, transposed)
          - "patch_laz": dict mapping patch_id -> LAZ binary block (bytes)
          - "patch_transpose": dict mapping patch_id -> transposed flag
        If frame_id is provided, only the meta information for that frame is returned.
        """

        io_cost = 0  # Track the size of data read from disk

        with open(path, "rb") as f:
            # Read meta data size and meta data
            meta_data_size = struct.unpack("I", f.read(4))[0]
            meta_data = f.read(meta_data_size)
            io_cost += 4
            io_cost += meta_data_size

            # Read patch offsets and transpose information sizes
            f.seek(-24, os.SEEK_END)
            offsets_size = struct.unpack("I", f.read(4))[0]
            transpose_size = struct.unpack("I", f.read(4))[0]
            anchor_size = struct.unpack("I", f.read(4))[0]
            subset_size = struct.unpack("I", f.read(4))[0]
            intensity_size = struct.unpack("I", f.read(4))[0]
            poses_size = struct.unpack("I", f.read(4))[0]

            io_cost += 24

            # Read patch offsets and transpose information
            f.seek(-offsets_size - transpose_size - anchor_size - subset_size - intensity_size - poses_size - 24,
                   os.SEEK_END)
            patch_offsets = pickle.load(f)
            patch_transpose = pickle.load(f)
            patch_anchor = pickle.load(f)
            patch_subset = pickle.load(f)
            intensity_signal = pickle.load(f)
            poses = pickle.load(f)
            io_cost += offsets_size + transpose_size + anchor_size + subset_size + intensity_size + poses_size

        meta_index = {}
        offset = 0
        num_frames = struct.unpack_from("I", meta_data, offset)[0]
        offset += 4
        for _ in range(num_frames):
            fid = struct.unpack_from("I", meta_data, offset)[0]
            offset += 4
            num_patches = struct.unpack_from("I", meta_data, offset)[0]
            offset += 4
            meta_index[fid] = []
            for _ in range(num_patches):
                pid = struct.unpack_from("I", meta_data, offset)[0]
                offset += 4
                meta_index[fid].append(pid)

        # Initialize the Indexer instance
        indexer = Indexer()

        indexer.poses = poses
        indexer.Tr = Tr
        indexer.Tr_inv = np.linalg.inv(Tr)
        indexer.args = self.args

        indexer.patch_transpose = patch_transpose
        indexer.patch_anchor = patch_anchor
        indexer.patch_subset = patch_subset
        indexer.intensity_signal = intensity_signal
        indexer.index = meta_index

        patch_laz = {}
        patch_store = {}

        if frame_id is not None:
            required_patches = set()
            for fid in frame_id:
                if fid in indexer.index:
                    required_patches.update(indexer.index[fid])

            with open(path, "rb") as f:
                for patch_id in required_patches:
                    offsets = patch_offsets[patch_id]

                    # Load xyz_data
                    f.seek(offsets["xyz_offset"])
                    xyz_data = f.read(offsets["xyz_size"])
                    io_cost += offsets["xyz_size"]

                    # Load only required intensities
                    intensity_data = []
                    for offset, size, subset_frame_id in zip(
                            offsets["intensity_offsets"], offsets["intensity_sizes"], patch_subset[patch_id]
                    ):
                        if subset_frame_id in frame_id:
                            f.seek(offset)
                            intensity_data.append(f.read(size))
                            io_cost += size
                        else:
                            intensity_data.append(None)

                    patch_store[patch_id] = self.reconstruct_patches(xyz_data, intensity_data)

            indexer.patch_store = patch_store
            return indexer, io_cost
        else:
            with open(path, "rb") as f:
                for pid, offsets in patch_offsets.items():
                    # Load xyz_data
                    f.seek(offsets["xyz_offset"])
                    xyz_data = f.read(offsets["xyz_size"])
                    io_cost += offsets["xyz_size"]

                    # Load intensity_data
                    intensity_data = []
                    for offset, size in zip(offsets["intensity_offsets"], offsets["intensity_sizes"]):
                        f.seek(offset)
                        intensity_data.append(f.read(size))
                        io_cost += size

                    # Reconstruct patches
                    patch_store[pid] = self.reconstruct_patches(xyz_data, intensity_data)

            indexer.patch_store = patch_store
            return indexer, io_cost
