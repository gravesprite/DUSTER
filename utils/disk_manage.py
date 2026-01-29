'''
This is the function that store the indexed PC frames into disk as well as load it back to memory.
'''
from utils.index import Indexer

import struct, heapq, pickle
import numpy as np


class DiskManager:
    def __init__(self):
        pass

    # ----- Helper functions for quantization & delta encoding -----

    def quantize_value(self, x, factor=1000):
        """Convert a float to an integer via quantization."""
        return int(round(x * factor))

    def delta_encode(self, values):
        """
        Given a list of integers, subtract the minimum.
        Returns (base, delta_list) where base = min(values) and
        delta_list = [v - base for v in values].
        """
        if not values:
            return 0, []
        base = min(values)
        delta = [v - base for v in values]
        return base, delta

    def serialize_integers(self, ints):
        """Serialize a list of integers to bytes (each integer 4 bytes)."""
        b = bytearray()
        for v in ints:
            b += struct.pack("i", v)
        return bytes(b)

    # ----- Huffman encoding routines (operating on bytes) -----

    def huffman_encode(self, data):
        """
        Encode the input bytes 'data' using a simple Huffman encoder.
        Returns a tuple (compressed_data, tree_data).
        Uses a tie-breaker counter to avoid type-comparison issues.
        """
        # Build frequency table.
        freq = {}
        for b in data:
            freq[b] = freq.get(b, 0) + 1

        # Build Huffman tree with tie-breaker counter.
        heap = []
        counter = 0
        for byte, f in freq.items():
            counter += 1
            heap.append((f, counter, byte, None, None))
        heapq.heapify(heap)
        while len(heap) > 1:
            f1, count1, byte1, left1, right1 = heapq.heappop(heap)
            f2, count2, byte2, left2, right2 = heapq.heappop(heap)
            counter += 1
            new_node = (f1 + f2, counter, None, (f1, count1, byte1, left1, right1), (f2, count2, byte2, left2, right2))
            heapq.heappush(heap, new_node)
        tree = heap[0]

        # Build code table.
        code_table = {}

        def traverse(node, code):
            frequency, count, byte, left, right = node
            if byte is not None:
                code_table[byte] = code
            else:
                traverse(left, code + "0")
                traverse(right, code + "1")

        traverse(tree, "")

        # Encode data to bitstring.
        encoded_bits = "".join(code_table[b] for b in data)
        padding = (8 - len(encoded_bits) % 8) % 8
        encoded_bits += "0" * padding

        # Pack padding and code table.
        tree_data = struct.pack("B", padding)
        tree_entries = bytearray()
        # Use unsigned short ("H") for number of entries.
        tree_entries += struct.pack("H", len(code_table))
        for byte, code in code_table.items():
            tree_entries += struct.pack("B", byte)
            tree_entries += struct.pack("B", len(code))
            tree_entries += code.encode("utf-8")
        tree_data += tree_entries

        # Convert bitstring to bytes.
        compressed_data = bytearray()
        for i in range(0, len(encoded_bits), 8):
            compressed_data.append(int(encoded_bits[i:i + 8], 2))
        return bytes(compressed_data), bytes(tree_data)

    def huffman_decode(self, compressed_data, tree_data):
        """Decode data using the Huffman tree stored in tree_data."""
        padding = struct.unpack("B", tree_data[:1])[0]
        tree_entries_data = tree_data[1:]
        num_entries = struct.unpack("H", tree_entries_data[:2])[0]
        pos = 2
        code_table = {}
        for _ in range(num_entries):
            byte = struct.unpack("B", tree_entries_data[pos:pos + 1])[0]
            pos += 1
            code_len = struct.unpack("B", tree_entries_data[pos:pos + 1])[0]
            pos += 1
            code = tree_entries_data[pos:pos + code_len].decode("utf-8")
            pos += code_len
            code_table[code] = byte
        bitstring = "".join(format(b, "08b") for b in compressed_data)
        if padding:
            bitstring = bitstring[:-padding]
        decoded_bytes = bytearray()
        current_code = ""
        for bit in bitstring:
            current_code += bit
            if current_code in code_table:
                decoded_bytes.append(code_table[current_code])
                current_code = ""
        return bytes(decoded_bytes)

    # ----- Processing a group (quantize, delta encode, serialize, Huffman encode) -----

    def process_group(self, values, factor=1000):
        """
        Process a list of float numbers:
          1. Quantize to integers.
          2. Delta-encode by subtracting the smallest value.
          3. Serialize the delta values as binary data.
          4. Huffman encode the binary data.
        Returns a dictionary with:
          - "base": the smallest integer (base)
          - "num": number of values
          - "compressed": Huffman-compressed bytes
          - "tree": Huffman tree data (for decoding)
        """
        qvals = [self.quantize_value(v, factor) for v in values]
        base, deltas = self.delta_encode(qvals)
        serialized = self.serialize_integers(deltas)
        comp, tree = self.huffman_encode(serialized)
        return {"base": base, "num": len(deltas), "compressed": comp, "tree": tree}

    # ----- Main save method -----

    def save(self, indexer, path="index.bin", factor=1000):
        """
        Save the indexed multi-frame data in a compact binary form.
        For each patch in the patch_store, we:
          - Extract all xyz values and intensities.
          - Convert floats to integers (quantization) using the given factor.
          - Delta-encode each sequence (subtracting the smallest value).
          - Apply Huffman encoding on the resulting binary data.
        Specifically, we build three groups:
          - One Huffman tree for untransposed patches’ xyz values.
          - One Huffman tree for transposed patches’ xyz values.
          - One Huffman tree for intensity values.
        Also, we serialize meta information (the index) as before.
        Debug prints show the size and base for each group.
        """
        meta_index = indexer.index
        patch_store = indexer.patch_store
        patch_transpose = indexer.patch_transpose

        # Build groups for untransposed xyz, transposed xyz, and intensities.
        untrans_xyz = []
        trans_xyz = []
        intensities = []
        for pid, patch_list in patch_store.items():
            trans_flag = patch_transpose.get(pid, False)
            for subpatch in patch_list:
                # Append the three quantizable coordinates.
                if trans_flag:
                    trans_xyz.extend(subpatch["xyz"])
                else:
                    untrans_xyz.extend(subpatch["xyz"])
                # Append all intensities.
                intensities.extend(subpatch["intensities"])

        # Process each group.
        group_un_xyz = self.process_group(untrans_xyz, factor) if untrans_xyz else {"base": 0, "num": 0,
                                                                                    "compressed": b"", "tree": b""}
        group_trans_xyz = self.process_group(trans_xyz, factor) if trans_xyz else {"base": 0, "num": 0,
                                                                                   "compressed": b"", "tree": b""}
        group_intensity = self.process_group(intensities, factor) if intensities else {"base": 0, "num": 0,
                                                                                       "compressed": b"", "tree": b""}

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

        # Package everything into one dictionary.
        data_to_save = {
            "meta_data": bytes(meta_data),
            "group_un_xyz": group_un_xyz,
            "group_trans_xyz": group_trans_xyz,
            "group_intensity": group_intensity
        }
        with open(path, "wb") as f:
            pickle.dump(data_to_save, f)
        # Debug prints.
        print(f"[DiskManager.save] Data saved to {path}")
        print(f"  Meta data: {len(meta_data)} bytes")
        print(
            f"  Untransposed XYZ: base={group_un_xyz['base']}, num={group_un_xyz['num']}, compressed={len(group_un_xyz['compressed'])} bytes, tree={len(group_un_xyz['tree'])} bytes")
        print(
            f"  Transposed XYZ: base={group_trans_xyz['base']}, num={group_trans_xyz['num']}, compressed={len(group_trans_xyz['compressed'])} bytes, tree={len(group_trans_xyz['tree'])} bytes")
        print(
            f"  Intensity: base={group_intensity['base']}, num={group_intensity['num']}, compressed={len(group_intensity['compressed'])} bytes, tree={len(group_intensity['tree'])} bytes")

    # ----- Main load method -----

    def load(self, path, factor=1000, frame_id=None):
        """
        Load the indexed data from disk.
        This restores:
          - Meta information (the index)
          - For each group (untransposed xyz, transposed xyz, intensities):
            the Huffman‑encoded compressed data, along with the stored base.
        The original integer sequences are recovered by Huffman decoding, then each value has the base added.
        Finally, inverse quantization (dividing by factor) yields floats.
        Returns a dictionary with keys:
          "meta_index", "untrans_xyz", "trans_xyz", "intensities".
        If frame_id is provided, only the meta info for that frame is returned.
        """
        with open(path, "rb") as f:
            data_loaded = pickle.load(f)
        meta_data = data_loaded["meta_data"]
        group_un_xyz = data_loaded["group_un_xyz"]
        group_trans_xyz = data_loaded["group_trans_xyz"]
        group_intensity = data_loaded["group_intensity"]

        # --- Deserialize meta information ---
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

        def decode_group(group):
            if group["num"] == 0:
                return []
            comp = group["compressed"]
            tree = group["tree"]
            decoded_bytes = self.huffman_decode(comp, tree)
            # There should be group["num"] integers, each 4 bytes.
            ints = [struct.unpack("i", decoded_bytes[i:i + 4])[0] for i in range(0, len(decoded_bytes), 4)]
            # Reconstruct original quantized values by adding base.
            qvals = [v + group["base"] for v in ints]
            # Inverse quantization.
            return [v / factor for v in qvals]

        untrans_xyz = decode_group(group_un_xyz)
        trans_xyz = decode_group(group_trans_xyz)
        intensities = decode_group(group_intensity)

        result = {
            "meta_index": meta_index,
            "untrans_xyz": untrans_xyz,
            "trans_xyz": trans_xyz,
            "intensities": intensities
        }
        if frame_id is not None:
            if frame_id in meta_index:
                return {"meta_index": {frame_id: meta_index[frame_id]},
                        "untrans_xyz": untrans_xyz,
                        "trans_xyz": trans_xyz,
                        "intensities": intensities}
            else:
                return {"meta_index": {}, "untrans_xyz": [], "trans_xyz": [], "intensities": []}
        else:
            return result
