# DUSTER: A Deduplication Framework for Efficient Point Cloud Storage and Retrieval
optimizing the storage of point cloud, an fast retrieving the point cloud.


## Libraries

```
pip install laspy[laszip]

# install pcdet following https://github.com/open-mmlab/OpenPCDet

pip install pipwin
pipwin install pycuda
pip install faiss-gpu
```

## Datasets

The datasets can be downloaded from:

### SemanticKitti

http://semantic-kitti.org/dataset.html#format

### ONCE
https://once-for-auto-driving.github.io/download.html#downloads


## Running Command

```
# run a benchmark with octree method, and semantic_kitti data, sequence 00
python benchmark_exp.py \
  --method octree \
  --data_path semantic_kitti \
  --sequence_id 00

# run with duster method
python benchmark_exp.py \
--method duster \
--data_path semantic_kitti \
--sequence_id 00 \
--time_budget 0.25
```
