import argparse
import glob
import os.path
from pathlib import Path
import time

import math
import json
import copy


try:
    import open3d
    # from visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
except:
    # import mayavi.mlab as mlab
    # from visual_utils import visualize_utils as V

    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


import csv


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    # return the file list of point clouds
    def get_file_list(self):
        return self.sample_file_list

    def get_item(self, file_name):
        if self.ext == '.bin':
            points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(file_name)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': self.sample_file_list.index(file_name),
        }

    def get_item_ind(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default="../models/pv_rcnn_8369.pth", help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    # Add the parameters
    parser.add_argument("--decision_tree_max_depth", type=int, default=10, help='the max depth of the decision tree')
    parser.add_argument("--c_para", type=float, default=2,
                        help='The constant parameter that compute the value of mab')
    parser.add_argument("--detect_radius", type=float, default=75.0, help='The radius of the object detection')
    parser.add_argument("--sampling_method", type=str, default='ma_mab', help='The selection of sampling method')
    parser.add_argument("--predict_method", type=str, default='velocity',
                        help='The predict method, velocity or no-velocity')
    parser.add_argument("--move_distance_ratio", type=float, default=1,
                        help='The distance move of the not appeared object')
    parser.add_argument("--generate_gt", action='store_true', help="whether compute the ground truth result or not")
    parser.add_argument("--budget_ratio", type=float, default=0.1, help='The budget of deep model sampling')
    parser.add_argument("--uniform_sampling_budget_ratio", type=float, default=0.05,
                        help='The budget of deep model sampling')
    parser.add_argument("--sequence_id", type=str, default='01', help='The  id of experiment sequence')

    # The hyperparameters
    parser.add_argument("--reward_number_factor", type=float, default=0.0, help='The factor of vary number reward')

    parser.add_argument("--process_percentage", type=float, default=1.0, help='processing percentage of the dataset')

    args = parser.parse_args()

    # cfg_from_yaml_file(args.cfg_file, cfg)

    return args


