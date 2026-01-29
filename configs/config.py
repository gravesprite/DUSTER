import argparse

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
    # parser.add_argument("--detect_radius", type=float, default=75.0, help='The radius of the object detection')
    # parser.add_argument("--sampling_method", type=str, default='ma_mab', help='The selection of sampling method')
    # parser.add_argument("--predict_method", type=str, default='velocity',
    #                     help='The predict method, velocity or no-velocity')
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



    # The parameters for the opus-d
    parser.add_argument("--method", type=str, default='opus-d', help='The storage method')
    parser.add_argument("--deduplicate_intensity", action='store_true', default=False, help="whether include the intensity in the deduplication process")
    parser.add_argument("--intensity_threshold", type=float, default=0.01, help='intensity threshold when deduplication of itnensity enabled')
    parser.add_argument("--distance_threshold", type=float, default=0.03,
                        help='distance threshold when deduplication')
    parser.add_argument("--max_cluster_length", type=int, default=4, help='the max depth of a pc point cluster')
    parser.add_argument("--relative_absolute", type=str, default='both', help='the choice that only consider the relative or absolute deduplication')

    parser.add_argument("--intensity_scale", type=float, default=32767.0, help='intensity scaling factor when applying laszip')

    parser.add_argument("--workload_aware", action='store_true', default=False, help='whether to use workload aware deduplication')

    parser.add_argument("--log_file_dir", type=str, default='log.txt', help='log file directory')
    parser.add_argument("--root_store_path", type=str, default='data', help='root store path')
    parser.add_argument("--root_store_path_wa", type=str, default='data_wa', help='root store path workload aware')
    parser.add_argument("--root_store_path_laz", type=str, default='data_laz', help='root store path of laz baseline')
    parser.add_argument("--root_store_path_octree", type=str, default='data_octree', help='root store path of laz baseline')
    parser.add_argument("--query_path", type=str, default='query', help='root store path')






    args = parser.parse_args()


    return args