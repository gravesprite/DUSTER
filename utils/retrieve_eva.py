import sys

from pathlib import Path


# pc_storage_optimize_path = Path("/home/user1/jiangneng/pc_storage_optimize").resolve()
# sys.path.append(str(pc_storage_optimize_path))
# sys.path.append("/home/user1/jiangneng/MAST-demo/OpenPCDet-master/pcdet/tools")
#
# sys.path.append("/home/user1/jiangneng/pcdb/OpenPCDet/tools")



import numpy as np
import torch

# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.datasets import DatasetTemplate
# from pcdet.models import build_network, load_data_to_gpu
# from pcdet.utils import common_utils
#
# from pcdet.models import build_network, load_data_to_gpu
# # from pcdet.config import cfg, cfg_from_yaml_file
# from MAST.inference_opt import parse_config
#
# from MAST.inference_opt import DemoDataset
#
# args = parse_config()
#
# cfg_from_yaml_file("/home/user1/jiangneng/pcdb/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml", cfg)
# data_path = "/home/user1/jiangneng/pcdb/OpenPCDet/data/semantic_kitti/dataset/sequences/00"

from pcdet.models import build_network, load_data_to_gpu

def move_to_cpu(dict_obj):
    for key, value in dict_obj.items():
        if isinstance(value, torch.Tensor):
            dict_obj[key] = value.cpu()
    return dict_obj


# def compute_matching_accuracy_with_alignment(pred1_raw, pred2_raw, score_thresh=0.55,
#                                              dist_thresh=1.0, angle_thresh=np.pi/12):  # ~15 degrees
#     def filter_predictions(pred, score_thresh=0.5):
#         mask = pred['pred_scores'] > score_thresh
#         return {
#             'boxes': pred['pred_boxes'][mask],
#             'scores': pred['pred_scores'][mask],
#             'labels': pred['pred_labels'][mask],
#             'indices': torch.arange(len(pred['pred_boxes']))[mask]
#         }
#
#     def angle_diff_rad(a1, a2):
#         return torch.abs(torch.atan2(torch.sin(a1 - a2), torch.cos(a1 - a2)))
#
#     pred1 = filter_predictions(pred1_raw, score_thresh)
#     pred2 = filter_predictions(pred2_raw, score_thresh)
#
#     matched_pairs = []
#     unmatched_pred1 = []
#     used_pred2 = set()
#
#     for i in range(pred1['boxes'].shape[0]):
#         box1 = pred1['boxes'][i]
#         label1 = pred1['labels'][i]
#         xyz1 = box1[:3]
#         angle1 = box1[6]
#
#         match_found = False
#         for j in range(pred2['boxes'].shape[0]):
#             if j in used_pred2:
#                 continue
#
#             box2 = pred2['boxes'][j]
#             label2 = pred2['labels'][j]
#             xyz2 = box2[:3]
#             angle2 = box2[6]
#
#             if label1 != label2:
#                 continue
#
#             dist = torch.norm(xyz1 - xyz2)
#             angle_diff = angle_diff_rad(angle1, angle2)
#
#             if dist < dist_thresh and angle_diff < angle_thresh:
#                 matched_pairs.append((pred1['indices'][i].item(), pred2['indices'][j].item()))
#                 used_pred2.add(j)
#                 match_found = True
#                 break
#
#         if not match_found:
#             unmatched_pred1.append(pred1['indices'][i].item())
#
#     unmatched_pred2 = [pred2['indices'][j].item() for j in range(pred2['boxes'].shape[0]) if j not in used_pred2]
#
#     total = max(pred1['boxes'].shape[0], pred2['boxes'].shape[0])
#     accuracy = len(matched_pairs) / total if total > 0 else 0.0
#
#     return accuracy, matched_pairs, unmatched_pred1, unmatched_pred2

# def compute_matching_accuracy_with_alignment(pred1_raw, pred2_raw, score_thresh=0.5,
#                                              dist_thresh=1.0, angle_thresh=np.pi/12):  # ~15 degrees
#     def filter_predictions(pred):
#         mask = pred['pred_scores'] > score_thresh
#         return {
#             'boxes': pred['pred_boxes'][mask],
#             'scores': pred['pred_scores'][mask],
#             'labels': pred['pred_labels'][mask],
#             'indices': torch.arange(len(pred['pred_boxes']))[mask]
#         }
#
#     def angle_diff_rad(a1, a2):
#         return torch.abs(torch.atan2(torch.sin(a1 - a2), torch.cos(a1 - a2)))
#
#     pred1 = filter_predictions(pred1_raw)
#     pred2 = filter_predictions(pred2_raw)
#
#     matched = []
#     unmatched_pred1 = []
#     used_pred2 = set()
#
#     for i in range(pred1['boxes'].shape[0]):
#         box1 = pred1['boxes'][i]
#         label1 = pred1['labels'][i]
#         idx1 = pred1['indices'][i]
#         xyz1 = box1[:3]
#         angle1 = box1[6]
#
#         match_found = False
#         for j in range(pred2['boxes'].shape[0]):
#             if j in used_pred2:
#                 continue
#
#             box2 = pred2['boxes'][j]
#             label2 = pred2['labels'][j]
#             idx2 = pred2['indices'][j]
#             xyz2 = box2[:3]
#             angle2 = box2[6]
#
#             if label1 != label2:
#                 continue
#
#             dist = torch.norm(xyz1 - xyz2)
#             angle_diff = angle_diff_rad(angle1, angle2)
#
#             if dist < dist_thresh and angle_diff < angle_thresh:
#                 matched.append((
#                     {'index': idx1.item(), 'box': box1, 'label': label1.item()},
#                     {'index': idx2.item(), 'box': box2, 'label': label2.item()}
#                 ))
#                 used_pred2.add(j)
#                 match_found = True
#                 break
#
#         if not match_found:
#             unmatched_pred1.append({
#                 'index': idx1.item(),
#                 'box': box1,
#                 'label': label1.item()
#             })
#
#     unmatched_pred2 = []
#     for j in range(pred2['boxes'].shape[0]):
#         if j not in used_pred2:
#             unmatched_pred2.append({
#                 'index': pred2['indices'][j].item(),
#                 'box': pred2['boxes'][j],
#                 'label': pred2['labels'][j].item()
#             })
#
#     total = max(pred1['boxes'].shape[0], pred2['boxes'].shape[0])
#     accuracy = len(matched) / total if total > 0 else 0.0
#
#     return {
#         'accuracy': accuracy,
#         'matched': matched,
#         'unmatched_pred1': unmatched_pred1,
#         'unmatched_pred2': unmatched_pred2
#     }

# def normalize_angle_2pi(theta):
#     """Normalize angle to [0, 2π)."""
#     return theta % (2 * np.pi)

def normalize_angle_2pi(theta):
    """Normalize angle to [0, π)."""
    return theta % np.pi

def angle_diff_rad(a1, a2):
    """Shortest difference between two angles in [0, 2π)."""
    diff = torch.abs(torch.atan2(torch.sin(a1 - a2), torch.cos(a1 - a2)))
    return diff  # Always in [0, π]

def compute_matching_accuracy_with_alignment(pred1_raw, pred2_raw,
                                             score_thresh=0.45,
                                             dist_thresh=1.5,
                                             angle_thresh=np.pi/10):
    def filter_predictions(pred):
        mask = pred['pred_scores'] > score_thresh
        return {
            'boxes': pred['pred_boxes'][mask],
            'scores': pred['pred_scores'][mask],
            'labels': pred['pred_labels'][mask],
            'indices': torch.arange(len(pred['pred_boxes']))[mask]
        }

    pred1 = filter_predictions(pred1_raw)
    pred2 = filter_predictions(pred2_raw)

    matched = []
    unmatched_pred1 = []
    used_pred2 = set()

    for i in range(pred1['boxes'].shape[0]):
        box1 = pred1['boxes'][i]
        label1 = pred1['labels'][i]
        idx1 = pred1['indices'][i]

        xyz1 = box1[:3]
        l1, w1 = box1[3], box1[4]
        theta1 = normalize_angle_2pi(box1[6])

        match_found = False
        for j in range(pred2['boxes'].shape[0]):
            if j in used_pred2:
                continue

            box2 = pred2['boxes'][j]
            label2 = pred2['labels'][j]
            idx2 = pred2['indices'][j]

            if label1 != label2:
                continue

            xyz2 = box2[:3]
            l2, w2 = box2[3], box2[4]
            theta2 = normalize_angle_2pi(box2[6])

            dist = torch.norm(xyz1 - xyz2)
            angle_diff = angle_diff_rad(theta1, theta2)

            # check box symmetry
            symmetric = torch.abs(l1 - w1) < 1e-2 and torch.abs(l2 - w2) < 1e-2
            angle_ok = angle_diff < angle_thresh or (symmetric and torch.abs(angle_diff - np.pi) < angle_thresh)

            if dist < dist_thresh and angle_ok:
                matched.append((
                    {'index': idx1.item(), 'box': box1, 'label': label1.item()},
                    {'index': idx2.item(), 'box': box2, 'label': label2.item()}
                ))
                used_pred2.add(j)
                match_found = True
                break

        if not match_found:
            unmatched_pred1.append({
                'index': idx1.item(),
                'box': box1,
                'label': label1.item()
            })

    unmatched_pred2 = []
    for j in range(pred2['boxes'].shape[0]):
        if j not in used_pred2:
            unmatched_pred2.append({
                'index': pred2['indices'][j].item(),
                'box': pred2['boxes'][j],
                'label': pred2['labels'][j].item()
            })

    # total = max(pred1['boxes'].shape[0], pred2['boxes'].shape[0])
    total = pred2['boxes'].shape[0]
    accuracy = len(matched) / total if total > 0 else 1.0

    return {
        'accuracy': accuracy,
        'matched': matched,
        'unmatched_pred1': unmatched_pred1,
        'unmatched_pred2': unmatched_pred2
    }

def compute_reverse_matching_accuracy(pred1_raw, pred2_raw,
                                      score_thresh=0.5,
                                      dist_thresh=1.5,
                                      angle_thresh=np.pi/6):  # ~30°
    def normalize_angle_2pi(theta):
        return theta % (2 * np.pi)

    def angle_diff_rad(a1, a2):
        return torch.abs(torch.atan2(torch.sin(a1 - a2), torch.cos(a1 - a2)))

    def filter_predictions(pred):
        mask = pred['pred_scores'] > score_thresh
        return {
            'boxes': pred['pred_boxes'][mask],
            'scores': pred['pred_scores'][mask],
            'labels': pred['pred_labels'][mask],
            'indices': torch.arange(len(pred['pred_boxes']))[mask]
        }

    pred2 = filter_predictions(pred2_raw)  # only filter pred2
    if pred2['boxes'].shape[0] == 0:  # Check if pred2 has no boxes
        return {
            'accuracy': 1.0,
            'matched': [],
            'unmatched_pred2': []
        }
    pred1 = {
        'boxes': pred1_raw['pred_boxes'],
        'labels': pred1_raw['pred_labels'],
        'indices': torch.arange(len(pred1_raw['pred_boxes']))
    }

    matched = []
    unmatched_pred2 = []
    used_pred1 = set()

    for j in range(pred2['boxes'].shape[0]):
        box2 = pred2['boxes'][j]
        label2 = pred2['labels'][j]
        idx2 = pred2['indices'][j]

        xyz2 = box2[:3]
        l2, w2 = box2[3], box2[4]
        theta2 = normalize_angle_2pi(box2[6])

        match_found = False
        for i in range(pred1['boxes'].shape[0]):
            if i in used_pred1:
                continue

            box1 = pred1['boxes'][i]
            label1 = pred1['labels'][i]
            idx1 = pred1['indices'][i]

            if label1 != label2:
                continue

            xyz1 = box1[:3]
            l1, w1 = box1[3], box1[4]
            theta1 = normalize_angle_2pi(box1[6])

            dist = torch.norm(xyz1 - xyz2)
            angle_diff = angle_diff_rad(theta1, theta2)

            # check box symmetry
            symmetric = torch.abs(l1 - w1) < 1e-2 and torch.abs(l2 - w2) < 1e-2
            angle_ok = angle_diff < angle_thresh or (symmetric and torch.abs(angle_diff - np.pi) < angle_thresh)

            if dist < dist_thresh and angle_ok:
                matched.append((
                    {'index': idx1.item(), 'box': box1, 'label': label1.item()},
                    {'index': idx2.item(), 'box': box2, 'label': label2.item()}
                ))
                used_pred1.add(i)
                match_found = True
                break

        if not match_found:
            unmatched_pred2.append({
                'index': idx2.item(),
                'box': box2,
                'label': label2.item()
            })

    total = pred2['boxes'].shape[0]
    accuracy = len(matched) / total if total > 0 else 1.0

    return {
        'accuracy': accuracy,
        'matched': matched,
        'unmatched_pred2': unmatched_pred2
    }

class Evaluator:
    def __init__(self, queries, logger, original_data_path, args):

        try:
            import open3d
            # from visual_utils import open3d_vis_utils as V

            OPEN3D_FLAG = True
        except:
            # import mayavi.mlab as mlab
            # from visual_utils import visualize_utils as V
            OPEN3D_FLAG = False

        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils


        # from pcdet.config import cfg, cfg_from_yaml_file
        # from MAST.inference_opt import parse_config
        # from configs.config import parse_config

        from MAST.inference_opt import DemoDataset

        # args = parse_config()

        cfg_from_yaml_file("/home/user1/jiangneng/pcdb/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml", cfg)
        # data_path = "/home/user1/jiangneng/pcdb/OpenPCDet/data/semantic_kitti/dataset/sequences/00/velodyne"
        data_path = original_data_path

        self.queries = queries
        self.logger = logger

        dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path=Path(data_path),
            ext=args.ext,
            logger=logger
        )

        self.dataloader = dataset

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename="/home/user1/jiangneng/pcdb/OpenPCDet/models/pv_rcnn_8369.pth", logger=logger, to_cpu=True)
        model.cuda()
        model.eval()

        self.model = model

    def evaluate(self, index, method='opus'):



        metrics = {
            "random_queries": {
                "accuracy": [],
                "io_cost": [],
                "load_time": []
            },
            "sequential_queries": {
                "io_cost": [],
                "load_time": []
            }
        }

        # Split queries into random and sequential queries
        random_queries = self.queries.get("random", [])
        sequential_queries = self.queries.get("sequential", [])

        # Process random queries
        for query_id in random_queries:
            # Perform random search
            pc_data, io_cost, load_time = index.random_search(query_id)
            # print("pc_id, pc_data:", query_id, pc_data)
            input_dict = {
                'points': pc_data,
                'frame_id': query_id,
            }
            input_dict = self.dataloader.prepare_data(input_dict)

            # # Predict using the model
            data_dict = self.dataloader.collate_batch([input_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)
            pred_dicts = move_to_cpu(pred_dicts[0])

            # print(pred_dicts)

            # Read raw data from the DemoDataset
            raw_data = self.dataloader.get_item_ind(query_id)
            # print(raw_data)
            raw_data = self.dataloader.collate_batch([raw_data])
            load_data_to_gpu(raw_data)
            pred_dicts_raw, _ = self.model.forward(raw_data)
            pred_dicts_raw = move_to_cpu(pred_dicts_raw[0])

            # print(pred_dicts_raw)

            # Compare predictions and calculate accuracy
            # accuracy, matched_pairs, unmatched_pred1, unmatched_pred2 = compute_matching_accuracy_with_alignment(pred_dicts, pred_dicts_raw, score_thresh=0.55)
            compare_result = compute_matching_accuracy_with_alignment(
                pred_dicts, pred_dicts_raw, score_thresh=0.5)
            accuracy, matched_pairs, unmatched_pred1, unmatched_pred2 = compare_result['accuracy'], compare_result['matched'], compare_result['unmatched_pred1'], compare_result['unmatched_pred2']
            compare_result =  compute_reverse_matching_accuracy(
                pred_dicts, pred_dicts_raw, score_thresh=0.5)
            accuracy, matched, un_matched_pred2 = compare_result['accuracy'], compare_result['matched'], compare_result['unmatched_pred2']
            print(accuracy)
            # print(matched_pairs)
            # print(unmatched_pred1)
            # print(unmatched_pred2)

            # Record metrics
            metrics["random_queries"]["accuracy"].append(accuracy)
            metrics["random_queries"]["io_cost"].append(io_cost)
            metrics["random_queries"]["load_time"].append(load_time)

        # Process sequential queries
        for sequential_query in sequential_queries:
            # Perform sequential search
            # print(sequential_query)
            pc_frames, io_cost, load_time = index.sequential_search(sequential_query)

            # Record metrics
            metrics["sequential_queries"]["io_cost"].append(io_cost)
            metrics["sequential_queries"]["load_time"].append(load_time)

        return metrics


if __name__ == '__main__':
    import logging
    from pcdet.utils import common_utils

    logger = common_utils.create_logger("logger.txt")

    queries = None  # Replace with actual queries
    evaluator = Evaluator(queries, logger)
    # evaluator.evaluate(index=0)