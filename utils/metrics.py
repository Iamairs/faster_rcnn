"""
评估指标计算
"""
import torch
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile
import os


def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()

    # 收集预测结果和真实标签
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_idx, (images, targets_batch) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets_batch = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]

            # 获取预测结果
            outputs = model(images)

            predictions.extend(outputs)
            targets.extend(targets_batch)

    # 计算mAP
    evaluator = CocoEvaluator()
    results = evaluator.evaluate(predictions, targets)

    return results


class CocoEvaluator:
    """COCO格式评估器"""

    def __init__(self):
        self.results = []
        self.targets = []

    def evaluate(self, predictions, targets):
        """评估预测结果"""
        # 转换为COCO格式
        coco_predictions = self.convert_to_coco_format(predictions, targets)
        coco_targets = self.convert_targets_to_coco_format(targets)

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_targets, f)
            gt_file = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_predictions, f)
            pred_file = f.name

        try:
            # 使用COCO API评估
            coco_gt = COCO(gt_file)
            coco_pred = coco_gt.loadRes(pred_file)

            coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # 提取评估结果
            results = {
                'map_50_95': coco_eval.stats[0],  # mAP@0.5:0.95
                'map_50': coco_eval.stats[1],  # mAP@0.5
                'map_75': coco_eval.stats[2],  # mAP@0.75
                'map_small': coco_eval.stats[3],  # mAP for small objects
                'map_medium': coco_eval.stats[4],  # mAP for medium objects
                'map_large': coco_eval.stats[5],  # mAP for large objects
                'ar_1': coco_eval.stats[6],  # AR@1
                'ar_10': coco_eval.stats[7],  # AR@10
                'ar_100': coco_eval.stats[8],  # AR@100
                'ar_small': coco_eval.stats[9],  # AR for small objects
                'ar_medium': coco_eval.stats[10],  # AR for medium objects
                'ar_large': coco_eval.stats[11],  # AR for large objects
            }

        finally:
            # 清理临时文件
            os.unlink(gt_file)
            os.unlink(pred_file)

        return results


    def convert_to_coco_format(self, predictions, targets):
        """转换预测结果为COCO格式"""
        coco_predictions = []

        for i, (pred, target) in enumerate(zip(predictions, targets)):
            image_id = target['image_id'].item()

            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()

            # 检查是否有预测结果
            if len(boxes) == 0:
                print(f"Warning: No predictions for image_id {image_id}")
                continue

            for box, score, label in zip(boxes, scores, labels):

                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                coco_predictions.append({
                    'image_id': int(image_id),  # 确保是整数
                    'category_id': int(label),  # 确保是整数
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score)
                })

        return coco_predictions

    def convert_targets_to_coco_format(self, targets):
        """转换真实标签为COCO格式"""
        images = []
        annotations = []
        categories = set()

        annotation_id = 1

        for target in targets:
            image_id = target['image_id'].item()

            images.append({
                'id': image_id,
                'width': 0,  # 默认宽度
                'height': 0,  # 默认高度
                'file_name': f'image_{image_id}'
            })

            boxes = target['boxes'].cpu().numpy()
            labels = target['labels'].cpu().numpy()
            areas = target['area'].cpu().numpy()

            for box, label, area in zip(boxes, labels, areas):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                annotations.append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'area': float(area),
                    'iscrowd': 0
                })

                categories.add(int(label))
                annotation_id += 1

        # 创建类别信息
        category_list = [{'id': cat_id, 'name': f'class_{cat_id}'} for cat_id in sorted(categories)]

        # 创建COCO格式数据结构
        info = {
            "description": "Model Evaluation Dataset",
            "url": "https://example.com",
            "version": "1.0",
            "year": 2023,
            "contributor": "Evaluator",
            "date_created": "2023-10-01"
        }
        licenses = [{
            "url": "https://example.com/license",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }]


        coco_format = {
            'info': info,
            'licenses': licenses,
            'images': images,
            'annotations': annotations,
            'categories': category_list
        }

        return coco_format


    def _get_empty_results(self):
        """返回空的评估结果"""
        return {
            'map_50_95': 0.0,
            'map_50': 0.0,
            'map_75': 0.0,
            'map_small': 0.0,
            'map_medium': 0.0,
            'map_large': 0.0,
            'ar_1': 0.0,
            'ar_10': 0.0,
            'ar_100': 0.0,
            'ar_small': 0.0,
            'ar_medium': 0.0,
            'ar_large': 0.0,
        }


def calculate_iou(box1, box2):
    """计算两个边框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union


def calculate_ap(precisions, recalls):
    """计算平均精度(AP)"""
    # 添加边界点
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # 计算precision的包络
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    # 寻找recall变化的点
    indices = np.where(recalls[1:] != recalls[:-1])[0]

    # 计算面积
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap